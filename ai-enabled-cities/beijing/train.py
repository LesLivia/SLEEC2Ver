"""Independent Beijing acquisition, training and existing-tool verification."""
import argparse
import copy
import hashlib
import io
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
CITY = ROOT.name


def download(url: str, path: Path) -> bytes:
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def prepare(args):
    source = 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
    archive = zipfile.ZipFile(io.BytesIO(download(source, ROOT / 'data' / 'uci.zip')))
    while not any('Dongsi' in n and n.endswith('.csv') for n in archive.namelist()):
        nested = [n for n in archive.namelist() if n.endswith('.zip')]
        assert len(nested) == 1, archive.namelist()
        archive = zipfile.ZipFile(io.BytesIO(archive.read(nested[0])))
    name = next(n for n in archive.namelist() if 'Dongsi' in n and n.endswith('.csv'))
    frame = pd.read_csv(archive.open(name))
    frame.index = pd.to_datetime(frame[['year', 'month', 'day', 'hour']])
    frame = frame.loc[(frame.index >= args.start) & (frame.index < args.end)]
    count = len(frame)
    duplicates = int(frame.index.duplicated().sum())
    assert duplicates == 0
    frame = frame.resample('h').asfreq()
    data = pd.DataFrame(
        {'current': frame['PM2.5'], 'previous': frame['PM2.5'].shift(1), 'temperature': frame.TEMP, 'wind': frame.WSPM,
         'future': frame['PM2.5'].shift(-1)}).dropna()
    assert (data[['current', 'previous', 'wind', 'future']] >= 0).all().all()
    x = data.iloc[:, :4].to_numpy(dtype=np.float32) / np.array([500, 500, 50, 10], dtype=np.float32)
    y = (data.future <= args.threshold).to_numpy(dtype=np.int64)
    return x, y, dict(source=source, member=name, raw_rows=count, duplicates=duplicates, complete_rows=len(data),
                      dropped_rows=count - len(data)), ['pm25Current', 'pm25Previous', 'temperature', 'windSpeed'], [
        'outputIssueAlert', 'outputNoAlert'], [500, 500, 50, 10]


def scores(y, pred):
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average='macro', zero_division=0),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2024-04-01' if CITY == 'chicago' else '2015-01-01')
    parser.add_argument('--end', default='2024-07-01' if CITY == 'chicago' else '2016-01-01')
    parser.add_argument('--threshold', type=float, default=150)
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x, y, acquisition, names, events, scales = prepare(args)
    a, b = int(len(x) * .7), int(len(x) * .85)
    # Purge two rows between partitions so adjacent lag/target windows cannot overlap.
    partitions = [(0, a), (a + 2, b), (b + 2, len(x))]
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(),
                                torch.nn.Linear(16, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    best, state, stale = float('inf'), None, 0
    epoch = 0
    for epoch in range(150):
        for indices in torch.randperm(a).split(128):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val = torch.nn.functional.cross_entropy(model(tx[a + 2:b]), ty[a + 2:b]).item()
        if val < best - 1e-5:
            best, state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 20:
            break
    assert state is not None
    model.load_state_dict(state)
    model.eval()
    path = ROOT / 'models' / f'{CITY}.onnx'
    torch.onnx.export(model, (torch.zeros(1, 4),), str(path), input_names=['observation'],
                      output_names=['action_logits'], opset_version=11, dynamo=False)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == [1, 4] and session.get_outputs()[0].shape == [1, 2]
    with torch.no_grad():
        logits = model(tx[b + 2:]).numpy()
    runtime = np.concatenate([session.run(None, {'observation': row[None]})[0] for row in x[b + 2:]])
    deviation = float(np.max(np.abs(runtime - logits)))
    assert np.allclose(runtime, logits, atol=1e-5, rtol=1e-5)
    lo, hi = np.floor(x.min(axis=0) * 100) / 100, np.ceil(x.max(axis=0) * 100) / 100
    if CITY == 'chicago':
        low = np.quantile(x[:a, :2], .25, axis=0)
        high = np.quantile(x[:a, :2], .5, axis=0)
        conditions = [f'({names[0]} < {low[0]:.8f}) and ({names[1]} > {high[1]:.8f})',
                      f'({names[1]} < {low[1]:.8f}) and ({names[0]} > {high[0]:.8f})']
        masks = [(x[:, 0] < low[0]) & (x[:, 1] > high[1]), (x[:, 1] < low[1]) & (x[:, 0] > high[0])]
    else:
        conditions = [f'pm25Current > {args.threshold / 500:.8f}']
        masks = [x[:, 0] > args.threshold / 500]
    conditions = ["(" + c + ")" for c in conditions]
    assert all(mask.sum() > 0 for mask in masks)
    spec = 'def_start\n event cmdRequired\n'
    for name, lower, upper in zip(names, lo, hi):
        spec += f' measure {name}: numeric\n constant {name}LB = {lower:.8f}\n constant {name}UB = {upper:.8f}\n'
    spec += ''.join(f' event {event}\n' for event in events) + 'def_end\nrule_start\n'
    spec += ''.join(f'r{i + 1} when cmdRequired and {condition} then {events[i]}\n' for i, condition in
                    enumerate(conditions)) + 'rule_end\n'
    (ROOT / 'requirements' / f'{CITY}.sleec').write_text(spec)
    sys.path.insert(0, str(ROOT.parent.parent))
    from webapp import core
    generator, _, module = core.generate_module(spec, ROOT / 'results')
    assert list(generator.measures) == names and list(generator.outputs) == events
    compatibility = dict(onnx_checker='passed', shapes=[[1, 4], [1, 2]], max_onnx_deviation=deviation, parser='passed',
                         ordering='passed', preconditions='satisfiable: observed witnesses',
                         rule_trigger_counts=[int(m.sum()) for m in masks],
                         test_trigger_counts=[int(m[b + 2:].sum()) for m in masks])
    try:
        from maraboupy import Marabou
        Marabou.read_onnx(str(path))
        compatibility['marabou_load'] = 'passed'
        answer = core.solve_module(module, path)
        (ROOT / 'results' / 'verification.json').write_text(json.dumps(answer, default=str, indent=2))
        compatibility['verification'] = 'executed; see verification.json'
    except (ImportError, RuntimeError, OSError) as error:
        compatibility['verification'] = f'not executed: {type(error).__name__}: {error}'
    metrics = dict(acquisition=acquisition,
                   splits=[dict(size=j - i, classes=np.bincount(y[i:j], minlength=2).tolist()) for i, j in partitions],
                   test=scores(y[b + 2:], logits.argmax(1)),
                   majority_baseline=scores(y[b + 2:], np.full(len(y) - b - 2, np.bincount(y[:a]).argmax())),
                   compatibility=compatibility)
    metadata = dict(interval=vars(args), seed=42, architecture=[4, 16, 16, 2], epochs=epoch + 1, optimizer='Adam',
                    learning_rate=.003, batch_size=128, early_stopping_patience=20,
                    inputs=[dict(index=i, measure=n, scale=s, lower=float(l), upper=float(h)) for i, (n, s, l, h) in
                            enumerate(zip(names, scales, lo, hi))], outputs=events,
                    domain='full observed feature ranges rounded outward to 0.01; Cartesian product',
                    test_domain_fraction=float(((x[b + 2:] >= lo) & (x[b + 2:] <= hi)).all(1).mean()),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    versions={m.__name__: m.__version__ for m in [torch, np, pd, onnx, ort]},
                    data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'data').iterdir()})
    for filename, value in [('metrics.json', metrics), ('model_metadata.json', metadata)]:
        (ROOT / 'results' / filename).write_text(json.dumps(value, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
