"""Independent Chicago acquisition, training and existing-tool verification."""
import argparse
import copy
import hashlib
import json
import sys
import urllib.parse
import urllib.request
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
    source = 'https://data.cityofchicago.org/resource/kf7e-cur8.json'
    where = f"time >= '{args.start}T00:00:00' AND time < '{args.end}T00:00:00' AND region_id in ('9','13')"

    def fetch(query, suffix):
        return json.loads(download(source + '?' + urllib.parse.urlencode(query),
                                   ROOT / 'data' / f'{args.start}_{args.end}_{suffix}.json'))

    expected = int(fetch({'$select': 'count(*) AS n', '$where': where}, 'count')[0]['n'])
    rows, pages = [], []
    for offset in range(0, expected, 10000):
        page = fetch(
            {'$select': 'time,region_id,speed,region', '$where': where, '$order': 'time,region_id', '$limit': 10000,
             '$offset': offset}, f'page_{offset}')
        rows.extend(page)
        pages.append(len(page))
    assert len(rows) == expected, (len(rows), expected)
    frame = pd.DataFrame(rows)
    frame['time'] = pd.to_datetime(frame['time'])
    frame['speed'] = pd.to_numeric(frame['speed'], errors='coerce')
    duplicates = int(frame.duplicated(['time', 'region_id']).sum())
    invalid = int((~frame.speed.between(0.000001, 80)).sum())
    raw_counts = frame.groupby([frame.time.dt.strftime('%Y-%m'), 'region_id']).size()
    frame = frame.drop_duplicates(['time', 'region_id'])
    frame.loc[~frame.speed.between(0.000001, 80), 'speed'] = np.nan
    grid = pd.date_range(args.start, args.end, freq='h', inclusive='left')
    # Aggregate each region independently before aligning: no exact timestamp match required.
    hourly = frame.groupby([frame.time.dt.floor('h'), 'region_id']).speed.mean().unstack().reindex(index=grid,
                                                                                                   columns=['9', '13'])
    windows = pd.concat([hourly, hourly.shift(1).add_suffix('_previous'), hourly.shift(-1).add_suffix('_next')], axis=1)
    data = windows.dropna()
    x = data.iloc[:, :4].to_numpy(dtype=np.float32) / 50
    a = int(len(data) * .7)
    drops = data[['9', '13']].to_numpy() - data[['9_next', '13_next']].to_numpy()
    mean, std = drops[:a].mean(axis=0), drops[:a].std(axis=0, ddof=0)
    assert (std > 0).all()
    deterioration = (drops - mean) / std
    y = (deterioration[:, 1] > deterioration[:, 0]).astype(np.int64)
    assert len(data) >= 2000, f'Only {len(data)} windows; select a longer interval'
    assert np.bincount(y[:a], minlength=2).min() > 0
    audit = dict(source=source, api_count=expected, page_sizes=pages, raw_rows=len(rows),
                 first_record=str(frame.time.min()), last_record=str(frame.time.max()),
                 monthly_region_counts={str(k): int(v) for k, v in raw_counts.items()}, duplicates=duplicates,
                 invalid_or_zero=invalid, hourly_rows=len(hourly),
                 missing_hours_per_region=hourly.isna().sum().to_dict(),
                 synchronized_current_hours=int(hourly.notna().all(axis=1).sum()), complete_rows=len(data),
                 discarded_windows=int(windows.isna().any(axis=1).sum()), split_boundary_purged=4, target=dict(
            formula='d_r=(speed_r(t)-speed_r(t+1)-mean_drop_r)/std_drop_r; class 0 if d_Austin >= d_Loop, else 1',
            baseline_fit='first 70% complete windows only', mean_drop_mph=mean.tolist(), std_drop_mph=std.tolist(),
            std_ddof=0, exact_ties=int((deterioration[:, 0] == deterioration[:, 1]).sum())))
    (ROOT / 'results' / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    return x, y, audit, ['speedAustin', 'speedLoop', 'speedAustinPrevious', 'speedLoopPrevious'], [
        'outputPrioritizeAustin', 'outputPrioritizeLoop'], [50, 50, 50, 50]


def scores(y, pred):
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average='macro', zero_division=0),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2024-07-01')
    parser.add_argument('--end', default='2024-11-01')
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
    low = np.quantile(x[:a, :2], .25, axis=0)
    high = np.quantile(x[:a, :2], .5, axis=0)
    conditions = [f'({names[0]} < {low[0]:.8f}) and ({names[1]} > {high[1]:.8f})',
                  f'({names[1]} < {low[1]:.8f}) and ({names[0]} > {high[0]:.8f})']
    masks = [np.logical_and(x[:, 0] < low[0], x[:, 1] > high[1]), np.logical_and(x[:, 1] < low[1], x[:, 0] > high[0])]
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
        compatibility['verification'] = f'attempted but failed: {type(error).__name__}: {error}'
    metrics = dict(acquisition=acquisition,
                   splits=[dict(size=j - i, classes=np.bincount(y[i:j], minlength=2).tolist()) for i, j in partitions],
                   test=scores(y[b + 2:], logits.argmax(1)),
                   majority_baseline=scores(y[b + 2:], np.full(len(y) - b - 2, np.bincount(y[:a]).argmax())),
                   compatibility=compatibility)
    metadata = dict(target=acquisition['target'], interval=vars(args), seed=42, architecture=[4, 16, 16, 2],
                    epochs=epoch + 1, optimizer='Adam', learning_rate=.003, batch_size=128, early_stopping_patience=20,
                    inputs=[dict(index=i, measure=n, scale=s, lower=float(l), upper=float(h)) for i, (n, s, l, h) in
                            enumerate(zip(names, scales, lo, hi))], outputs=events,
                    domain='full observed feature ranges rounded outward to 0.01; Cartesian product',
                    test_domain_fraction=float(((x[b + 2:] >= lo) & (x[b + 2:] <= hi)).all(1).mean()),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    versions={m.__name__: m.__version__ for m in [torch, np, pd, onnx, ort]},
                    data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'data').iterdir()
                                 if p.is_file() and p.suffix == '.json'})
    for filename, value in [('metrics.json', metrics), ('model_metadata.json', metadata)]:
        (ROOT / 'results' / filename).write_text(json.dumps(value, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
