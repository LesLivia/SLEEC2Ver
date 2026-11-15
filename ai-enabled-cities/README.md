# AI-enabled cities

Two independent CPU experiments using real urban observations. Run from the repository root in the existing Python environment, with torch, numpy, pandas, scikit-learn, onnx, textX and Marabou available. Install missing extras locally:

```sh
python -m pip install --target ai-enabled-cities/.deps onnxruntime sympy
PYTHONPATH=ai-enabled-cities/.deps PYTHONDONTWRITEBYTECODE=1 python -B ai-enabled-cities/chicago/train.py
PYTHONPATH=ai-enabled-cities/.deps PYTHONDONTWRITEBYTECODE=1 python -B ai-enabled-cities/beijing/train.py
```

See each city's README and results JSON for the exact versions, dataset hashes, bounds, ordered interfaces and metrics. Raw downloads and local packages are ignored. Each script independently downloads, preprocesses, trains, exports, evaluates, parses and invokes the existing verifier. Network access is required on first run. No shared training framework is used.

Both networks are float32 4–16–16–2 Linear/ReLU MLPs, fixed ONNX input [1,4], output [1,2], opset 11. Output selection is argmax (first index wins exact ties in evaluation); verifier boundary/tie behavior remains its existing default. Training uses seed 42, CPU deterministic operations, Adam, cross entropy, chronological 70/15/15 partitions with two rows purged at boundaries, shuffled training minibatches only, validation early stopping (20 epochs), maximum 150 epochs. No hyperparameter search or manual weight manipulation was performed.

Bounds are observed full-subset extrema rounded outward to 0.01 in normalized coordinates. They describe a retrospective verification domain, not a training transformation fitted to test labels. All test inputs lie inside it. Its Cartesian product contains combinations never observed. Observed rule witnesses prove nonempty preconditions, not compliance.

## Current limitations

Chicago now uses July–October 2024 with 2,450 complete windows and a nondegenerate target comparing next-hour speed drops standardized against each region's training-only historical changes. The original sparse April–June subset was caused by a source coverage gap, documented in chicago/results/source_coverage.json. See chicago/README.md for the corrected audit and target.

Both models pass ONNX checker, PyTorch/ONNX Runtime equivalence (maximum absolute deviation 9.54e-7), parser/generator ordering checks and Marabou loading. Actual SLEEC-PATROL solving was attempted but failed in the installed Marabou Python/native interface: `addDisjunctionConstraint` rejects the supplied `MarabouUtils.Equation` objects at InputQuery. See each run.log for the full traceback. No compliance result is available. No verifier implementation was changed.

IDE working analysis covered both train.py files. Unused imports and state/epoch initialization warnings were fixed; NumPy/ONNX type-inference warnings and existing environment dependency warnings remain. Runtime checks passed as described above.

These are illustrative decision components, not actual municipal policies. Predictive accuracy does not demonstrate normative compliance or equity outcomes.

## Successful end-to-end compatibility run

Use these model/specification pairs with the existing tool (exclusive/argmax semantics):

| Model | Specification |
|---|---|
| chicago/models/chicago.onnx | chicago/requirements/chicago_compatibility.sleec |
| beijing/models/beijing.onnx | beijing/requirements/beijing_compatibility.sleec |

Paths in this table are relative to ai-enabled-cities/. Both pairs now complete the existing parser, interface validation, code generator, Marabou solver and report builder. Chicago's two rules and Beijing's one rule return VIOLATED with counterexamples. These are successful tool executions, not compliance claims. Solver witnesses are replayed against ONNX Runtime, checking logits and selected actions.

The installed Marabou is 1.0.0, whereas environment.yml declares 2.0.0. The positive-action specifications retain their historical interface failure recorded above. The compatibility specifications use the existing supported `then not <other output>` syntax. With exactly two exclusive outputs, prohibiting the other action expresses the same decision obligation; the generated violating inequality and default 0.0001 logit margin match the original query. This avoids the older interface's unsupported disjunction-object conversion. It is specific to binary exclusive outputs, not a general replacement for positive actions in multiclass or independent-output models. Exact output ties retain the existing verifier's tolerance limitations.

Reproduce without retraining or editing SLEEC-PATROL:

```sh
PYTHONPATH=ai-enabled-cities/.deps PYTHONDONTWRITEBYTECODE=1 python -B ai-enabled-cities/verify_compatibility.py
```

The script regenerates the compatibility specs from the originals and invokes `webapp.core` directly. Each city's results/compatibility/ contains the generated module, verification.json (model/spec hashes, solver verdicts, witness replays), and report.json (existing UI report-builder output). Console output is captured in compatibility.log for the recorded run. These reports supersede the earlier failure for the compatibility specifications only. The same pairs can be uploaded in the existing web UI; browser interaction itself was not tested.

IDE working analysis of verify_compatibility.py reports only the existing environment's unsatisfied-package warning; no code diagnostics remain.
