# Beijing pollution alert

Source: [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing), Chen (2017), DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0. Download the public archive once into data/uci.zip; inspect nested ZIP members and select the Dongsi CSV by its actual filename. Select calendar 2015 by default (`--start` inclusive and `--end` exclusive are configurable).

Timestamps come from year/month/day/hour. Duplicate timestamps are rejected. Resample hourly before constructing lags and future labels; discard windows missing any required value. No interpolation or future-value imputation. Assert nonnegative pollution and wind. Exact retained counts and class distributions are in results/metrics.json.

Ordered features and normalization:

| Measure | Raw field/unit | Model input |
|---|---|---|
| pm25Current | PM2.5 at t, µg/m³ | raw / 500 |
| pm25Previous | PM2.5 at t−1, µg/m³ | raw / 500 |
| temperature | TEMP at t, °C | raw / 50 |
| windSpeed | WSPM at t, m/s | raw / 10 |

Class 0 / outputIssueAlert means next-hour PM2.5 >150 µg/m³; class 1 / outputNoAlert means ≤150. `--threshold` configures this illustrative policy parameter. It is not an applicable Beijing legal alert threshold. The normative rule independently requires an alert whenever current PM2.5 >150, normalized 0.3, regardless of the future prediction. There are 1,481 observed triggering windows, including 508 test windows.

The 4–16–16–2 MLP achieves test accuracy 0.95997, balanced accuracy 0.95845, macro-F1 0.95830. Majority accuracy is 0.60047. Confusion matrix, rows true and columns predicted in logit order: [[484,25],[26,739]]. See ../README.md for protocol and results/model_metadata.json for exact bounds and versions. All test points lie within the observed Cartesian domain; unobserved combinations remain possible.

Reproduce from repository root:

```sh
PYTHONPATH=ai-enabled-cities/.deps PYTHONDONTWRITEBYTECODE=1 python -B ai-enabled-cities/beijing/train.py
```

Launch `PYTHONDONTWRITEBYTECODE=1 python -B -m uvicorn webapp.app:app --port 8000`, then upload `ai-enabled-cities/beijing/models/beijing.onnx` and `ai-enabled-cities/beijing/requirements/beijing.sleec` with exclusive/argmax semantics. The script also invokes the existing parser and solver, keeping generated code in results/. Current solver failure is documented in ../README.md and run.log. No compliance verdict is available. A single station/year and an illustrative threshold limit generalization; predictive performance is not evidence of normative compliance.

## Working SLEEC-PATROL compatibility specification

Upload models/beijing.onnx with requirements/beijing_compatibility.sleec using exclusive/argmax semantics. This pair successfully runs the existing solver and report builder. The compatibility file expresses each binary decision obligation as prohibition of the other output, avoiding the installed Marabou 1.0.0 disjunction interface mismatch. Original specs remain available. See ../README.md for rationale and the repository-root command to reproduce both verification runs. Results and replayed witnesses are in results/compatibility/; VIOLATED is a completed verification result, not an execution failure.
