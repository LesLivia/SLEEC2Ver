# Chicago: standardized deterioration priority

Source: [official Chicago Traffic Tracker](https://data.cityofchicago.org/d/kf7e-cur8), API `resource/kf7e-cur8.json`, Austin (9) and Chicago Loop (13). This illustrative learned allocation rule is not a municipal intervention policy.

## Acquisition audit

The original April–June 2024 query returned 5,586 records, all between June 11 11:01 and June 30 23:50. Monthly API counts show a gap after March 3 until June 11 for these regions, including no April/May records. This was not a pagination cutoff. The original resampling spanned only 469 hours; 82 windows were lost to missing/invalid data or boundary lags, leaving 387. Full requested calendar coverage would be 2,184 hours. See results/source_coverage.json for the exact audit query and response.

The revised default is July 1–November 1, 2024 (end exclusive). Four pages contain 10,000/10,000/10,000/5,328 records, matching an independent API count of 35,328. The script asserts count equality. There are no duplicate region/timestamp keys. It excludes 2,142 nonpositive, missing or >80 mph readings; zero is not assumed to mean standstill. Each region is averaged independently by floored calendar hour before alignment onto the full 2,952-hour grid. Thus differences in minute/second timestamps cannot cause false mismatches. No interpolation is performed. Timestamps use the source's local, timezone-naive representation.

Austin lacks valid speed in 252 hours, Loop in 5; 2,700 hours have both. Constructing lag/current/future windows on the complete grid before dropping missing values leaves **2,450 samples**, discarding 502 candidate windows including boundaries. No gap is bridged. Two rows are purged at each chronological split boundary. The audit is regenerated in results/data_audit.json.

## Revised target

For each region r, define the following-hour speed drop Δr = speed_r(t) − speed_r(t+1), positive for deterioration. Fit μr and σr (population standard deviation) on the first 70% of complete windows only. Define standardized deterioration dr = (Δr − μr)/σr. Class 0 prioritizes Austin when dAustin ≥ dLoop; class 1 prioritizes Loop otherwise. Exact ties select Austin (none in this run). When both regions improve, the relatively worse standardized change still receives priority; this forced binary choice is an experimental limitation.

Training baselines in mph: mean drops [-0.1454833, -0.0270457], standard deviations [1.1605690, 1.6912781], Austin then Loop. This defines deterioration relative to historical hourly changes, rather than a deficit relative to average speed. Validation/test future speeds are used only to label their respective examples; they never fit the baseline or enter network inputs. No resampling, class weighting or artificial balancing is used.

Inputs remain speedAustin, speedLoop, speedAustinPrevious, speedLoopPrevious, in this order; all mph divided by 50. Multiply by 50 to recover physical speeds. Outputs remain outputPrioritizeAustin and outputPrioritizeLoop, selected by argmax. The float32 4–16–16–2 ReLU MLP, opset 11 and fixed [1,4]/[1,2] interface are unchanged. See ../README.md for deterministic training and evaluation protocol, results/metrics.json for all split/class counts and metrics, and results/model_metadata.json for exact baseline, bounds, versions and hashes.

## Normative specification and verification

The two geographic obligations still prioritize a region experiencing relatively severe current congestion (below its own training 25th percentile) when the other is above its own training median. These obligations concern present congestion, independently of the predictive objective of future deterioration. Exact normalized thresholds appear in requirements/chicago.sleec; trigger counts appear in metrics.json. Region-relative categories do not imply equal absolute mph or prove equity outcomes.

Bounds are full observed feature ranges rounded outward to 0.01 normalized units. Their Cartesian product includes unobserved combinations. ONNX validation, runtime equivalence, parsing, ordering and Marabou loading are rerun. Existing Marabou disjunction-interface incompatibility prevents a compliance verdict; see run.log. No verifier code was changed.

Reproduce from repository root:

```sh
PYTHONPATH=ai-enabled-cities/.deps PYTHONDONTWRITEBYTECODE=1 python -B ai-enabled-cities/chicago/train.py --start 2024-07-01 --end 2024-11-01
```

Scripts reject fewer than 2,000 complete windows or a single-class training set. Start the existing UI with `PYTHONDONTWRITEBYTECODE=1 python -B -m uvicorn webapp.app:app --port 8000`, upload `ai-enabled-cities/chicago/models/chicago.onnx` and `ai-enabled-cities/chicago/requirements/chicago.sleec`, and select exclusive/argmax semantics. Predictive performance is not normative compliance.

## Working SLEEC-PATROL compatibility specification

Upload models/chicago.onnx with requirements/chicago_compatibility.sleec using exclusive/argmax semantics. This pair successfully runs the existing solver and report builder. The compatibility file expresses each binary decision obligation as prohibition of the other output, avoiding the installed Marabou 1.0.0 disjunction interface mismatch. Original specs remain available. See ../README.md for rationale and the repository-root command to reproduce both verification runs. Results and replayed witnesses are in results/compatibility/; VIOLATED is a completed verification result, not an execution failure.
