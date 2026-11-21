# Temporal graph fraud detection for digital payments

This experiment compares three ways to score the same synthetic payment at the moment it arrives. How many fraudulent transfers can each detector catch when its alert threshold is calibrated to flag at most 1% of legitimate validation payments?

The project ran on **Databricks Free Edition** with 36,000 synthetic payments. The temporal graph model detected 88 of 140 fraudulent transfers in the chronological holdout and flagged 67 of 7,061 legitimate transfers. Adding the anomaly score detected 87 fraudulent transfers and flagged 60 legitimate transfers. See the [hosted experiment report](hosted_run/report.html), [Databricks notebook](https://dbc-c0bcb182-ce0e.cloud.databricks.com/editor/notebooks/2015115595514686) and [managed MLflow run](https://dbc-c0bcb182-ce0e.cloud.databricks.com/ml/experiments/2449797490053810/runs/0dd050c153c14d2197db9cb1db39bbf1).

The [local reference run](example_run/report.html) includes a copy of the generated payments, held-out predictions and three fitted detectors for the scoring demo. The hosted and local runtimes used different software versions. For comparisons, use the metrics file and manifest from the same run.

## Run in Databricks Free Edition

Import [the self-contained notebook](databricks/temporal_payment_fraud.ipynb) into your Free Edition workspace and choose **Run all** on serverless compute. It uses the same Python package as the local command and saves its artifacts to workspace MLflow. See the [workspace setup guide](databricks/README.md) for the environment settings and an alternative using the bundled wheel. Free Edition applies fair-use quotas for personal, noncommercial projects. [Databricks Free Edition](https://www.databricks.com/learn/free-edition)

## Run locally

From the unzipped project folder, use Python 3.12 for the versions in the reference run. These commands set up a virtual environment and run the experiment on your CPU.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-local.txt '.[local,test]'
payment-fraud train --output-dir runs
python -m pytest -q
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. The training command prints the MLflow parent run ID and the path to the exported results. Start the local tracking UI from the same project directory, then open the address printed by MLflow.

```sh
mlflow server --backend-store-uri sqlite:///runs/tracking/mlflow.db --host 127.0.0.1 --port 5000
```

Each training run creates one parent and three child runs. The parent stores the full experiment, and each child records one detector's parameters, validation and test metrics, and fitted model. Repeat with a different simulated population using `payment-fraud train --seed 17`.

## Data and models

Each payment is a directed edge from sender to recipient, with a UTC timestamp and an amount. The default generator emits 400 transfers per day for 90 days. It mixes routine payments with legitimate group collections and payouts. Simulated fraud episodes include many accounts paying one recipient, one account paying many recipients, and transfers cycling through a small group. A new episode ID identifies each simulated event group.

| Detector | Information available when a payment arrives |
| --- | --- |
| Transaction baseline | Random forest using the amount and cyclical time-of-day and day-of-week features. |
| Temporal graph | Same classifier plus directed activity counts over 10 minutes, one hour and 24 hours, counts of distinct counterparties, and forward/reverse payment history for the pair. |
| Graph + anomaly | Same graph inputs plus an Isolation Forest score learned from an earlier reference period of legitimate payments. |

## Evaluation and leakage checks

The pipeline sorts distinct timestamps and divides them into four consecutive periods. The first 20% supplies legitimate examples for the anomaly detector. The next 40% trains all three classifiers. Each detector's threshold is fixed on the following 20% using the chosen false-positive budget. The final 20% is the test period. Equal timestamps stay together.

For a payment at time `t`, a window of length `w` reads events from `[t − w, t)`. Payments with the same timestamp cannot see one another. Earlier observed payments can update the graph during validation and test. The models and thresholds stay fixed. Transaction IDs, account IDs, fraud labels, generator patterns and episode IDs are excluded from the model inputs. The graph builder uses account IDs only to reconstruct payment relationships.

Every run repeats feature construction after removing audit columns and after truncating the future, and checks the validation false-positive budget. Regression tests also exercise ties, window boundaries, reloading a model, and changes to held-out labels. The held-out false-positive rate can differ from the 1% validation target. Episode detection means at least one fraudulent payment in that episode triggered an alert.

This is a controlled synthetic benchmark. The generator determines the fraud prevalence and behavior. Confirmed legitimate reference labels are available immediately in the experiment. Applying these results to an actual payment system would require evidence on representative transactions and the operational delay before labels become available.

## Score another batch

Use a model created by a trusted run. The input and history CSVs use the following fields.

| Field | Meaning |
| --- | --- |
| `transaction_id` | Unique ID across both files |
| `timestamp` | ISO 8601 payment time, preferably with UTC offset |
| `sender`, `recipient` | Stable account identifiers |
| `amount` | Positive finite amount in a common currency |

```sh
payment-fraud score --model example_run/models/hybrid.joblib --history example_run/demo/history.csv --payments example_run/demo/payments.csv --output scored.csv
```

The demo contains the first 100 test payments and their preceding day of history. Its `expected.csv` contains the reference scores and decisions. For your own batches, provide the preceding 24 hours of observed payments and all incoming events in the batch. Omitting history gives a cold start. Outputs preserve input order and include `fraud_score` and `alert`. Use the local constraints file to load the bundled scikit-learn models with the reference versions. For a different runtime, train fresh models there.

## Exported files

| File | Contents |
| --- | --- |
| `report.html` | Offline experiment report |
| `payments.csv.gz` | All labeled synthetic transactions |
| `predictions.csv.gz` | All held-out payments, scores and decisions for each detector |
| `metrics.csv`, `by_pattern.csv`, `splits.csv` | Model comparisons, episode detections and chronological split boundaries |
| `models/*.joblib` | Fitted estimators with frozen alert thresholds |
| `manifest.json` | Run IDs, settings, feature definitions, software versions and audit results |

The Python implementation is in `src/payment_fraud/`. Tests are in `tests/`. The packaged wheel in `dist/` can be installed locally or added to a Databricks notebook environment.
