# Databricks notebook source
# MAGIC %md
# MAGIC # Temporal graph fraud detection for digital payments
# MAGIC
# MAGIC Follow a payment from an account to its recipient. Test whether their recent activity improves fraud detection beyond the amount and time of the payment. The experiment uses synthetic transactions, including ordinary payments, legitimate group collections and payouts, and coordinated fraud episodes.
# MAGIC
# MAGIC This source notebook uses the project wheel. In Databricks Free Edition, add `dist/payment_fraud-1.0.0-py3-none-any.whl` to the notebook environment as described in `databricks/README.md`.

# COMMAND ----------

from importlib.metadata import version
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Serverless Free Edition uses Unity Catalog and restricts Spark configuration lookups.
os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"

import mlflow
from databricks.sdk import WorkspaceClient
from IPython.display import Markdown, display as show_table

from payment_fraud.pipeline import run_pipeline

print(f"Project {version('payment-fraud')} | MLflow {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the experiment
# MAGIC
# MAGIC The defaults generate 36,000 payments across 90 days. The first 20% of unique timestamps form the anomaly reference period, the next 40% train all three classifiers, the next 20% calibrate the alert thresholds, and the last 20% measure performance. Each threshold permits at most one false alert per 100 legitimate validation payments. The rate on later payments can differ.

# COMMAND ----------

seed = 42
days = 90
transactions_per_day = 400
target_fpr = 0.01

workspace_user = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{workspace_user}/temporal-payment-fraud"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log three detectors
# MAGIC
# MAGIC Compare a random forest using amount and time with a forest that also uses rolling graph features. The hybrid model adds an Isolation Forest anomaly score learned from legitimate payments in the reference period. MLflow records a parent run and three child runs in your workspace user folder.

# COMMAND ----------

with TemporaryDirectory(prefix="payment-fraud-run-") as staging_directory:
    result = run_pipeline(
        output_dir=Path(staging_directory) / "results",
        seed=seed,
        days=days,
        transactions_per_day=transactions_per_day,
        target_fpr=target_fpr,
        tracking_uri="databricks",
        experiment_name=experiment_name,
    )

tracking = mlflow.tracking.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
run_info = tracking.get_run(result.run_id).info
print(f"{experiment_name}\nRun {run_info.run_id} | {run_info.status}")
print("MLflow artifacts:", ", ".join(entry.path for entry in tracking.list_artifacts(result.run_id)))
workspace_url = WorkspaceClient().config.host.rstrip("/")
run_url = f"{workspace_url}/#mlflow/experiments/{run_info.experiment_id}/runs/{run_info.run_id}"
show_table(Markdown(f"[Open this run in MLflow]({run_url})"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the holdout
# MAGIC
# MAGIC Check the date boundaries and number of fraudulent payments in each period. Compare recall at the frozen threshold alongside the achieved false-positive rate. Precision measures the fraction of alerts that corresponded to fraud.

# COMMAND ----------

show_table(result.evaluation.splits)
show_table(result.evaluation.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review coordinated activity
# MAGIC
# MAGIC The pattern table counts the fraud payments detected and the rings with at least one detected payment. Below it, the 20 highest-scored hybrid alerts show which methods flagged each payment. Use the parent run's artifacts for the full holdout and report.

# COMMAND ----------

show_table(result.evaluation.by_pattern)
predictions = result.evaluation.predictions
hybrid_alerts = predictions.loc[predictions["hybrid_alert"]]
review_columns = [
    "timestamp", "sender", "recipient", "amount", "pattern", "ring_id", "is_fraud",
    "baseline_alert", "graph_alert", "hybrid_score",
]
show_table(hybrid_alerts.nlargest(20, "hybrid_score")[review_columns])
