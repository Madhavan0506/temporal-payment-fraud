# Run in Databricks Free Edition

The notebook trains on serverless CPU compute and records its results in your workspace's managed MLflow experiment tracker. Free Edition provides a personal, noncommercial workspace with compute quotas. Compute pauses when the quota is reached. [Free Edition](https://www.databricks.com/learn/free-edition) · [Usage limits](https://learn.microsoft.com/en-us/azure/databricks/getting-started/free-edition-limitations)

## Import the standalone notebook

1. Open your **Free Edition** workspace. Select **Workspace**, open your user folder, and choose **⋮ → Import**. Import `databricks/temporal_payment_fraud.ipynb`. The file includes the project's Python package. [Import a notebook](https://learn.microsoft.com/en-us/azure/databricks/notebooks/notebook-export-import)
2. Open the notebook. Select **Serverless** compute. Under **Environment**, choose the **Standard** base environment. Version **5** includes pandas, NumPy, scikit-learn, joblib and the MLflow client used by this project. If you change the environment, select **Apply**. [Serverless environment controls](https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies) · [Version 5 package inventory](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/five)
3. Select **Run all**. The configuration cell sets the seed, simulation size and target false-positive rate. The training cell generates payments, trains the detectors and logs a new run under `/Users/<your-workspace-user>/temporal-payment-fraud`. If the notebook asks for a classic cluster or offers a paid trial, return to the Free Edition workspace before running.
4. Open **Open this run in MLflow** in the training cell's output. The parent run holds the report and result files. Its three child runs record the individual models' results. The notebook also displays split dates, held-out metrics, fraud-pattern breakdowns and sample alerts. [MLflow tracking in Databricks](https://learn.microsoft.com/en-us/azure/databricks/mlflow/tracking)

## Use the source notebook and wheel

For separate package and notebook updates, import `dist/payment_fraud-1.0.0-py3-none-any.whl` and `databricks/temporal_payment_fraud.py` into your workspace folder. Copy the wheel's workspace path. Open the source notebook, choose the same serverless environment, and add the full path under **Environment → Dependencies**. It will look like `/Workspace/Users/your-email/payment-fraud/payment_fraud-1.0.0-py3-none-any.whl`. Select **Apply**, then **Run all**. [Upload a workspace library](https://learn.microsoft.com/en-us/azure/databricks/libraries/workspace-files-libraries)
