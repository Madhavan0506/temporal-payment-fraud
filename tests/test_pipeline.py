"""Integration checks for MLflow persistence and scoring raw payment records."""
import joblib
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from payment_fraud.pipeline import run_pipeline
from payment_fraud.scoring import PAYMENT_FIELDS, score_payments


def test_logged_run_and_replayed_scores(tmp_path):
    run = run_pipeline(tmp_path, days=30, transactions_per_day=50, seed=7)
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking' / 'mlflow.db'}")
    parent = client.get_run(run.run_id)
    assert parent.info.status == "FINISHED"
    children = client.search_runs([parent.info.experiment_id], f"tags.`mlflow.parentRunId` = '{run.run_id}'")
    assert len(children) == 3
    assert all(child.info.status == "FINISHED" for child in children)
    assert all("test_recall" in child.data.metrics for child in children)
    assert (run.output_dir / "report.html").is_file()
    assert (run.output_dir / "manifest.json").is_file()

    all_payments = pd.read_csv(run.output_dir / 'payments.csv.gz', keep_default_na=False)
    held_out = run.evaluation.predictions
    history = all_payments[~all_payments.transaction_id.isin(held_out.transaction_id)]
    raw = held_out[PAYMENT_FIELDS].sample(frac=1, random_state=42)
    detector = joblib.load(run.output_dir / 'models' / 'hybrid.joblib')
    replay = score_payments(detector, raw, history)
    expected = held_out.set_index('transaction_id').loc[replay.transaction_id]
    assert raw.index.equals(replay.index)
    np.testing.assert_array_equal(replay.alert, expected.hybrid_alert)
    np.testing.assert_allclose(replay.fraud_score, expected.hybrid_score, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match='unique transaction_id'):
        score_payments(detector, raw, all_payments)
