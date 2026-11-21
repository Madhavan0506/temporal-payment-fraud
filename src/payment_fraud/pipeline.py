"""Train, audit, and record one synthetic payment experiment in MLflow."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import pandas as pd
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from . import __version__
from .data import generate_payments
from .features import BASE_FEATURES, GRAPH_FEATURES, build_features
from .model import EvaluationResult, fit_evaluate
from .report import write_report


@dataclass
class PipelineRun:
    run_id: str
    output_dir: Path
    evaluation: EvaluationResult


def _experiment(client, name, artifact_location):
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    try:
        return client.create_experiment(name, artifact_location=artifact_location)
    except MlflowException as error:
        # Concurrent starts may create the same experiment between lookup and create.
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        existing = client.get_experiment_by_name(name)
        if existing is None:
            raise
        return existing.experiment_id


@contextmanager
def _run(client, experiment_id, name, parent=None):
    tags = {"mlflow.runName": name, "data_source": "synthetic"}
    if parent:
        tags["mlflow.parentRunId"] = parent
    run_id = client.create_run(experiment_id, tags=tags).info.run_id
    try:
        yield run_id
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise
    else:
        client.set_terminated(run_id, status="FINISHED")


def _audit(payments, features, evaluation, target_fpr):
    observable = ["timestamp", "sender", "recipient", "amount"]
    pd.testing.assert_frame_equal(features, build_features(payments[observable]), check_exact=True)
    test_start = evaluation.splits.set_index("period").at["test", "start"]
    prefix = payments.loc[payments.timestamp < test_start]
    pd.testing.assert_frame_equal(features.loc[prefix.index], build_features(prefix), check_exact=True)
    if not evaluation.metrics.validation_fpr.le(target_fpr).all():
        raise ValueError("At least one model exceeded its validation false-positive budget.")
    return {"features_ignore_labels_and_ids": True, "pretest_features_match_truncated_history": True,
            "all_validation_fprs_within_budget": True, "prefix_rows_checked": len(prefix)}


def _log_values(client, run_id, parameters, metrics):
    timestamp = int(time.time() * 1000)
    client.log_batch(run_id, params=[Param(key, str(value)) for key, value in parameters.items()],
                     metrics=[Metric(key, float(value), timestamp, 0) for key, value in metrics.items()])


def run_pipeline(output_dir: Path | str, *, seed: int = 42, days: int = 90,
                 transactions_per_day: int = 400, target_fpr: float = 0.01,
                 tracking_uri: str | None = None,
                 experiment_name: str = "temporal-payment-fraud") -> PipelineRun:
    """Create a new run and export its data, models and results under output_dir/run_id."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_location = None
    if tracking_uri is None:
        tracking_dir = output_dir / "tracking"
        tracking_dir.mkdir(exist_ok=True)
        tracking_uri = f"sqlite:///{tracking_dir / 'mlflow.db'}"
        artifact_location = (tracking_dir / "artifacts").as_uri()
    # Some serverless runtimes block reads of the Spark model-registry setting.
    # Selecting the managed URI here keeps experiment tracking independent of Spark.
    registry_uri = "databricks-uc" if tracking_uri == "databricks" else tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    experiment_id = _experiment(client, experiment_name, artifact_location)
    params = {"seed": seed, "days": days, "transactions_per_day": transactions_per_day,
              "target_fpr": target_fpr, "split_by": "ordered unique UTC timestamps",
              "period_shares": "0.2,0.4,0.2,0.2"}
    versions = {name: version(name) for name in ("mlflow-skinny", "numpy", "pandas", "scikit-learn", "joblib", "scipy")}
    versions.update({"payment-fraud": __version__, "python": platform.python_version()})
    with _run(client, experiment_id, f"Synthetic payments · seed {seed}") as run_id:
        run_dir = output_dir / run_id
        model_dir = run_dir / "models"
        model_dir.mkdir(parents=True)
        payments = generate_payments(seed=seed, days=days, transactions_per_day=transactions_per_day)
        features = build_features(payments)
        evaluation = fit_evaluate(payments, features, target_fpr=target_fpr, seed=seed)
        audit = _audit(payments, features, evaluation, target_fpr)
        _log_values(client, run_id, params, {"transactions": len(payments), "fraud_transactions": payments.is_fraud.sum(),
                                             "test_transactions": len(evaluation.predictions)})
        payments.to_csv(run_dir / "payments.csv.gz", index=False)
        evaluation.predictions.to_csv(run_dir / "predictions.csv.gz", index=False)
        evaluation.metrics.to_csv(run_dir / "metrics.csv", index=False)
        evaluation.by_pattern.to_csv(run_dir / "by_pattern.csv", index=False)
        evaluation.splits.to_csv(run_dir / "splits.csv", index=False)

        children = {}
        for row in evaluation.metrics.to_dict("records"):
            name = row.pop("model")
            detector = evaluation.models[name]
            model_file = model_dir / f"{name}.joblib"
            joblib.dump(detector, model_file, compress=3)
            with _run(client, experiment_id, name, parent=run_id) as child_id:
                settings = {f"classifier.{key}": value for key, value in detector.classifier.get_params().items()}
                if detector.anomaly is not None:
                    settings.update({f"anomaly.{key}": value for key, value in detector.anomaly.get_params().items()})
                settings["feature_columns"] = ", ".join(detector.feature_columns)
                _log_values(client, child_id, settings, row)
                client.log_artifact(child_id, str(model_file), artifact_path="models")
            children[name] = child_id
        write_report(run_dir / "report.html", evaluation, seed=seed, days=days, target_fpr=target_fpr, run_id=run_id)
        manifest = {"run_id": run_id, "experiment_id": experiment_id, "child_runs": children,
                    "parameters": params, "versions": versions, "audit": audit,
                    "base_features": BASE_FEATURES, "graph_features": GRAPH_FEATURES}
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        client.log_artifacts(run_id, str(run_dir))
    return PipelineRun(run_id, run_dir, evaluation)
