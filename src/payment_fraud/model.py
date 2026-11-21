"""Chronological training, false-positive budgeting and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score

from .features import BASE_FEATURES, GRAPH_FEATURES


@dataclass
class FittedDetector:
    """Scorer for causal feature rows produced by build_features."""

    feature_columns: tuple[str, ...]
    classifier: RandomForestClassifier
    anomaly: IsolationForest | None = None
    threshold: float = 0.5

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        values = features.loc[:, list(self.feature_columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Model features must be finite.")
        if self.anomaly is None:
            return values
        # sklearn assigns lower scores to more anomalous observations.
        anomaly_scores = -self.anomaly.score_samples(values)
        return np.column_stack((values, anomaly_scores))

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return fraud risk scores in feature-row order."""
        return self.classifier.predict_proba(self._matrix(features))[:, 1]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the operating threshold frozen on validation data."""
        return self.score(features) >= self.threshold


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    by_pattern: pd.DataFrame
    splits: pd.DataFrame
    models: dict[str, FittedDetector]


def threshold_at_fpr(legitimate_scores: np.ndarray, target_fpr: float) -> float:
    """Lowest achievable cutoff on [0, 1] within the validation FP budget.

    All observations tied at a score receive the same decision. A threshold
    one float above 1 is possible when a score of 1 exceeds the budget.
    """
    scores = np.asarray(legitimate_scores, dtype=np.float64)
    if not np.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Provide at least one legitimate validation score.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Fraud risk scores must be finite and in [0, 1].")
    budget = int(np.floor(scores.size * target_fpr))
    # Floating-point multiplication can round an exact integer down (e.g.
    # 100 * 0.29). Compare the realized rate using the same arithmetic as evaluation.
    if budget < scores.size and (budget + 1) / scores.size <= target_fpr:
        budget += 1
    if budget / scores.size > target_fpr:
        budget -= 1
    if budget == scores.size:
        return 0.0
    # Include a maximum of 'budget' legitimate payments. The next highest
    # score and everything tied at that value must be below the cutoff.
    boundary = scores.size - budget - 1
    excluded_score = np.partition(scores, boundary)[boundary]
    return float(np.nextafter(excluded_score, np.inf))


def _split_indices(timestamps: pd.Series) -> dict[str, pd.Index]:
    if timestamps.isna().any():
        raise ValueError("Payments must have valid timestamps.")
    times = timestamps.sort_values(kind="stable")
    distinct = times.drop_duplicates().reset_index(drop=True)
    count = len(distinct)
    if count < 5:
        raise ValueError("At least five distinct timestamps are required.")
    a, b, c = distinct.iloc[[count // 5, 3 * count // 5, 4 * count // 5]]
    return {
        "reference": times.index[times < a],
        "supervised": times.index[(times >= a) & (times < b)],
        "validation": times.index[(times >= b) & (times < c)],
        "test": times.index[times >= c],
    }


def _pattern_rows(
    payments: pd.DataFrame, alerts: np.ndarray, model_name: str
) -> list[dict]:
    frame = payments.loc[:, ["pattern", "ring_id", "is_fraud"]].copy()
    frame["alert"] = alerts
    rows = []
    for pattern, group in frame.loc[frame.is_fraud == 1].groupby("pattern"):
        ring_rows = group.loc[group.ring_id.notna() & group.ring_id.ne("")]
        ring_hits = ring_rows.groupby("ring_id").alert.any()
        rows.append({
            "model": model_name,
            "pattern": pattern,
            "fraud_transactions": len(group),
            "transactions_detected": int(group.alert.sum()),
            "recall": float(group.alert.mean()),
            "rings": len(ring_hits),
            "rings_detected": int(ring_hits.sum()),
        })
    return rows


def fit_evaluate(
    payments: pd.DataFrame,
    features: pd.DataFrame,
    *,
    target_fpr: float = 0.01,
    seed: int = 42,
) -> EvaluationResult:
    """Fit three predetermined models and evaluate their frozen decisions.

    Split ordered unique timestamps 20/40/20/20. Only legitimate payments
    from the first period fit the novelty detector. The next period trains all
    classifiers. Validation selects the threshold separately for each model.
    Labels, pattern and ring identity are simulator annotations and never
    enter a feature matrix.
    """
    if not payments.index.is_unique or not payments.index.equals(features.index):
        raise ValueError("Payments and features must have the same unique index in order.")
    if not payments["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0 and 1.")
    if not np.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    labels = payments["is_fraud"].astype(int)
    timestamps = pd.to_datetime(payments["timestamp"], utc=True)
    periods = _split_indices(timestamps)
    reference = periods["reference"]
    reference_clean = reference[labels.loc[reference].to_numpy() == 0]
    train, validation, test = (periods[p] for p in ("supervised", "validation", "test"))
    if len(reference_clean) < 2:
        raise ValueError("The reference period needs at least two legitimate payments.")
    for name, indices in (("supervised", train), ("test", test)):
        if labels.loc[indices].nunique() < 2:
            raise ValueError(f"The {name} period needs both fraudulent and legitimate payments.")
    val_clean = labels.loc[validation].to_numpy() == 0
    if not val_clean.any():
        raise ValueError("The validation period needs legitimate payments for calibration.")

    graph_columns = tuple(BASE_FEATURES) + tuple(GRAPH_FEATURES)
    reference_values = features.loc[reference_clean, list(graph_columns)].to_numpy(dtype=float)
    if not np.isfinite(reference_values).all():
        raise ValueError("Reference features must be finite.")
    anomaly = IsolationForest(
        n_estimators=80, max_samples=min(256, len(reference_clean)), random_state=seed, n_jobs=1
    ).fit(reference_values)
    variants = (
        ("baseline", tuple(BASE_FEATURES), None),
        ("graph", graph_columns, None),
        ("hybrid", graph_columns, anomaly),
    )
    models = {}
    predictions = payments.loc[test].copy()
    actual = labels.loc[test].to_numpy() == 1
    pattern_rows = []
    metric_rows = []
    for name, columns, novelty_model in variants:
        classifier = RandomForestClassifier(
            n_estimators=96,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        model = FittedDetector(columns, classifier, novelty_model)
        classifier.fit(model._matrix(features.loc[train]), labels.loc[train].to_numpy())
        validation_scores = model.score(features.loc[validation])
        model.threshold = threshold_at_fpr(validation_scores[val_clean], target_fpr)
        scores = model.score(features.loc[test])
        alerts = scores >= model.threshold
        tp, fp = int((alerts & actual).sum()), int((alerts & ~actual).sum())
        fn, tn = int((~alerts & actual).sum()), int((~alerts & ~actual).sum())
        models[name] = model
        predictions[f"{name}_score"] = scores
        predictions[f"{name}_alert"] = alerts
        metric_rows.append({
            "model": name,
            "threshold": model.threshold,
            "target_fpr": target_fpr,
            "validation_fpr": float(np.mean(validation_scores[val_clean] >= model.threshold)),
            "test_fpr": fp / (fp + tn),
            "test_recall": tp / (tp + fn),
            "test_precision": tp / (tp + fp) if tp + fp else 0.0,
            "test_average_precision": float(average_precision_score(actual, scores)),
            "test_tp": tp,
            "test_fp": fp,
            "test_tn": tn,
            "test_fn": fn,
        })
        pattern_rows.extend(_pattern_rows(payments.loc[test], alerts, name))

    splits = pd.DataFrame([
        {
            "period": name,
            "start": timestamps.loc[indices].min(),
            "end": timestamps.loc[indices].max(),
            "transactions": len(indices),
            "fraud": int(labels.loc[indices].sum()),
        }
        for name, indices in periods.items()
    ])
    return EvaluationResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=predictions,
        by_pattern=pd.DataFrame(pattern_rows),
        splits=splits,
        models=models,
    )
