"""Regression checks for operating thresholds, split isolation and scoring."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from payment_fraud.features import BASE_FEATURES, GRAPH_FEATURES
from payment_fraud.model import fit_evaluate, threshold_at_fpr


def test_threshold_respects_ties_and_small_false_positive_budgets():
    scores = np.array([0.2, 0.4, 0.4, 0.4, 0.8])
    threshold = threshold_at_fpr(scores, 0.4)
    assert 0.4 < threshold <= 0.8
    assert (scores >= threshold).sum() == 1
    assert threshold_at_fpr(scores, 0.0) > scores.max()
    assert threshold_at_fpr(scores, 1.0) <= scores.min()
    # The budget of 29 is a whole number even though 0.29 * 100 rounds down.
    uniform = np.linspace(0, 1, 100)
    assert (uniform >= threshold_at_fpr(uniform, 0.29)).sum() == 29
    with pytest.raises(ValueError, match="at least one"):
        threshold_at_fpr(np.array([]), 0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        threshold_at_fpr(scores, -0.1)


def fixture_rows():
    rng = np.random.default_rng(91)
    sequence = np.arange(500)
    # Timestamp ties straddle many potential row-based boundaries.
    timestamps = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(sequence // 3, unit="m")
    fraud = (sequence % 11 == 0)
    payments = pd.DataFrame({
        "transaction_id": sequence,
        "timestamp": timestamps,
        "sender": sequence % 100,
        "recipient": sequence % 20,
        "amount": 10.0 + sequence % 50,
        "is_fraud": fraud.astype(int),
        "pattern": np.where(fraud, "fan_in", "normal"),
        "ring_id": np.where(fraud, "group", ""),
    })
    features = pd.DataFrame({
        key: rng.normal(size=len(payments)) for key in tuple(BASE_FEATURES) + tuple(GRAPH_FEATURES)
    })
    features[GRAPH_FEATURES[0]] += 2 * fraud
    return payments, features


@pytest.fixture(scope="module")
def evaluated():
    payments, features = fixture_rows()
    result = fit_evaluate(payments, features, seed=7, target_fpr=0.1)
    return payments, features, result


def test_periods_are_ordered_and_scores_reproduce_after_serialization(evaluated):
    payments, features, result = evaluated
    splits = result.splits
    assert splits.period.to_list() == ["reference", "supervised", "validation", "test"]
    assert all(a < b for a, b in zip(splits.end.iloc[:-1], splits.start.iloc[1:]))
    assert splits.transactions.sum() == len(payments)
    assert (result.metrics.validation_fpr <= 0.1).all()
    assert set(result.models) == {"baseline", "graph", "hybrid"}
    for row in result.metrics.itertuples():
        test_features = features.loc[result.predictions.index]
        buffer = io.BytesIO()
        joblib.dump(result.models[row.model], buffer)
        buffer.seek(0)
        reloaded = joblib.load(buffer)
        scores = reloaded.score(test_features)
        alerts = reloaded.predict(test_features)
        np.testing.assert_array_equal(scores, result.predictions[f"{row.model}_score"])
        np.testing.assert_array_equal(alerts, result.predictions[f"{row.model}_alert"])
        truth = result.predictions["is_fraud"].to_numpy(dtype=bool)
        assert int((alerts & truth).sum()) == row.test_tp
        assert int((alerts & ~truth).sum()) == row.test_fp
        assert row.test_fp / (row.test_tn + row.test_fp) == pytest.approx(row.test_fpr)
        assert row.test_tp / (row.test_fn + row.test_tp) == pytest.approx(row.test_recall)
        pattern = result.by_pattern.query("model == @row.model").iloc[0]
        assert pattern.transactions_detected == row.test_tp
        assert pattern.rings_detected == int(row.test_tp > 0)


def test_test_labels_and_excluded_reference_rows_cannot_change_scores(evaluated):
    payments, features, first = evaluated
    altered_payments = payments.copy()
    test = first.predictions.index
    altered_payments.loc[test, "is_fraud"] = 1 - altered_payments.loc[test, "is_fraud"]
    altered_features = features.copy()
    reference_end = first.splits.loc[first.splits.period == "reference", "end"].iloc[0]
    excluded = (payments.timestamp <= reference_end) & (payments.is_fraud == 1)
    # These feature rows must not be seen by any trained estimator.
    altered_features.loc[excluded] = 99999.0
    second = fit_evaluate(altered_payments, altered_features, seed=7, target_fpr=0.1)
    for name in first.models:
        assert first.models[name].threshold == second.models[name].threshold
        np.testing.assert_array_equal(
            first.predictions[f"{name}_score"], second.predictions[f"{name}_score"]
        )


def test_rejects_misaligned_input_and_unusable_holdout():
    payments, features = fixture_rows()
    with pytest.raises(ValueError, match="same unique index"):
        fit_evaluate(payments, features.iloc[::-1])
    payments.loc[payments.timestamp >= payments.timestamp.quantile(0.75), "is_fraud"] = 0
    with pytest.raises(ValueError, match="test period needs both"):
        fit_evaluate(payments, features)
