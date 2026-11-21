"""Replay historical payments and score a new batch with a fitted detector."""
from __future__ import annotations

import pandas as pd

from .features import build_features
from .model import FittedDetector


PAYMENT_FIELDS = ["transaction_id", "timestamp", "sender", "recipient", "amount"]


def score_payments(detector: FittedDetector, payments: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Return scores in input order. Historical and incoming IDs must be disjoint."""
    if payments.empty:
        raise ValueError("Provide at least one new payment to score.")
    rows = pd.concat([history[PAYMENT_FIELDS], payments[PAYMENT_FIELDS]], ignore_index=True)
    if rows.transaction_id.isna().any() or rows.transaction_id.duplicated().any():
        raise ValueError("Each payment needs a unique transaction_id across history and new payments.")
    times = pd.to_datetime(rows.timestamp, utc=True, errors="raise")
    if len(history) and times.iloc[:len(history)].max() > times.iloc[len(history):].min():
        raise ValueError("History must end at or before the earliest new payment.")
    features = build_features(rows).iloc[len(history):]
    scores = detector.score(features)
    result = payments[PAYMENT_FIELDS].copy()
    result["fraud_score"] = scores
    result["alert"] = scores >= detector.threshold
    return result
