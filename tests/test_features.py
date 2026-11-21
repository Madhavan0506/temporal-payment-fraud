"""Regression tests for temporal boundaries, audit-label isolation and generation."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from payment_fraud.data import generate_payments
from payment_fraud.features import BASE_FEATURES, GRAPH_FEATURES, build_features


class PaymentFeatureTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(
                [0, 60, 90, 3600, 3600, 3601, 86400, 86401, 90000], unit="s"),
            "sender": list("AABAAABAB"),
            "recipient": list("BBCBCCABA"),
            "amount": np.arange(1.0, 10.0),
        }, index=np.arange(9) + 30)

    def test_counts_match_independent_window_queries(self):
        data = self.events.sample(frac=1, random_state=10)
        result = build_features(data)
        for row_id, event in data.iterrows():
            for name, delta in [("10m", pd.Timedelta(minutes=10)), ("1h", pd.Timedelta(hours=1)),
                                ("24h", pd.Timedelta(hours=24))]:
                past = data.loc[(data.timestamp >= event.timestamp - delta) & (data.timestamp < event.timestamp)]
                outgoing = past.loc[past.sender == event.sender]
                incoming = past.loc[past.recipient == event.recipient]
                expected = {f"sender_sent_{name}": len(outgoing), f"recipient_received_{name}": len(incoming)}
                if name in ("10m", "1h"):
                    expected[f"sender_received_{name}"] = (past.recipient == event.sender).sum()
                    expected[f"recipient_sent_{name}"] = (past.sender == event.recipient).sum()
                if name in ("1h", "24h"):
                    expected[f"sender_distinct_recipients_{name}"] = outgoing.recipient.nunique()
                    expected[f"recipient_distinct_senders_{name}"] = incoming.sender.nunique()
                if name == "24h":
                    expected["pair_payments_24h"] = (outgoing.recipient == event.recipient).sum()
                    expected["reverse_pair_payments_24h"] = (
                        (past.sender == event.recipient) & (past.recipient == event.sender)).sum()
                for feature, value in expected.items():
                    self.assertEqual(result.at[row_id, feature], value, (row_id, feature))
        self.assertTrue(result.index.equals(data.index))
        self.assertEqual(result.columns.tolist(), BASE_FEATURES + GRAPH_FEATURES)

    def test_tied_events_do_not_see_each_other(self):
        independent = self.events.iloc[[0, 1, 2, 3, 5, 6, 7, 8]]
        before = build_features(independent).loc[[33]]
        together = build_features(self.events.sample(frac=1, random_state=3))
        assert_frame_equal(before, together.loc[[33]])
        assert_frame_equal(build_features(self.events), together.loc[self.events.index])

    def test_altering_or_appending_future_cannot_change_past(self):
        prefix = self.events.iloc[:5]
        changed = self.events.copy()
        changed.loc[changed.index[5:], ["sender", "recipient", "amount"]] = ["Q", "R", 999999]
        assert_frame_equal(build_features(prefix), build_features(changed).loc[prefix.index])

    def test_labels_ids_and_audit_metadata_do_not_enter_features(self):
        payments = generate_payments(days=3, transactions_per_day=24)
        observable = payments[["timestamp", "sender", "recipient", "amount"]]
        corrupted = payments.assign(is_fraud=1-payments.is_fraud, transaction_id="unknown",
                                    pattern="irrelevant", ring_id="irrelevant")
        assert_frame_equal(build_features(corrupted), build_features(observable))

    def test_generation_shape_labels_and_reproducibility(self):
        data = generate_payments(seed=9, days=6, transactions_per_day=64)
        assert_frame_equal(data, generate_payments(seed=9, days=6, transactions_per_day=64))
        self.assertTrue(data.timestamp.is_monotonic_increasing)
        self.assertEqual(str(data.timestamp.dt.tz), "UTC")
        self.assertTrue(data.transaction_id.is_unique)
        self.assertTrue((data.groupby(data.timestamp.dt.date).size() == 64).all())
        self.assertTrue((data.sender != data.recipient).all())
        self.assertTrue((data.amount > 0).all())
        self.assertEqual(set(data.loc[data.is_fraud.eq(1), "pattern"]), {"fan_in", "fan_out", "cycle"})
        legitimate = data.loc[data.is_fraud.eq(0)]
        self.assertTrue(legitimate.pattern.isin(["group_collection", "group_payout"]).any())
        self.assertTrue(legitimate.ring_id.eq("").all())
        self.assertTrue(data.loc[data.is_fraud.eq(1), "ring_id"].ne("").all())
        self.assertTrue(np.isfinite(build_features(data)).all().all())


if __name__ == "__main__":
    unittest.main()
