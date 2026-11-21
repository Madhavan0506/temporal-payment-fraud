"""Point-in-time features for a directed, timestamped payment multigraph."""
from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pandas as pd


BASE_FEATURES = ["log_amount", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
GRAPH_FEATURES = [
    "sender_sent_10m", "recipient_received_10m", "sender_received_10m", "recipient_sent_10m",
    "sender_sent_1h", "recipient_received_1h", "sender_received_1h", "recipient_sent_1h",
    "sender_distinct_recipients_1h", "recipient_distinct_senders_1h",
    "sender_sent_24h", "recipient_received_24h",
    "sender_distinct_recipients_24h", "recipient_distinct_senders_24h",
    "pair_payments_24h", "reverse_pair_payments_24h",
]
_HOUR_NS = 3600 * 10**9


class _Window:
    """Rolling directed edge counts, endpoint counts and unique partner counts."""

    def __init__(self, duration_ns: int):
        self.duration_ns = duration_ns
        self.events = deque()
        self.sent = Counter()
        self.received = Counter()
        self.edges = Counter()
        self.distinct_out = Counter()
        self.distinct_in = Counter()

    def add(self, timestamp: int, sender, recipient) -> None:
        pair = (sender, recipient)
        if self.edges[pair] == 0:
            self.distinct_out[sender] += 1
            self.distinct_in[recipient] += 1
        self.sent[sender] += 1
        self.received[recipient] += 1
        self.edges[pair] += 1
        self.events.append((timestamp, sender, recipient))

    @staticmethod
    def _decrement(counter: Counter, key) -> None:
        counter[key] -= 1
        if counter[key] == 0:
            del counter[key]

    def expire(self, timestamp: int) -> None:
        earliest = timestamp - self.duration_ns
        while self.events and self.events[0][0] < earliest:
            _, sender, recipient = self.events.popleft()
            pair = (sender, recipient)
            self._decrement(self.sent, sender)
            self._decrement(self.received, recipient)
            self._decrement(self.edges, pair)
            if self.edges[pair] == 0:
                self._decrement(self.distinct_out, sender)
                self._decrement(self.distinct_in, recipient)


def build_features(payments: pd.DataFrame) -> pd.DataFrame:
    """Return one numeric feature row per input row, in the original index/order.

    At time t, each graph window uses [t - duration, t). Events at the same
    instant are all scored before any become visible. Only sender, recipient,
    timestamp and amount are read. Sorting is O(n log n); each event then
    enters and leaves each window once, with expected O(1) counter updates.
    """
    required = ["timestamp", "sender", "recipient", "amount"]
    missing = set(required).difference(payments.columns)
    if missing:
        raise ValueError(f"Missing payment columns: {sorted(missing)}")
    if payments[required].isna().to_numpy().any():
        raise ValueError("Payment timestamps, endpoints and amounts must be present.")
    timestamps = pd.to_datetime(payments["timestamp"], utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError("Payment timestamps must represent valid instants.")
    amounts = pd.to_numeric(payments["amount"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise ValueError("Amounts must be positive finite numbers.")

    # Force ns units explicitly. Pandas runtimes may infer us or ns resolution.
    times = timestamps.to_numpy(dtype="datetime64[ns]").view("int64")
    day_angle = 2 * np.pi * (times % (24 * _HOUR_NS)) / (24 * _HOUR_NS)
    week_angle = 2 * np.pi * timestamps.dt.dayofweek.to_numpy() / 7
    base = np.column_stack([np.log1p(amounts), np.sin(day_angle), np.cos(day_angle),
                            np.sin(week_angle), np.cos(week_angle)])
    graph = np.empty((len(payments), len(GRAPH_FEATURES)), dtype=float)
    senders = payments["sender"].to_numpy()
    recipients = payments["recipient"].to_numpy()
    order = np.argsort(times, kind="stable")
    ten_minutes, hour, day = _Window(_HOUR_NS // 6), _Window(_HOUR_NS), _Window(24 * _HOUR_NS)
    start = 0
    while start < len(order):
        now = int(times[order[start]])
        end = start + 1
        while end < len(order) and times[order[end]] == now:
            end += 1
        ten_minutes.expire(now)
        hour.expire(now)
        day.expire(now)
        batch = order[start:end]
        for position in batch:
            sender, recipient = senders[position], recipients[position]
            graph[position] = (
                ten_minutes.sent[sender], ten_minutes.received[recipient],
                ten_minutes.received[sender], ten_minutes.sent[recipient],
                hour.sent[sender], hour.received[recipient], hour.received[sender], hour.sent[recipient],
                hour.distinct_out[sender], hour.distinct_in[recipient],
                day.sent[sender], day.received[recipient],
                day.distinct_out[sender], day.distinct_in[recipient],
                day.edges[(sender, recipient)], day.edges[(recipient, sender)],
            )
        for position in batch:
            sender, recipient = senders[position], recipients[position]
            ten_minutes.add(now, sender, recipient)
            hour.add(now, sender, recipient)
            day.add(now, sender, recipient)
        start = end
    return pd.DataFrame(np.column_stack([base, graph]), columns=BASE_FEATURES + GRAPH_FEATURES,
                        index=payments.index)
