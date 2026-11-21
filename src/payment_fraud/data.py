"""Reproducible, entirely synthetic payment events and episode-level audit labels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daytime(rng: np.random.Generator, size: int) -> np.ndarray:
    """Most, but not all, transactions occur during waking hours."""
    return np.where(
        rng.random(size) < 0.8,
        rng.integers(7 * 3600, 23 * 3600, size),
        rng.integers(0, 86400, size),
    )


def generate_payments(seed: int = 42, days: int = 90, transactions_per_day: int = 400) -> pd.DataFrame:
    """Generate exact daily counts. Fraud labels and IDs describe synthetic ground truth.

    Ordinary transfers favor a mix of local peers and merchant-like accounts.
    Legitimate collections and payouts overlap in size, amount and timing with
    fraud episodes. All groups draw members from the same account population.
    """
    if days < 1 or transactions_per_day < 12:
        raise ValueError("Use at least one day and 12 transactions per day.")

    rng = np.random.default_rng(seed)
    n_accounts = max(48, round(transactions_per_day * 1.25))
    account_ids = [f"acct-{n:06d}" for n in range(n_accounts)]
    fraud_patterns = ("fan_in", "fan_out", "cycle")
    rows = []
    for day in range(days):
        count_at_start = len(rows)
        day_start = day * 86400
        # Small runs retain enough positives for chronological smoke tests.
        fraud_episodes = max(1, round(transactions_per_day / 400))
        legal_episodes = max(1, round(transactions_per_day / 220))
        for episode in range(fraud_episodes + legal_episodes):
            is_fraud = episode < fraud_episodes
            if is_fraud:
                pattern = fraud_patterns[(day + episode) % len(fraud_patterns)]
                ring_id = f"ring-{day:04d}-{episode:03d}"
            else:
                pattern = str(rng.choice(["group_collection", "group_payout"]))
                ring_id = ""

            max_size = min(11, max(3, transactions_per_day // 5))
            size = int(rng.integers(min(5, max_size), max_size + 1))
            actors = rng.choice(n_accounts, size + 1, replace=False)
            if pattern == "cycle":
                # Closing edges and repeated loops exercise time-ordered flow features.
                loop_size = int(rng.integers(3, min(size, 5) + 1))
                senders = actors[np.arange(size) % loop_size]
                recipients = actors[(np.arange(size) + 1) % loop_size]
            elif pattern in ("fan_in", "group_collection"):
                senders, recipients = actors[1:], np.repeat(actors[0], size)
            else:
                senders, recipients = np.repeat(actors[0], size), actors[1:]

            # Some benign episodes are fast and some fraud episodes are slow.
            mean_gap = rng.uniform(25, 360) if is_fraud else rng.uniform(100, 1500)
            gaps = np.clip(rng.exponential(mean_gap, size - 1), 1, 2400).astype(int)
            offsets = np.r_[0, np.cumsum(gaps)]
            start = min(int(_daytime(rng, 1)[0]), 86399 - int(offsets[-1]))
            amount_center = rng.lognormal(4.1 if is_fraud else 3.7, 0.85)
            amounts = np.clip(amount_center * rng.lognormal(0, 0.35, size), 0.01, 10000)
            for offset, sender, recipient, amount in zip(offsets, senders, recipients, amounts):
                rows.append((day_start + start + int(offset), account_ids[sender], account_ids[recipient],
                             round(float(amount), 2), int(is_fraud), pattern, ring_id))

        remaining = transactions_per_day - (len(rows) - count_at_start)
        times = _daytime(rng, remaining)
        senders = rng.integers(0, n_accounts, remaining)
        recipients = rng.integers(0, n_accounts, remaining)
        choice = rng.random(remaining)
        # Fixed neighborhoods create repeat partners across days.
        local_peer = ((senders // 16) * 16 + rng.integers(0, 16, remaining)) % n_accounts
        merchants = rng.integers(0, max(3, n_accounts // 32), remaining)
        recipients = np.where(choice < 0.25, merchants, np.where(choice < 0.65, local_peer, recipients))
        recipients = np.where(recipients == senders, (recipients + 1) % n_accounts, recipients)
        amounts = np.clip(rng.lognormal(3.6, 1.05, remaining), 0.01, 10000)
        for second, sender, recipient, amount in zip(times, senders, recipients, amounts):
            rows.append((day_start + int(second), account_ids[sender], account_ids[recipient],
                         round(float(amount), 2), 0, "normal", ""))

    payments = pd.DataFrame(rows, columns=[
        "timestamp", "sender", "recipient", "amount", "is_fraud", "pattern", "ring_id",
    ])
    payments["timestamp"] = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(payments["timestamp"], unit="s")
    payments = payments.sort_values("timestamp", kind="stable", ignore_index=True)
    payments.insert(0, "transaction_id", [f"tx-{n:09d}" for n in range(len(payments))])
    return payments
