"""Command-line training and replay for the payment fraud project."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .pipeline import run_pipeline
from .scoring import PAYMENT_FIELDS, score_payments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Generate payments, train detectors and log an MLflow experiment")
    train.add_argument("--output-dir", type=Path, default=Path("runs"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--days", type=int, default=90)
    train.add_argument("--transactions-per-day", type=int, default=400)
    train.add_argument("--target-fpr", type=float, default=0.01)
    score = commands.add_parser("score", help="Score a CSV using a trusted detector saved by train")
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--payments", type=Path, required=True)
    score.add_argument("--history", type=Path, help="Past payments for the preceding 24 hours")
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        result = run_pipeline(args.output_dir, seed=args.seed, days=args.days,
                              transactions_per_day=args.transactions_per_day, target_fpr=args.target_fpr)
        print(result.evaluation.metrics.to_string(index=False))
        print(f"MLflow parent run {result.run_id}\nResults {result.output_dir}")
    else:
        id_types = {name: str for name in ("transaction_id", "sender", "recipient")}
        new = pd.read_csv(args.payments, dtype=id_types, keep_default_na=False)
        history = (pd.read_csv(args.history, dtype=id_types, keep_default_na=False)
                   if args.history else pd.DataFrame(columns=PAYMENT_FIELDS))
        scored = score_payments(joblib.load(args.model), new, history)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Fail if the destination already exists, including either input CSV.
        scored.to_csv(args.output, index=False, mode="x")
        print(f"Scored {len(scored)} payments. Saved {args.output}")


if __name__ == "__main__":
    main()
