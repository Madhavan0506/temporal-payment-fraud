"""Write a standalone, offline-readable report for a completed experiment."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import EvaluationResult


_MODEL_NAMES = {"baseline": "Transaction only", "graph": "Transaction + graph", "hybrid": "Graph + anomaly"}
_PATTERN_NAMES = {"fan_in": "Many senders to one account", "fan_out": "One account to many recipients",
                  "cycle": "Circular transfers"}
_PERIOD_NAMES = {"reference": "Anomaly reference", "supervised": "Classifier training",
                 "validation": "Threshold calibration", "test": "Final holdout"}


def _table(headers, rows, label):
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = []
    for first, *rest in rows:
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in rest)
        body.append(f'<tr><th scope="row">{escape(str(first))}</th>{cells}</tr>')
    return (f'<div class="table-wrap" tabindex="0" role="region" aria-label="{escape(label)}">'
            f'<table aria-label="{escape(label)}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


_STYLE = """
:root{color-scheme:light;--ink:#102e39;--muted:#415d66;--rule:#c5d4d4;--teal:#00685c}
*{box-sizing:border-box}
body{margin:0;background:#fafcf9;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1020px;margin:auto;padding:48px 40px 30px}
header{border-top:6px solid var(--teal);padding:24px 0 20px}
h1,h2{font-family:Georgia,serif;font-weight:normal;line-height:1.12;margin:0}
h1{font-size:clamp(2rem,4.8vw,3.5rem);max-width:22ch}
h2{font-size:1.7rem;margin-bottom:10px}
p{max-width:82ch;margin:10px 0;color:var(--muted)}
section{margin:30px 0;padding-top:24px;border-top:1px solid var(--rule)}
.summary{font-size:1.1rem;color:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);background:#eaf3ef;border:1px solid var(--rule)}
.stat{padding:16px 20px}.stat+.stat{border-left:1px solid var(--rule)}
.stat strong{display:block;font-size:1.6rem;color:var(--teal);line-height:1.25}
.stat span{color:var(--muted);font-size:.9rem}
.table-wrap{overflow-x:auto;margin:15px 0;outline-offset:3px}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:right;vertical-align:top}
th:first-child{text-align:left}tr>*{border-bottom:1px solid var(--rule)}
thead th{color:#2f535a;background:#e8f1ed;vertical-align:bottom;line-height:1.25}
tbody th{font-weight:600}td{white-space:nowrap}
li{margin:6px 0}details{margin:12px 0}summary{cursor:pointer;color:var(--teal);font-weight:600}
footer{border-top:1px solid var(--rule);margin-top:32px;padding-top:12px;font-size:.85rem;color:var(--muted)}
code{overflow-wrap:anywhere}
@media(max-width:640px){main{padding:20px 14px}.stat{padding:12px 8px}.stat strong{font-size:1.25rem}
.stat span{font-size:.8rem}th,td{padding:8px}h2{font-size:1.45rem}}
@media print{main{padding:0}.table-wrap{overflow:visible}tr,.stats{break-inside:avoid}body{background:white}}
"""


def write_report(output_path: Path, evaluation: EvaluationResult, *, seed: int, days: int,
                 target_fpr: float, run_id: str) -> None:
    """Write the synthetic holdout outcomes after the pipeline's audit checks pass."""
    metrics = evaluation.metrics.set_index("model")
    baseline, graph, hybrid = (metrics.loc[name] for name in ("baseline", "graph", "hybrid"))
    total = int(evaluation.splits.transactions.sum())
    fraud = int(hybrid.test_tp + hybrid.test_fn)
    legitimate = int(hybrid.test_tn + hybrid.test_fp)
    comparison = _table(
        ["Model", "Fraud recall", "Test FPR", "Precision", "Average precision", "Validation FPR"],
        [(_MODEL_NAMES[name], f"{r.test_recall:.2%}", f"{r.test_fpr:.2%}", f"{r.test_precision:.2%}",
          f"{r.test_average_precision:.3f}", f"{r.validation_fpr:.3%}") for name, r in metrics.iterrows()],
        "Model performance on the final holdout and false-positive rates on validation",
    )
    counts = _table(
        ["Model", "Fraud flagged", "Fraud missed", "False alerts", "Legitimate cleared", "Score threshold"],
        [(_MODEL_NAMES[name], int(r.test_tp), int(r.test_fn), int(r.test_fp), int(r.test_tn),
          f"{r.threshold:.6g}") for name, r in metrics.iterrows()], "Decision counts and frozen thresholds",
    )
    patterns = _table(
        ["Simulated pattern", "Model", "Fraud flagged / total", "Fraud recall", "Episodes detected / total"],
        [(_PATTERN_NAMES.get(r.pattern, r.pattern), _MODEL_NAMES[r.model],
          f"{r.transactions_detected:,} / {r.fraud_transactions:,}", f"{r.recall:.1%}",
          f"{r.rings_detected:,} / {r.rings:,}")
         for r in evaluation.by_pattern.sort_values(["pattern", "model"]).itertuples()],
        "Detection outcomes by coordinated fraud pattern",
    )
    periods = _table(
        ["Period", "Start (UTC)", "End (UTC)", "Payments", "Fraud"],
        [(_PERIOD_NAMES[r.period], r.start.strftime("%Y-%m-%d %H:%M:%S"),
          r.end.strftime("%Y-%m-%d %H:%M:%S"), f"{r.transactions:,}", f"{r.fraud:,}")
         for r in evaluation.splits.itertuples()], "Chronological experiment periods",
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Temporal graph fraud detection | Holdout results</title><style>{_STYLE}</style></head>
<body><main><header><h1>Temporal graph fraud detection for digital payments</h1>
<p class="summary">Across {fraud:,} fraudulent payments in the holdout, the transaction-only model flagged
{int(baseline.test_tp):,}, the graph model flagged {int(graph.test_tp):,}, and the model with the additional
anomaly score flagged {int(hybrid.test_tp):,}. They generated {int(baseline.test_fp):,}, {int(graph.test_fp):,},
and {int(hybrid.test_fp):,} false alerts respectively.</p></header>
<div class="stats"><div class="stat"><strong>{total:,}</strong><span>synthetic payments over {days:,} days</span></div>
<div class="stat"><strong>{fraud + legitimate:,}</strong><span>payments in the final holdout</span></div>
<div class="stat"><strong>{target_fpr:.2%}</strong><span>validation false-positive budget</span></div></div>
<section><h2>Model comparison</h2>
<p>All three models used the same supervised training period. The baseline used payment amount and time.
The graph model added recent transfer counts, distinct counterparties and pair history.
The hybrid model also used an Isolation Forest anomaly score.</p>
{comparison}
<p>Recall is the share of the {fraud:,} fraudulent payments flagged. The test false-positive rate (FPR)
is the share of the {legitimate:,} legitimate payments flagged. Precision is the share of alerts that
were fraud. Average precision measures how well the scores rank fraud across thresholds.</p>
<details open><summary>Alert counts and operating thresholds</summary>{counts}</details></section>
<section><h2>Coordinated activity</h2>
<p>An episode counts as detected when a model flags at least one fraudulent transfer from that episode
during the holdout. Transaction recall shows how much of each pattern the model caught.</p>{patterns}</section>
<section><h2>Chronological evaluation</h2>{periods}
<p>The earliest period supplied legitimate examples for the anomaly detector. The next trained the classifiers.
Each model used its own threshold calibrated on legitimate validation payments. The final period measures
how that frozen threshold performed on later payments.</p></section>
<section><h2>Leakage checks and scope</h2>
<ul><li>Recomputing features on an earlier prefix reproduced those rows from the full history.</li>
<li>Removing simulator labels and audit identifiers preserved the model inputs.</li>
<li>Every model stayed within the validation false-positive budget before the holdout comparison.</li></ul>
<p>The simulator includes legitimate group collections and payouts alongside coordinated fraud.
These numbers describe this generated population. Estimating performance on an operating payment system
requires independently labeled transactions from that system.</p></section>
<footer>Generator seed {seed} &nbsp;·&nbsp; MLflow run <code>{escape(run_id)}</code></footer>
</main></body></html>"""
    Path(output_path).write_text(content, encoding="utf-8")
