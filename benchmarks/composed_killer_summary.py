#!/usr/bin/env python
"""The quality side of the killer table: does the state hold as the session grows?

`run_killer_composed.sh` measures what a cold route transfers at composed lengths; this reads the
*quality* receipts for the same buckets and reports, per bucket, the state's size and its end-task
score against the 131,072-token reference view - the most a resident system can hold on this card.
A full-history reference is impossible past the model's window, and that is a property of the
hardware rather than of the state, so the reference is the strongest thing a resident baseline can
serve.

    python benchmarks/composed_killer_summary.py            # prints the table
    python benchmarks/composed_killer_summary.py --json     # and writes artifacts/g2-composed-quality-v1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BUCKETS = ("128k", "256k", "512k", "1m")


def decision(path: Path):
    try:
        rows = json.loads(path.read_text())["rows"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None, 0, None
    values = [row["active"]["f1"] for row in rows if row.get("recorded_files")]
    state = [row["active_tokens"] for row in rows]
    return (statistics.mean(values) if values else None), len(values), (
        statistics.median(state) if state else None)


def fidelity(path: Path):
    """median token-accuracy delta against the receipt's own reference, in percentage points."""
    try:
        rows = json.loads(path.read_text())["rows"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None, None
    deltas, refs = [], []
    for row in rows:
        active, full = row.get("active") or {}, row.get("full") or {}
        if active.get("token_accuracy") is None or full.get("token_accuracy") is None:
            continue
        deltas.append(100 * (active["token_accuracy"] - full["token_accuracy"]))
        refs.append(full.get("context_tokens"))
    return (statistics.median(deltas) if deltas else None, len(deltas))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true", help="also write the receipt")
    p.add_argument("--out", default="artifacts/g2-composed-quality-v1.json")
    args = p.parse_args(argv)

    rows = []
    for bucket in BUCKETS:
        state_receipt = Path(f"artifacts/g2b-patch-localization-composed-{bucket}-b8192-v1.json")
        reference = Path(f"artifacts/g2-composed-{bucket}-reference131k-v1.json")
        decision_f1, n, state_tokens = decision(state_receipt)
        fidelity_pp, n_fid = fidelity(reference)
        rows.append({"bucket": bucket, "state_tokens_p50": state_tokens,
                     "state_decision_f1": decision_f1, "decision_instances": n,
                     "state_vs_reference_pp_p50": fidelity_pp,
                     "fidelity_instances": n_fid,
                     "state_receipt": str(state_receipt), "reference_receipt": str(reference)})

    print(f"{'bucket':<8}{'state tokens p50':>18}{'state decision F1':>19}"
          f"{'vs 131K reference (pp)':>24}")
    for row in rows:
        tokens = row["state_tokens_p50"]
        decision_text = row["state_decision_f1"]
        fidelity_text = row["state_vs_reference_pp_p50"]
        print(f"{row['bucket']:<8}"
              f"{(f'{tokens:.0f}' if tokens else 'n/a'):>18}"
              f"{(f'{decision_text:.3f}' if decision_text is not None else 'n/a'):>19}"
              f"{(f'{fidelity_text:+.2f}' if fidelity_text is not None else 'n/a'):>24}")
    if args.json:
        payload = {"schema": "ephemeral-kv-composed-quality-v1", "kind": "derived_summary",
                   "what": ("the state's size and end-task score at composed history lengths, "
                            "against a 131,072-token reference view"),
                   "rows": rows}
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
