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
    # the two harnesses name the state size differently; both are the same measurement
    state = [row.get("active_tokens", row.get("active_tokens_estimate")) for row in rows]
    state = [value for value in state if value]
    return (statistics.mean(values) if values else None), len(values), (
        statistics.median(state) if state else None)


def accuracy_by_session(path: Path):
    """{session_id: active token accuracy} for one arm."""
    try:
        rows = json.loads(path.read_text())["rows"]
    except (OSError, json.JSONDecodeError, KeyError):
        return {}
    return {row.get("session_id"): (row.get("active") or {}).get("token_accuracy")
            for row in rows if (row.get("active") or {}).get("token_accuracy") is not None}


def state_vs_reference(state_path: Path, reference_path: Path):
    """Paired accuracy difference between the bounded state and the reference view, in pp.

    Both arms are scored on the same sessions, so this is a paired comparison; the reference is what
    a resident system can serve on this card (the newest tokens that fit its window), not the full
    history, which past that window cannot be served at all.
    """
    state, reference = accuracy_by_session(state_path), accuracy_by_session(reference_path)
    shared = sorted(set(state) & set(reference))
    if not shared:
        return None, 0, None
    deltas = [100 * (state[key] - reference[key]) for key in shared]
    return (statistics.median(deltas), len(shared),
            statistics.median(100 * state[key] for key in shared))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true", help="also write the receipt")
    p.add_argument("--out", default="artifacts/g2-composed-quality-v1.json")
    args = p.parse_args(argv)

    rows = []
    for bucket in BUCKETS:
        decision_receipt = Path(f"artifacts/g2b-patch-localization-composed-{bucket}-b8192-v1.json")
        state_receipt = Path(f"artifacts/g2-composed-{bucket}-state8192-v1.json")
        reference = Path(f"artifacts/g2-composed-{bucket}-reference131k-v1.json")
        decision_f1, n, _ = decision(decision_receipt)
        _, _, state_tokens = decision(state_receipt)
        fidelity_pp, n_fid, state_accuracy = state_vs_reference(state_receipt, reference)
        rows.append({"bucket": bucket, "state_tokens_p50": state_tokens,
                     "state_decision_f1": decision_f1, "decision_instances": n,
                     "state_vs_reference_pp_p50": fidelity_pp,
                     "state_accuracy_p50": state_accuracy,
                     "fidelity_instances": n_fid,
                     "state_receipt": str(state_receipt), "decision_receipt": str(decision_receipt),
                     "reference_receipt": str(reference)})

    print(f"{'bucket':<8}{'state tokens p50':>18}{'state decision F1':>19}"
          f"{'state acc p50':>15}{'vs resident reference (pp)':>28}")
    for row in rows:
        tokens = row["state_tokens_p50"]
        decision_text = row["state_decision_f1"]
        fidelity_text = row["state_vs_reference_pp_p50"]
        accuracy_text = row["state_accuracy_p50"]
        print(f"{row['bucket']:<8}"
              f"{(f'{tokens:.0f}' if tokens else 'n/a'):>18}"
              f"{(f'{decision_text:.3f}' if decision_text is not None else 'n/a'):>19}"
              f"{(f'{accuracy_text:.3f}' if accuracy_text is not None else 'n/a'):>15}"
              f"{(f'{fidelity_text:+.2f}' if fidelity_text is not None else 'n/a'):>28}")
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
