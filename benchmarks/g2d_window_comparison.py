#!/usr/bin/env python
"""Two window rules on the same turns: does spending the window on actions recover the emit rate?

The mid-session action measurement found the bounded state emitting an executable action in 0.375 of
turns against the full transcript's 0.625, and located the difference in what the view *shows* - a
median of 6 prior actions against 44.  `action_window` is the fix that follows from that: keep the
newest span, then spend the rest of the window on the most recent action-bearing spans, and fill
only what is left with ordinary recency.

This reads the two receipts side by side.  The comparison is only meaningful if both scored the same
turns, so that is checked first and the script refuses to report if it does not hold: the `full` arm
is built from the full transcript, which neither window rule touches, and generation is greedy, so
its continuations must be byte-identical across the two runs.  When they are, the difference between
the receipts is the window rule and nothing else.

Usage:
    python benchmarks/g2d_window_comparison.py \\
        --baseline artifacts/g2d-action-turns-state4096-v1.json \\
        --candidate artifacts/g2d-action-turns-actionwindow4096-v1.json \\
        --out artifacts/g2d-window-comparison-v1.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import statistics
from pathlib import Path

ARMS = ("full", "active", "retrieval")


def bootstrap_ci(diffs: list[float], repeats: int = 20000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats))
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


def rate(rows, arm, key, subset=""):
    considered = rows if not subset else [r for r in rows if r[subset].get("parseable")]
    values = [r[arm].get(key) for r in considered if r[arm].get(key) is not None]
    return (sum(1 for v in values if v) / len(values)) if values else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", default="artifacts/g2d-action-turns-state4096-v1.json")
    p.add_argument("--candidate", default="artifacts/g2d-action-turns-actionwindow4096-v1.json")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    base = json.loads(Path(args.baseline).read_text())
    cand = json.loads(Path(args.candidate).read_text())
    key = lambda r: (r.get("session_id"), r.get("turn_index"))
    base_rows = {key(r): r for r in base["rows"]}
    cand_rows = {key(r): r for r in cand["rows"]}
    shared = sorted(set(base_rows) & set(cand_rows), key=lambda k: (str(k[0]), k[1]))
    if not shared:
        raise SystemExit("the two receipts share no turns; there is nothing to compare")

    # The full arm is the control: neither window rule touches it, and generation is greedy, so its
    # continuations must match exactly.  If they do not, the two runs scored different turns and
    # every difference below would be confounded with that.
    differing = [k for k in shared
                 if base_rows[k]["full"]["continuation"] != cand_rows[k]["full"]["continuation"]]
    if differing:
        raise SystemExit(
            f"{len(differing)} of {len(shared)} turns have a different full-transcript continuation; "
            "the two runs did not score the same turns and the comparison would be confounded")

    rows = []
    for k in shared:
        rows.append({
            "session_id": k[0], "turn_index": k[1],
            "history_tokens": base_rows[k]["history_tokens"],
            "recorded_action": base_rows[k]["recorded_action"],
            "recency": {**{m: base_rows[k]["active"].get(m) for m in
                           ("parseable", "tool", "target", "exact")}},
            "action_window": {**{m: cand_rows[k]["active"].get(m) for m in
                                 ("parseable", "tool", "target", "exact")}},
            "retrieval": {**{m: base_rows[k]["retrieval"].get(m) for m in
                             ("parseable", "tool", "target", "exact")}},
        })

    def arm_stats(arm):
        return {"emits_an_action": rate(rows, arm, "parseable"),
                "tool_match": rate(rows, arm, "tool", subset=arm),
                "target_match": rate(rows, arm, "target", subset=arm)}

    diffs = [float(bool(r["action_window"]["parseable"])) - float(bool(r["recency"]["parseable"]))
             for r in rows]
    lo, hi = bootstrap_ci(diffs)
    paired = {"metric": "emits_an_action", "n": len(diffs), "mean_diff": statistics.mean(diffs),
              "ci95": [round(lo, 4), round(hi, 4)], "includes_zero": lo <= 0 <= hi,
              "wins": sum(1 for d in diffs if d > 0), "losses": sum(1 for d in diffs if d < 0)}
    payload = {
        "schema": "ephemeral-kv-g2d-window-comparison-v1",
        "kind": "derived_measurement",
        "baseline_receipt": args.baseline,
        "candidate_receipt": args.candidate,
        "summary": {
            "turns_in_common": len(shared),
            "turns_scored_by_both": len(shared),
            "full_arm_continuations_identical": True,
            "history_tokens_p50": int(statistics.median([r["history_tokens"] for r in rows])),
            "arms": {"full_transcript": arm_stats("full") if "full" in ARMS else None,
                     "recency_window": arm_stats("recency"),
                     "action_window": arm_stats("action_window"),
                     "plain_retrieval": arm_stats("retrieval")},
            "action_window_vs_recency": paired,
        },
        "rows": rows,
        "reading": (
            "the two window rules scored the same turns, verified by the full-transcript arm "
            "reproducing byte-identically, so the difference between them is the selection rule"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
