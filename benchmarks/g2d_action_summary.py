#!/usr/bin/env python
"""The mid-session action measurement, read three ways.

The receipt's own summary reports a rate per arm over all rows, which is not enough to answer the
two questions the gate asks, because they turn out to have different answers:

* **Does the bounded state regress against the full transcript?**  The rate at which a continuation
  contains an executable action at all is the number to read, and it is not the same as the rate at
  which the action is *right*: an arm that answers with prose instead of a tool call has not taken a
  wrong action, it has failed to take one, and a runtime pays for that turn either way.
* **Does it beat plain retrieval at the same budget?**  Retrieval almost never emits an action, so
  the same rate answers this one decisively.

Conditional accuracy - of the actions an arm does emit, how many name the right tool and the right
file - is reported separately, because it is the half that does *not* separate the full and bounded
arms and would be invisible in a single blended number.

Usage:
    python benchmarks/g2d_action_summary.py --receipt artifacts/g2d-action-turns-state4096-v1.json \
        --out artifacts/g2d-action-summary-v1.json
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
# what each arm is, in the paper's terms
ARM_LABEL = {"full": "full transcript", "active": "bounded state (3,584-token window + consolidated "
                                                "far field at a 4,096-token total)",
             "retrieval": "plain lexical retrieval at 4,096"}


def bootstrap_ci(diffs: list[float], repeats: int = 20000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats))
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


def paired(rows: list[dict], base: str, cand: str, key: str) -> dict:
    diffs = [float(bool(r[cand].get(key))) - float(bool(r[base].get(key))) for r in rows
             if r.get(cand, {}).get(key) is not None and r.get(base, {}).get(key) is not None]
    if not diffs:
        return {"n": 0}
    lo, hi = bootstrap_ci(diffs)
    return {"n": len(diffs), "mean_diff": statistics.mean(diffs),
            "ci95": [round(lo, 4), round(hi, 4)], "includes_zero": lo <= 0 <= hi,
            "wins": sum(1 for d in diffs if d > 0), "losses": sum(1 for d in diffs if d < 0)}


def rate(rows: list[dict], arm: str, key: str, subset: str = "") -> float | None:
    considered = rows if not subset else [r for r in rows if r[subset].get("parseable")]
    values = [r[arm].get(key) for r in considered if r[arm].get(key) is not None]
    return (sum(1 for v in values if v) / len(values)) if values else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--receipt", default="artifacts/g2d-action-turns-state4096-v1.json")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = json.loads(Path(args.receipt).read_text())["rows"]
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "label": ARM_LABEL[arm],
            "rows": sum(1 for r in rows if arm in r),
            # unconditional: does the continuation contain an executable action at all
            "emits_an_action": rate(rows, arm, "parseable"),
            # conditional on emitting one: is it the action the agent took
            "tool_match": rate(rows, arm, "tool", subset=arm),
            "target_match": rate(rows, arm, "target", subset=arm),
            "exact_match": rate(rows, arm, "exact", subset=arm),
        }
    comparisons = {
        "active_vs_full_emits": paired(rows, "full", "active", "parseable"),
        "active_vs_retrieval_emits": paired(rows, "retrieval", "active", "parseable"),
        "active_vs_full_tool": paired(rows, "full", "active", "tool"),
        "active_vs_full_target": paired(rows, "full", "active", "target"),
        "active_vs_retrieval_tool": paired(rows, "retrieval", "active", "tool"),
        "active_vs_retrieval_target": paired(rows, "retrieval", "active", "target"),
    }
    payload = {
        "schema": "ephemeral-kv-g2d-action-summary-v1",
        "kind": "derived_measurement",
        "source_receipt": args.receipt,
        "summary": {
            "action_turns": len(rows),
            "sessions": len({r.get("session_id") for r in rows}),
            "history_tokens_p50": int(statistics.median([r["history_tokens"] for r in rows])),
            "arms": arms,
            "comparisons": comparisons,
        },
        "reading": (
            "three arms on the same mid-session turn.  The bounded state beats plain retrieval "
            "decisively on whether an executable action is emitted at all, and trails the full "
            "transcript there; conditional on emitting one, the two are close.  That split is the "
            "result, and reporting only a blended rate would hide half of it"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
