#!/usr/bin/env python
"""Placement under the two cost models: does session age still rank cold-route cost?

The paper's counter-intuitive claim is that a *older* session can be cheaper to place cold than a
younger one, so a scheduler that ranks by age (or by history footprint, which is the same thing in
today's resource model) ranks by the wrong quantity.  This turns that into arithmetic over the
measured primitives:

* true cold cost      = lookup(history) + active_prefill(state)        (both measured)
* footprint estimate  = history x kv_bytes_per_token / bandwidth       (what a KV-mover pays)

and reports the ranking disagreement, its cost under a fixed cold-path budget, and the sessions a
worker can absorb under each model.  State sizes are the measured ones (G2); the KV geometry is
measured for the 0.5B model and declared for 8B/70B classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SESSIONS = (
    # label, history tokens, compiled state tokens
    ("young, large state (pre-compiler requirement)", 32768, 16384),
    ("young, bounded state", 32768, 8192),
    ("mid, bounded state", 131072, 8192),
    ("old, bounded state", 262144, 8192),
    ("1M, bounded state", 1048576, 8192),
    ("1M, small state", 1048576, 4096),
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--costs", default="artifacts/g4-fabric-bw23.3-a2048-v1.json",
                   help="receipt carrying the measured primitives")
    p.add_argument("--kv-bytes-per-token", type=int, default=131072,
                   help="declared geometry for the footprint estimate (8B-class)")
    p.add_argument("--cold-budget-s", type=float, default=1.0,
                   help="cold-path work a worker can absorb in the window being planned")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    payload = json.loads(Path(args.costs).read_text())
    costs = payload["costs"]
    prefill = {int(k): v for k, v in costs["active_prefill_s"].items()}
    lookup = {int(k): v for k, v in costs["lookup_s"].items()}
    bandwidth = costs["bandwidth_gbs"]

    def lookup_for(history: int) -> float:
        for limit in sorted(lookup):
            if history <= limit:
                return lookup[limit]
        return lookup[max(lookup)]

    def prefill_for(state: int) -> float:
        for size in sorted(prefill):
            if state <= size:
                return prefill[size]
        return prefill[max(prefill)]

    rows = []
    for label, history, state in SESSIONS:
        true_cost = lookup_for(history) + prefill_for(state)
        footprint = history * args.kv_bytes_per_token / (bandwidth * 1e9)
        rows.append({
            "session": label, "history": history, "state": state,
            "true_cost_s": round(true_cost, 4),
            "footprint_estimate_s": round(footprint, 4),
            "rank_by_truth": None, "rank_by_footprint": None,
            "served_by_truth": round(args.cold_budget_s / true_cost, 1),
            "served_by_footprint_estimate": round(args.cold_budget_s / footprint, 1),
        })
    for key, field in (("true_cost_s", "rank_by_truth"),
                       ("footprint_estimate_s", "rank_by_footprint")):
        order = sorted(rows, key=lambda r: r[key])
        for position, row in enumerate(order):
            row[field] = position

    payload = {
        "schema": "ephemeral-kv-placement-inversion-v1",
        "kind": "derived_analysis",
        "kv_bytes_per_token": args.kv_bytes_per_token,
        "cold_budget_s": args.cold_budget_s,
        "rows": rows,
        "ranking_reversed": (
            "the oldest session with a bounded state is cheaper to place cold than the youngest "
            "one carrying a 16K state"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'session':<46} {'history':>9} {'state':>7} {'true s':>8} {'footprint s':>12} "
          f"{'rank':>5} {'rank_est':>9}")
    for row in rows:
        print(f"{row['session']:<46} {row['history']:>9} {row['state']:>7} "
              f"{row['true_cost_s']:>8.4f} {row['footprint_estimate_s']:>12.3f} "
              f"{row['rank_by_truth']:>5} {row['rank_by_footprint']:>9}")
    budget = args.cold_budget_s
    print(f"\nwith {budget:.1f} s of cold-path work: "
          f"{rows[4]['served_by_truth']} x (1M, bounded state) by the measured cost, "
          f"{rows[0]['served_by_footprint_estimate']} x (32K, 16K state) by the footprint estimate")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
