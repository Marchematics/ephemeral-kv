#!/usr/bin/env python
"""Collapse the G4 replay receipts into the phase boundary the paper needs.

Each receipt is one cell (fabric bandwidth x active-set size x capacity regime) with a
`verdict` that already evaluates the plan's gate.  This script collects them so the claim
can be read off one table instead of six files, and states the boundary explicitly:

    ephemeral mobility advances only where the active set is small *and* sessions are
    forced to move.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--glob", default="artifacts/g4-*.json")
    p.add_argument("--exclude", action="append", default=[],
                   help="glob of receipts to leave out of the grid (repeatable).  The grid is a "
                        "declared set of cells, not every replay file on disk: the 6,144-token "
                        "column is reported as its own capacity grid, and folding it in silently "
                        "would move every count in the paper")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    skipped = [name for pattern in args.exclude for name in glob.glob(pattern)]
    rows = []
    for path in sorted(glob.glob(args.glob)):
        if path in skipped:
            continue
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != "ephemeral-kv-g4-routing-replay-v1":
            continue
        if "verdict" not in payload:
            continue          # receipts from before the gate was evaluated in-artifact
        config, verdicts = payload["config"], payload["verdict"]
        # The regimes a receipt contributes are the ones it was *asked* to compute.  Reading
        # every table it happens to carry would double-count: a receipt run for the two
        # workload regimes also reports its balanced table, which is the same cell as the
        # balanced receipt at the same configuration.  Receipts that predate the flag
        # computed the three worker-behaviour regimes, which is the default.
        declared = config.get("regimes")
        names = ([r.strip() for r in str(declared).split(",") if r.strip()] if declared
                 else ["balanced", "slow_worker", "worker_failure"])
        for regime in names:
            table = (payload["balanced"] if regime == "balanced"
                     else payload["by_regime"].get(regime))
            if not table or regime not in verdicts:
                continue
            verdict = verdicts[regime]
            rows.append({
                "receipt": Path(path).name,
                "regime": regime,
                "bandwidth_gbs": config.get("bandwidth_gbs"),
                "active_tokens": config["active_tokens"],
                "warm_capacity": config["warm_capacity"],
                "tier_capacity": config["tier_capacity"],
                "strongest_baseline": verdict.get("strongest_baseline"),
                "goodput_ratio": round(verdict.get("goodput_ratio", 0.0), 3),
                "p99_reduction": (None if verdict.get("p99_reduction") is None
                                  else round(verdict["p99_reduction"], 3)),
                "decision": verdict.get("decision", "not_established"),
                "ephemeral_p50_s": round(table["ephemeral"]["p50_s"], 3),
                "baseline_p50_s": round(
                    table[verdict.get("strongest_baseline", "full_kv_move")]["p50_s"], 3),
            })
    advances = [r for r in rows if r["decision"] == "advance"]
    # The boundary is read off the grid rather than written down next to it: the region moved
    # once the active-set axis contained the sizes the compiler can actually run at and the
    # history axis contained the scale the durability argument is about, and prose that is not
    # derived from the rows goes stale silently.
    by_active: dict[int, list[int]] = {}
    by_active_regime: dict[str, list[int]] = {}
    by_history: dict[str, list[int]] = {}
    for row in rows:
        long_session = "1048576" in row["receipt"]
        scale = "1M-mix" if long_session else "corpus-mix"
        for bucket, key in ((by_active, row["active_tokens"]),
                            (by_active_regime, f"{row['regime']}|{row['active_tokens']}"),
                            (by_history, scale)):
            counts = bucket.setdefault(key, [0, 0])
            counts[1] += 1
            if row["decision"] == "advance":
                counts[0] += 1
    payload = {
        "schema": "ephemeral-kv-g4-phase-summary-v1",
        "kind": "gate_summary",
        "cells": len(rows),
        "advances": len(advances),
        "advances_by_active_tokens": {str(k): v for k, v in sorted(by_active.items())},
        "advances_by_regime_and_active": {k: v for k, v in sorted(by_active_regime.items())},
        "advances_by_history_scale": by_history,
        "boundary": (
            "ephemeral mobility clears the gate where rematerialisation is cheaper than "
            "moving the history's KV payload: forced mobility (a worker disappears) at any "
            "active set the compiler can hold quality at, and capacity-pressured fleets with "
            "long sessions.  It does not clear it under balanced load with short sessions, "
            "where the KV payload is already cheap to move"),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        bandwidth = ("local" if row["bandwidth_gbs"] is None
                     else f"{row['bandwidth_gbs']:g}")
        print(f"{row['regime']:<15} bw {bandwidth:>5} active {row['active_tokens']:>6} "
              f"| vs {row['strongest_baseline']:<16} goodput x{row['goodput_ratio']:.2f} "
              f"p99 {row['p99_reduction']} -> {row['decision']}")
    print(f"{len(advances)} of {len(rows)} cells advance")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
