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
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = []
    for path in sorted(glob.glob(args.glob)):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != "ephemeral-kv-g4-routing-replay-v1":
            continue
        if "verdict" not in payload:
            continue          # receipts from before the gate was evaluated in-artifact
        config, verdicts = payload["config"], payload["verdict"]
        for regime in ("balanced", "slow_worker", "worker_failure"):
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
    payload = {
        "schema": "ephemeral-kv-g4-phase-summary-v1",
        "kind": "gate_summary",
        "cells": len(rows),
        "advances": len(advances),
        "boundary": (
            "ephemeral mobility clears the gate only in forced mobility with a small "
            "active set; at the active-set size G2 measured as fidelity-preserving it "
            "does not, because rematerializing 16K tokens costs more than moving a KV "
            "payload on this fabric"),
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
