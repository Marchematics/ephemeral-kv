#!/usr/bin/env python
"""Kill gate G3: does bounded rematerialization make long sessions movable?

This is an *accounting* artifact, not a serving measurement.

The key quantity is the cold-route penalty for moving a turn away from the worker
that happens to hold a warm KV prefix.

Traditional migration restores or recomputes state proportional to accumulated
history. The EphemeralKV hypothesis is that a durable, sublinear text/index plane
can identify a small active working set, so the remote penalty is approximately

    index_lookup + prefill(active_working_set)

rather than a function of all prior tokens.

A scheduler benefits from mobility whenever the queue-delay advantage of another
worker exceeds that cold-route penalty. The script reports that threshold and
its scaling with session age. Every rate here is declared, never measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = {
    "kv_bytes_per_token": 32 * 1024,
    "link_bandwidth_gbs": 25.0,
    "full_prefill_tokens_per_s": 4000.0,
    "active_prefill_tokens_per_s": 20000.0,
    "indexed_lookup_ms": 10.0,
}


def route_costs(history_tokens: int, working_set_tokens: int, model=None):
    """Return declared cold-route penalties, excluding common decode time."""
    m = {**DEFAULT_MODEL, **(model or {})}
    kv_bytes = history_tokens * m["kv_bytes_per_token"]
    full_kv_transfer = kv_bytes / (m["link_bandwidth_gbs"] * 1e9)
    full_reprefill = history_tokens / m["full_prefill_tokens_per_s"]
    ephemeral = (
        m["indexed_lookup_ms"] / 1000.0
        + working_set_tokens / m["active_prefill_tokens_per_s"]
    )
    return {
        "history_tokens": history_tokens,
        "working_set_tokens": working_set_tokens,
        "full_kv_transfer_s": full_kv_transfer,
        "full_reprefill_s": full_reprefill,
        "ephemeral_rematerialize_s": ephemeral,
        "queue_gap_threshold_s": {
            "full_kv_transfer": full_kv_transfer,
            "full_reprefill": full_reprefill,
            "ephemeral": ephemeral,
        },
        "resident_bytes": {
            "full_kv": kv_bytes,
            "ephemeral_active": working_set_tokens * m["kv_bytes_per_token"],
        },
    }


def scaling(rows):
    ordered = sorted(rows, key=lambda r: r["history_tokens"])
    first, last = ordered[0], ordered[-1]

    def ratio(key):
        a, b = first[key], last[key]
        return b / a if a else None

    return {
        "history_growth_x": last["history_tokens"] / first["history_tokens"],
        "full_kv_transfer_growth_x": ratio("full_kv_transfer_s"),
        "full_reprefill_growth_x": ratio("full_reprefill_s"),
        "ephemeral_rematerialize_growth_x": ratio("ephemeral_rematerialize_s"),
        "full_kv_resident_growth_x": (
            last["resident_bytes"]["full_kv"] / first["resident_bytes"]["full_kv"]
        ),
        "ephemeral_resident_growth_x": (
            last["resident_bytes"]["ephemeral_active"]
            / first["resident_bytes"]["ephemeral_active"]
        ),
    }


def inversion(long_history=1_048_576, long_working_set=2048,
              short_history=32_768, short_working_set=16_384, model=None):
    long = route_costs(long_history, long_working_set, model)
    short = route_costs(short_history, short_working_set, model)
    return {
        "long": long,
        "short": short,
        "history_ratio_long_over_short": long_history / short_history,
        "ephemeral_resident_ratio_long_over_short": (
            long["resident_bytes"]["ephemeral_active"]
            / short["resident_bytes"]["ephemeral_active"]
        ),
        "ephemeral_route_cost_ratio_long_over_short": (
            long["ephemeral_rematerialize_s"]
            / short["ephemeral_rematerialize_s"]
        ),
        "full_kv_route_cost_ratio_long_over_short": (
            long["full_kv_transfer_s"] / short["full_kv_transfer_s"]
        ),
        "resident_inversion": (
            long["resident_bytes"]["ephemeral_active"]
            < short["resident_bytes"]["ephemeral_active"]
        ),
        "route_cost_inversion": (
            long["ephemeral_rematerialize_s"]
            < short["ephemeral_rematerialize_s"]
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--histories", type=int, nargs="+",
                   default=[32768, 131072, 524288, 1048576])
    p.add_argument("--working-set", type=int, default=4096)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = [route_costs(h, args.working_set) for h in args.histories]
    payload = {
        "schema": "ephemeral-kv-g3-mobility-accounting-v1",
        "kind": "accounting",
        "cost_model": DEFAULT_MODEL,
        "config": vars(args),
        "rows": rows,
        "scaling": scaling(rows),
        "inversion": inversion(),
        "hypothesis": (
            "session migration should become a queueing decision instead of a "
            "history-length decision when remote materialization cost tracks the "
            "active working set rather than accumulated history"
        ),
        "decision_rule": (
            "advance only if measured index lookup plus active-set prefill is mostly "
            "working-set-bounded across 32K->1M histories, and a real/simulator-backed "
            "cluster trace shows that soft-affinity routing improves p99 or SLO goodput "
            "over sticky routing under realistic skew or failure without material "
            "quality loss"
        ),
        "honesty": (
            "indexed_lookup_ms and throughput rates are assumptions. This artifact "
            "establishes the phase-boundary arithmetic to measure; it is not evidence "
            "that the boundary is reached on hardware."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "kind": payload["kind"],
        "scaling": payload["scaling"],
        "inversion": {
            k: v for k, v in payload["inversion"].items()
            if k.endswith("ratio_long_over_short") or k.endswith("inversion")
        },
    }, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
