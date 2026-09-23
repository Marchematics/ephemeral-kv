#!/usr/bin/env python
"""G3 accounting: can the cost of moving a session stop growing with history?

This file defines the *measurement target*. It is not a serving benchmark and every
rate in DEFAULT_MODEL is an assumption.

For a cold route, conventional systems pay for history-sized state:

    transfer(full KV)  or  re-prefill(full history)

EphemeralKV targets:

    indexed lookup + prefill(active working set)

The key paper hypothesis is that the latter can stay approximately independent of
accumulated history on real agent traces. A router can then treat locality as a hint
instead of an increasingly hard constraint as a session ages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = {
    "kv_bytes_per_token": 32 * 1024,
    "link_bandwidth_gbs": 25.0,
    # IMPORTANT: the same model prefill rate is used for both full-history and
    # active-set materialization. We do not grant EphemeralKV a faster kernel.
    "prefill_tokens_per_s": 4000.0,
    "indexed_lookup_ms": 10.0,
}


def route_costs(history_tokens: int, working_set_tokens: int, model=None):
    """Return declared cold-route penalties, excluding common decode time."""
    m = {**DEFAULT_MODEL, **(model or {})}
    kv_bytes = history_tokens * m["kv_bytes_per_token"]

    # One-way movement is the favorable baseline for full-KV migration.
    full_kv_transfer = kv_bytes / (m["link_bandwidth_gbs"] * 1e9)
    full_reprefill = history_tokens / m["prefill_tokens_per_s"]
    ephemeral = (
        m["indexed_lookup_ms"] / 1000.0
        + working_set_tokens / m["prefill_tokens_per_s"]
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


def transfer_crossover_history(working_set_tokens: int, model=None) -> int:
    """History length where one-way full-KV load equals active rematerialization."""
    m = {**DEFAULT_MODEL, **(model or {})}
    active_s = (
        m["indexed_lookup_ms"] / 1000.0
        + working_set_tokens / m["prefill_tokens_per_s"]
    )
    return int(
        active_s * m["link_bandwidth_gbs"] * 1e9 / m["kv_bytes_per_token"]
    )


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


def inversion(
    long_history=1_048_576,
    long_working_set=2048,
    short_history=32_768,
    short_working_set=16_384,
    model=None,
):
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
    p.add_argument(
        "--histories",
        type=int,
        nargs="+",
        default=[32768, 131072, 524288, 1048576],
    )
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
        "full_kv_transfer_crossover_history_tokens": transfer_crossover_history(
            args.working_set
        ),
        "inversion": inversion(),
        "hypothesis": (
            "session affinity pressure should stop increasing with session age when "
            "cold-route cost tracks the active working set instead of full history"
        ),
        "decision_rule": (
            "advance only if measured index lookup plus active-set prefill is mostly "
            "working-set-bounded across 32K->1M histories, and cluster replay shows "
            "that the changed mobility tax improves p99 or SLO goodput over strong "
            "sticky/cache-aware routing in realistic skew or failure regimes without "
            "material quality loss"
        ),
        "honesty": (
            "bandwidth, prefill throughput, and indexed lookup latency are assumptions. "
            "This artifact defines a phase boundary to measure; it is not evidence that "
            "real hardware reaches that boundary."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        json.dumps(
            {
                "kind": payload["kind"],
                "scaling": payload["scaling"],
                "full_kv_transfer_crossover_history_tokens": (
                    payload["full_kv_transfer_crossover_history_tokens"]
                ),
                "inversion": {
                    k: v
                    for k, v in payload["inversion"].items()
                    if k.endswith("ratio_long_over_short") or k.endswith("inversion")
                },
            },
            indent=2,
        )
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
