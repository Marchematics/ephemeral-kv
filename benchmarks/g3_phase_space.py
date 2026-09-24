#!/usr/bin/env python
"""G3 phase-space accounting for session mobility.

This script is deliberately analytical. It asks where active-state rematerialization
would become cheaper than a favorable one-way full-KV transfer, under declared
hardware/model parameters.

The crossover follows directly from

    history * kv_bytes_per_token / link_Bps
      = lookup_s + active_tokens / prefill_tokens_per_s

and is useful for choosing the real hardware/model sweep. It is not a performance
measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def crossover_tokens(
    *,
    kv_bytes_per_token: int,
    link_bandwidth_gbs: float,
    prefill_tokens_per_s: float,
    active_tokens: int,
    lookup_ms: float,
) -> int:
    active_s = lookup_ms / 1000.0 + active_tokens / prefill_tokens_per_s
    return int(active_s * link_bandwidth_gbs * 1e9 / kv_bytes_per_token)


def phase_grid(
    kv_kib=(32, 64, 128, 256),
    link_gbs=(12.5, 25.0, 50.0, 100.0),
    active_tokens=(2048, 4096, 8192),
    prefill_tokens_per_s=4000.0,
    lookup_ms=10.0,
):
    rows = []
    for kv in kv_kib:
        for link in link_gbs:
            for active in active_tokens:
                cross = crossover_tokens(
                    kv_bytes_per_token=kv * 1024,
                    link_bandwidth_gbs=link,
                    prefill_tokens_per_s=prefill_tokens_per_s,
                    active_tokens=active,
                    lookup_ms=lookup_ms,
                )
                rows.append(
                    {
                        "kv_kib_per_token": kv,
                        "link_gbs": link,
                        "active_tokens": active,
                        "crossover_history_tokens": cross,
                        "wins_by_1m": cross <= 1_000_000,
                    }
                )
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kv-kib", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--link-gbs", type=float, nargs="+", default=[12.5, 25, 50, 100])
    p.add_argument("--active", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--prefill-tps", type=float, default=4000.0)
    p.add_argument("--lookup-ms", type=float, default=10.0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = phase_grid(
        kv_kib=args.kv_kib,
        link_gbs=args.link_gbs,
        active_tokens=args.active,
        prefill_tokens_per_s=args.prefill_tps,
        lookup_ms=args.lookup_ms,
    )
    payload = {
        "schema": "ephemeral-kv-g3-phase-accounting-v1",
        "kind": "accounting",
        "config": vars(args),
        "rows": rows,
        "interpretation": (
            "below the crossover, favorable full-KV transfer is cheaper; above it, "
            "active-state rematerialization is cheaper under the declared model"
        ),
        "honesty": "all rates are assumptions; use this grid to select real measurements",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
