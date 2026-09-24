#!/usr/bin/env python
"""How many sessions fit on a worker: the capacity consequence of the state bound.

Under the current resource model a worker's HBM holds *histories*: sessions per worker =
budget / (history_tokens x kv_bytes_per_token), so a 128K-token session costs 1.5 GiB on the
0.5B geometry this hardware measured and 16 GiB on an 8B-class one - the second number exceeds a
24 GiB card, which is why "one long session per GPU" is the folklore.

Under the measured bound the resident object is the *execution state* (7-8K tokens regardless of
history), so the same budget holds states, and the number is set by the budget rather than by
age.  This is arithmetic on two measured quantities (the state size from G2, the KV geometry from
G3) and it is reported as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GEOMETRIES = {
    "Qwen2.5-0.5B (measured)": 12288,
    "8B-class": 131072,
    "70B-class": 327680,
}
HISTORIES = (32768, 131072, 1048576)
STATE_TOKENS = (4096, 8192, 16384)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hbm-gib", type=float, default=20.0,
                   help="usable HBM per worker (a 24 GiB card holds ~20 GiB after weights)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    budget_bytes = args.hbm_gib * (1 << 30)
    rows = []
    for name, kv in GEOMETRIES.items():
        for history in HISTORIES:
            rows.append({
                "geometry": name, "kv_bytes_per_token": kv, "resident": "history",
                "tokens": history, "bytes": history * kv,
                "sessions_per_worker": round(budget_bytes / (history * kv), 1),
            })
        for state in STATE_TOKENS:
            rows.append({
                "geometry": name, "kv_bytes_per_token": kv, "resident": "compiled state",
                "tokens": state, "bytes": state * kv,
                "sessions_per_worker": round(budget_bytes / (state * kv), 1),
            })
    payload = {"schema": "ephemeral-kv-capacity-planning-v1", "kind": "derived_table",
               "hbm_gib": args.hbm_gib, "rows": rows,
               "note": ("state sizes are measured (G2); kv_bytes_per_token is measured for the "
                        "0.5B model and declared for the 8B/70B geometries")}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"usable HBM {args.hbm_gib:.0f} GiB per worker")
    for name in GEOMETRIES:
        print(f"\n{name} ({GEOMETRIES[name]:,} B/token)")
        for row in rows:
            if row["geometry"] != name:
                continue
            print(f"  resident {row['resident']:<14} {row['tokens']:>9,} tokens "
                  f"{row['bytes'] / (1 << 30):>8.2f} GiB  -> {row['sessions_per_worker']:>7} "
                  f"sessions/worker")
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
