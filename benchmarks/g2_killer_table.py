#!/usr/bin/env python
"""The killer table: does the state that must move grow with the session?

The paper's sharpest claim is not "we compress 4x".  It is that the *durable* session and the
*executable* state are different objects, and that only the first grows:

    session age / raw history  ->  the state that must move

This script reads the G2 receipts and reports, per session-length bucket, (a) how large the
compiled view is, (b) what it costs in teacher-forced fidelity, and (c) what it does to the
end task.  It also reports the mobility inversion the law implies, from the G3 measured
primitives: a *long* session with a compiled state is cheaper to move than a *short* session
with a raw state, which is what makes session age stop predicting placement cost.

Every number is read from an artifact; nothing is interpolated.  Buckets are declared below and
a bucket with no examples is reported as empty rather than skipped silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BUCKETS = (
    (0, 32768, "<=32K"),
    (32768, 65536, "32K-64K"),
    (65536, 131072, "64K-128K"),
    (131072, 262144, "128K-256K"),
    (262144, 1 << 62, ">=256K"),
)


def median(xs: list[float]):
    if not xs:
        return None
    ys = sorted(xs)
    mid = len(ys) // 2
    return ys[mid] if len(ys) % 2 else 0.5 * (ys[mid - 1] + ys[mid])


def bucket_of(history: int) -> str:
    for low, high, label in BUCKETS:
        if low <= history < high:
            return label
    return BUCKETS[-1][2]


def fidelity_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    out = []
    for row in payload.get("rows", []):
        full, active = row.get("full") or {}, row.get("active") or {}
        if full.get("token_accuracy") is None or active.get("token_accuracy") is None:
            continue
        out.append({
            "history": row["history_tokens_estimate"],
            "active": row.get("active_tokens_estimate") or 0,
            "acc_delta": active["token_accuracy"] - full["token_accuracy"],
            "nll_delta": active["nll"] - full["nll"],
        })
    return out


def end_task_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    out = []
    for row in payload.get("rows", []):
        full, active = row.get("full") or {}, row.get("active") or {}
        if full.get("f1") is None or active.get("f1") is None:
            continue
        # only instances whose recorded patch mentions files are scoreable; the harness summary
        # reports those (12 of 24, 26 of 48) and a median over all rows would be zero
        if "next_turn" not in (row.get("full") or {}):
            continue
        out.append({
            "history": row["history_tokens"],
            "active": row.get("active_tokens") or 0,
            "full_f1": full["f1"],
            "active_f1": active["f1"],
        })
    return out


def table(rows: list[dict], key: str) -> dict:
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(bucket_of(int(row["history"])), []).append(row)
    out = {}
    for _low, _high, label in BUCKETS:
        group = by.get(label)
        if not group:
            out[label] = {"n": 0}
            continue
        entry = {
            "n": len(group),
            "raw_history_p50": int(median([r["history"] for r in group])),
            "compiled_state_p50": int(median([r["active"] for r in group])),
        }
        if key == "fidelity":
            entry["acc_delta_pp_p50"] = round(100 * median([r["acc_delta"] for r in group]), 2)
            entry["nll_delta_p50"] = round(median([r["nll_delta"] for r in group]), 3)
        else:
            # the full-history baseline moves with the instance set (0.045 on the first 24 paired
            # sessions, 0.089 on the first 48), so only the *paired* difference is comparable
            # across receipts; it is reported first and the raw values after it
            entry["paired_f1_mean"] = round(
                sum(r["active_f1"] - r["full_f1"] for r in group) / len(group), 3)
            entry["full_f1_mean"] = round(sum(r["full_f1"] for r in group) / len(group), 3)
            entry["active_f1_mean"] = round(sum(r["active_f1"] for r in group) / len(group), 3)
        out[label] = entry
    return out


def inversion(costs_path: Path, pairs: list[tuple[int, int]]) -> list[dict]:
    """Mobility cost of each (history, compiled state) pair from the measured primitives."""
    payload = json.loads(costs_path.read_text())
    costs = payload["costs"]
    prefill = {int(k): v for k, v in costs["active_prefill_s"].items()}
    lookup = {int(k): v for k, v in costs["lookup_s"].items()}
    bandwidth = costs["bandwidth_gbs"]
    kv_bytes = costs["kv_bytes_per_token"]

    def lookup_for(history: int) -> float:
        for limit in sorted(lookup):
            if history <= limit:
                return lookup[limit]
        return lookup[max(lookup)]

    def prefill_for(active: int) -> float:
        for size in sorted(prefill):
            if active <= size:
                return prefill[size]
        return prefill[max(prefill)]

    rows = []
    for history, active in pairs:
        rows.append({
            "history": history,
            "compiled_state": active,
            "lookup_s": round(lookup_for(history), 6),
            "prefill_s": round(prefill_for(active), 4),
            "move_full_kv_s": round(history * kv_bytes / (bandwidth * 1e9), 4),
            "mobility_s": round(lookup_for(history) + prefill_for(active), 4),
        })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fidelity", action="append", default=[],
                   help="G2 model-quality receipt (teacher-forced deltas)")
    p.add_argument("--end-task", action="append", default=[],
                   help="G2b patch-localisation receipt")
    p.add_argument("--costs", default="artifacts/g4-fabric-bw23.3-a2048-v1.json",
                   help="receipt carrying the G3 measured primitives (prefill/lookup/bandwidth)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    report = {"schema": "ephemeral-kv-killer-table-v1", "kind": "derived_table",
              "buckets": [label for _l, _h, label in BUCKETS], "fidelity": {}, "end_task": {}}
    for name in args.fidelity:
        path = Path(name)
        if not path.exists():
            continue
        report["fidelity"][path.name] = table(fidelity_rows(path), "fidelity")
    for name in args.end_task:
        path = Path(name)
        if not path.exists():
            continue
        report["end_task"][path.name] = table(end_task_rows(path), "end_task")

    # The inversion: with a compiled state, a much older session is cheaper to move than a
    # young one carrying its raw view.  Active sizes are the budgets the G2 receipts measured at.
    pairs = [(32 * 1024, 16 * 1024), (1 << 20, 8 * 1024), (1 << 20, 4 * 1024), (262144, 8 * 1024)]
    if Path(args.costs).exists():
        report["inversion"] = inversion(Path(args.costs), pairs)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for label in report["buckets"]:
        for name, buckets in report["fidelity"].items():
            entry = buckets[label]
            if entry.get("n"):
                print(f"fidelity  {name[:34]:<36} {label:<9} n={entry['n']:<3} "
                      f"raw p50={entry['raw_history_p50']:>7} state p50="
                      f"{entry['compiled_state_p50']:>6} acc {entry['acc_delta_pp_p50']:+.2f} pp "
                      f"nll {entry['nll_delta_p50']:+.3f}")
    for label in report["buckets"]:
        for name, buckets in report["end_task"].items():
            entry = buckets[label]
            if entry.get("n"):
                print(f"end-task  {name[:34]:<36} {label:<9} n={entry['n']:<3} "
                      f"raw p50={entry['raw_history_p50']:>7} state p50="
                      f"{entry['compiled_state_p50']:>6} paired F1 "
                      f"{entry['paired_f1_mean']:+.3f} "
                      f"(active {entry['active_f1_mean']:.3f} vs full {entry['full_f1_mean']:.3f})")
    for row in report.get("inversion", []):
        print(f"inversion history {row['history']:>8} state {row['compiled_state']:>6} "
              f"mobility {row['mobility_s']:.4f} s   (full KV move {row['move_full_kv_s']:.4f} s)")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
