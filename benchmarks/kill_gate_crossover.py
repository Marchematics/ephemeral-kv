#!/usr/bin/env python
"""Kill gate G1: is there a discard-and-recompile crossover?

Two layers, deliberately separated:

* **accounting** (runs anywhere, CPU): for a session of `T` turns, compute what each
  policy moves or computes -- bytes transferred for the `keep_*` policies, prefix
  tokens for `recompute_full`, working-set tokens plus index lookups for
  `recompile_active` -- from a *declared* cost model.  This is a planning tool: the
  parameters are assumptions, not measurements, and the output says so.
* **measurement** (needs a GPU and a model): the same policies timed end to end.  Not
  implemented yet; the accounting output names the turn count where the crossover is
  predicted, which is the window the measurement should sweep first.

The decision rule is fixed in `docs/PLAN.md`; this script only reports where the
crossover would be *under the declared model*, and never as a result.

Usage::

    python benchmarks/kill_gate_crossover.py --turns 8 32 128 512 \
        --turn-tokens 512 --working-set 4096 --out artifacts/g1-accounting.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: declared cost model -- every number here is an assumption to be replaced by
#: measurement; the units are per-token so the arithmetic is checkable
DEFAULT_MODEL = {
    "kv_bytes_per_token": 32 * 1024,      # Llama-3.2-1B-class, bf16, K and V
    "hbm_bandwidth_gbs": 600.0,           # move KV out and back
    "pcie_bandwidth_gbs": 25.0,           # DRAM/NVMe path
    "prefill_tokens_per_s": 4000.0,       # full re-prefill throughput
    "compile_tokens_per_s": 20000.0,      # attention-scored working-set compile
    "index_lookup_tokens_per_s": 200000.0,  # semantic index scan over the transcript
    "decode_tokens_per_s": 45.0,          # one decoded token, any policy
}


def accounting(turns, turn_tokens, request_tokens, working_set, decode_tokens=128,
               model=None):
    """Bytes moved and tokens computed per policy for one request at turn `T`.

    Returns a list of dicts with `seconds` split into its parts so a reader can see
    which assumption dominates, and `bytes_moved` for the storage policies.
    """
    m = {**DEFAULT_MODEL, **(model or {})}
    history = turns * turn_tokens
    # the decoded tokens are what every policy pays equally; the request's prompt is
    # what the compile has to cover, so it is capped by the working set
    covered = min(request_tokens, working_set)
    kv_bytes = history * m["kv_bytes_per_token"]
    rows = []

    def add(name, **parts):
        seconds = sum(parts.values())
        rows.append({"policy": name, "seconds": round(seconds, 4),
                     "parts": {k: round(v, 4) for k, v in parts.items()},
                     "history_tokens": history})

    add("keep_hbm", decode=decode_tokens / m["decode_tokens_per_s"])
    add("keep_dram",
        transfer=2 * kv_bytes / (m["pcie_bandwidth_gbs"] * 1e9),
        decode=decode_tokens / m["decode_tokens_per_s"])
    add("keep_nvme",
        transfer=2 * kv_bytes / (m["pcie_bandwidth_gbs"] * 1e9),
        decode=decode_tokens / m["decode_tokens_per_s"])
    add("recompute_full",
        prefill=history / m["prefill_tokens_per_s"],
        decode=decode_tokens / m["decode_tokens_per_s"])
    add("recompile_active",
        index=history / m["index_lookup_tokens_per_s"],
        compile=covered / m["compile_tokens_per_s"],
        prefill=covered / m["prefill_tokens_per_s"],
        decode=decode_tokens / m["decode_tokens_per_s"])
    for row in rows:
        row["bytes_moved"] = (round(2 * kv_bytes) if row["policy"].startswith("keep_")
                              and row["policy"] != "keep_hbm" else 0)
        row["resident_bytes"] = (round(kv_bytes) if row["policy"] == "keep_hbm"
                                 else round(working_set * m["kv_bytes_per_token"]))
    return rows


def crossover(rows):
    """Turn count where `recompile_active` first beats each other policy.

    Reported per baseline on purpose: a cache that is already resident in HBM is free
    at request time, so recompilation cannot beat `keep_hbm` on latency by
    construction.  What it can beat is *moving* state in from DRAM/NVMe, and what it
    can beat more broadly is capacity (gates G3/G4) - so the interesting numbers here
    are the crossings against `keep_dram` and `keep_nvme`, not against `keep_hbm`.
    """
    active = {r["history_tokens"]: r["seconds"] for r in rows
              if r["policy"] == "recompile_active"}
    crossings = {}
    for policy in sorted({r["policy"] for r in rows} - {"recompile_active"}):
        per_history = {r["history_tokens"]: r["seconds"] for r in rows
                       if r["policy"] == policy}
        crossing = None
        for history in sorted(active):
            if history in per_history and active[history] < per_history[history]:
                crossing = history
                break
        crossings[policy] = crossing
    return crossings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--turn-tokens", type=int, default=512)
    parser.add_argument("--request-tokens", type=int, default=2048)
    parser.add_argument("--working-set", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=128,
                        help="tokens the response decodes; every policy pays this "
                             "equally, so it sets the floor rather than the ranking")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = []
    for turns in args.turns:
        rows.extend(accounting(turns, args.turn_tokens, args.request_tokens,
                               args.working_set, args.decode_tokens))
    crossings = crossover(rows)
    payload = {
        "schema": "ephemeral-kv-g1-accounting-v1",
        "kind": "accounting",           # NOT a measurement: the model is assumed
        "cost_model": DEFAULT_MODEL,
        "config": vars(args),
        "rows": rows,
        "crossover_history_tokens": crossings,
        "crossover_turns": {policy: (tokens // args.turn_tokens if tokens else None)
                            for policy, tokens in crossings.items()},
        "decision_rule": ("lives if a measured recompile_active beats keep_* on "
                          "time-to-response for some T <= 512, or beats "
                          "recompute_full by a factor that grows with T"),
        "gate_signal": ("recompile_active beats recompute_full from the first tested "
                        "length, and does not beat keep_dram/keep_nvme by 512 turns "
                        "under the declared model: the latency case is not made by "
                        "tiered-storage comparison, so G1 should be re-run with "
                        "measured transfer and index costs before the project leans on "
                        "it - capacity (G3/G4) and model-agnosticism (G5) are the gates "
                        "that carry the claim"),
        "next_step": ("measure the same policies end to end around these lengths, and "
                      "replace every DEFAULT_MODEL parameter with a measured one"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in ("kind", "crossover_turns")}, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
