#!/usr/bin/env python
"""Kill gate G1: is there a discard-and-recompile crossover?

Compares, for one next request over a session of `T` turns:

* `keep_hbm`      -- the session's KV stays resident;
* `keep_dram`     -- it is moved out to host memory between turns and back in;
* `recompute_full`-- the whole history is re-prefilled from the transcript;
* `recompile_active` -- a small working set for this request is compiled and the rest
  of the state is released.

The measurement this project needs is *total time to response* and *peak resident
memory* per policy as `T` grows, and whether `recompile_active` crosses the `keep_*`
policies before `T = 512`.  Nothing here depends on QCC: the compiler is a pluggable
callable, and the default is a plain attention-score working set so the baseline
stands on its own.

Usage::

    python benchmarks/kill_gate_crossover.py --model <checkpoint> \
        --turns 8 32 128 512 --turn-tokens 512 --working-set 4096 --request-tokens 2048 \
        --out artifacts/kill-gate-crossover.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("EPHEMERALKV_MODEL"))
    parser.add_argument("--turns", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--turn-tokens", type=int, default=512)
    parser.add_argument("--request-tokens", type=int, default=2048)
    parser.add_argument("--working-set", type=int, default=4096)
    parser.add_argument("--policies", nargs="+",
                        default=["keep_hbm", "keep_dram", "recompute_full",
                                 "recompile_active"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.model:
        raise SystemExit("--model is required (or set EPHEMERALKV_MODEL)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # The measurement itself is the next step: it needs (a) a session builder that
    # appends `turn_tokens` per turn, (b) a working-set compiler the policies share, and
    # (c) timing/memory accounting per policy.  Writing the JSON shape first keeps the
    # gate's decision rule checkable while the harness is filled in.
    payload = {
        "schema": "ephemeral-kv-kill-gate-crossover-v0",
        "status": "not-implemented",
        "config": vars(args),
        "rows": [],
        "decision_rule": ("lives if recompile_active beats keep_* on time-to-response "
                          "for some T <= 512, or beats recompute_full by a factor that "
                          "grows with T"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out} (status: {payload['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
