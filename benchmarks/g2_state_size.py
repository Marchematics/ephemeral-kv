#!/usr/bin/env python
"""What must move: the compiled state's tokens, bytes and compile time at each history length.

The claim under test is "the session keeps growing; the state that must move does not".  Tokens are
one way to state it, and the model harnesses measure those; but a cold route moves *bytes* and pays
*compile time*, and none of the three is a GPU measurement - the compile path is CPU work over the
durable index, so it can be measured directly, on real transcripts, at any history length.

This uses the shipped view configuration (a whole newest window over a consolidated far field at an
8,192-token budget) through the harness's own example builder, so the view measured here is the view
the quality arms measured - the difference is only that no model is loaded.

    python benchmarks/g2_state_size.py --jsonl data/sessions-composed-1m.jsonl \
        --min-history-tokens 900000 --out artifacts/g2-state-size-composed-1m-v1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

# the repository root, so a direct `python benchmarks/<harness>.py` run works from a clone without
# an exported PYTHONPATH; the runner scripts set it as well
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from benchmarks.g2_model_quality import build_examples      # noqa: E402
from benchmarks.g2_trace_index import messages_from_row      # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct",
                   help="only its tokenizer is used: this is a CPU measurement")
    p.add_argument("--max-examples", type=int, default=4, help="sessions to measure")
    p.add_argument("--min-history-tokens", type=int, default=0,
                   help="only turns whose history exceeds this; set it near the bucket size so the "
                        "measurement lands on the longest histories in each session")
    p.add_argument("--token-budget", type=int, default=8192)
    p.add_argument("--tail-tokens", type=int, default=4096)
    p.add_argument("--compile-mode", default="tail_state")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeats of the whole CPU path (each rebuilds the index)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731

    compiler = {"recency_spans": 3, "recency_fraction": 0.6, "max_span_fraction": 0.25,
                "provenance_terms": 8, "dedup": True, "snippet": False, "consolidate": True,
                "retrieve_multiplier": 2, "collapse_paths": True,
                "compile_mode": args.compile_mode, "tail_fraction": 0.6,
                "tail_tokens": args.tail_tokens, "summary_tokens": 512,
                "summary_input_tokens": 8192, "summary_stride": 8, "tail_cap": 0.5,
                "far_compiler": "consolidate", "keep_earlier_verbatim": False}

    rows = []
    with Path(args.jsonl).open() as handle:
        for line in handle:
            if not line.strip() or len(rows) >= args.max_examples:
                continue
            row = json.loads(line)
            try:
                messages = messages_from_row(row)
            except ValueError:
                continue
            times = []
            examples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                examples = build_examples(messages, token_budget=args.token_budget,
                                          max_spans=128,
                                          min_history_tokens=args.min_history_tokens,
                                          compiler=compiler, token_counter=counter)
                times.append((time.perf_counter() - start) * 1000.0)
            if not examples:
                continue
            for example in examples:
                rows.append({
                    "session_id": row.get("session_id"),
                    "history_tokens": example.history_tokens_estimate,
                    "state_tokens": example.active_tokens_estimate,
                    "state_bytes": len(example.active_context.encode("utf-8")),
                    "cpu_path_ms_p50": round(statistics.median(times), 1),
                    "turns": getattr(example, "turns", None),
                })
            print(f"[state] {row.get('session_id')} history={rows[-1]['history_tokens']:,} "
                  f"state={rows[-1]['state_tokens']:,} tokens / {rows[-1]['state_bytes']:,} bytes "
                  f"cpu_path={rows[-1]['cpu_path_ms_p50']} ms", flush=True)
            if len(rows) >= args.max_examples:
                break

    payload = {
        "schema": "ephemeral-kv-state-size-v1",
        "kind": "measured_primitive",
        "what": ("tokens and bytes of the shipped compiled state (what a cold route transfers), "
                 "measured through the harness's own view builder with no model loaded, plus the "
                 "CPU wall time of that builder - index build, scan and view - which is an upper "
                 "bound on the CPU side of a cold route, since the runtime reads a stored index"),
        "config": {k: v for k, v in sorted(vars(args).items()) if k != "out"},
        "rows": rows,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
