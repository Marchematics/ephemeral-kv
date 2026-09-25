#!/usr/bin/env python
"""Measure the indexed lookup the runtime pays, on real transcripts, at each history length.

The mobility law is `lookup + active-set prefill` against moving the session's KV, and the G3
receipt says where its own boundary is: "Indexed lookup must be measured on the real trace
separately before composing end-to-end mobility tax."  The corpus's real sessions stop at ~156K
tokens, so the 1M lookup row in the paper has been an extrapolation from postings growth.

This measures it instead, on the composed long histories (`build_composed_long_sessions.py`), with
the same query the runtime would use: the most recent user or tool message before the final turn.
Lookup is a CPU operation over the durable index - the thing a cold route pays before it can
prefill anything - so it is measured here without a GPU, with repeats, and reported as p50/p95.

    python benchmarks/g2_index_lookup.py --jsonl data/sessions-composed-1m.jsonl \
        --out artifacts/g2-index-lookup-composed-1m-v1.json --repeats 20
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

from ephemeralkv.index import DurableSpanIndex            # noqa: E402
from benchmarks.g2_model_quality import classify          # noqa: E402
from benchmarks.g2_trace_index import _content, messages_from_row  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct",
                   help="only its tokenizer is used: this is a CPU measurement")
    p.add_argument("--max-examples", type=int, default=4)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--max-spans", type=int, default=128)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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
            index = DurableSpanIndex()
            for turn, message in enumerate(messages):
                text = _content(message)
                index.append(turn=turn, role=str(message.get("role", "?")), text=text,
                             token_estimate=max(1, len(tokenizer.encode(text,
                                                                        add_special_tokens=False))),
                             kind=classify(text))
            query = ""
            for earlier in reversed(messages[:-1]):
                if str(earlier.get("role")) in {"user", "tool"}:
                    query = _content(earlier)
                    break
            if not query:
                continue
            times, candidates = [], 0
            for _ in range(args.repeats):
                start = time.perf_counter()
                spans, _stats = index.lookup(query, max_spans=args.max_spans)
                times.append((time.perf_counter() - start) * 1000.0)
                candidates = max(candidates, len(spans))
            rows.append({
                "session_id": row.get("session_id"),
                "history_tokens": sum(span.token_estimate for span in index.spans),
                "spans": len(index.spans),
                "query_tokens": len(tokenizer.encode(query, add_special_tokens=False)),
                "lookup_ms_p50": round(statistics.median(times), 4),
                "lookup_ms_p95": round(sorted(times)[int(0.95 * len(times)) - 1], 4),
                "lookup_ms_min": round(min(times), 4),
                "spans_returned": candidates,
            })
            print(f"[lookup] {row.get('session_id')} history={rows[-1]['history_tokens']:,} "
                  f"p50={rows[-1]['lookup_ms_p50']} ms", flush=True)
            if len(rows) >= args.max_examples:
                break

    payload = {
        "schema": "ephemeral-kv-index-lookup-v1",
        "kind": "measured_primitive",
        "what": "durable-index lookup (the cold route's first operation) on real transcripts",
        "config": {k: v for k, v in sorted(vars(args).items()) if k != "out"},
        "interpretation": ("indexed lookup is CPU work over the durable index; it is the term the "
                           "mobility law adds to active-set prefill, and it is what the paper "
                           "previously extrapolated to 1M"),
        "rows": rows,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
