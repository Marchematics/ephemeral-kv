#!/usr/bin/env python
"""How much of a long session is state that is never needed again?

The abstraction says the durable session and the execution state are different objects.  Part of
the reason is arithmetic: a coding-agent transcript is not thousands of comparable turns - it is a
handful of very large blocks (the task prompt with the workspace, a whole-file dump, a long plan)
plus a long tail of small turns.  A view that carries the session's *footprint* therefore carries
mostly state that was needed once, which is the cheapest possible thing for a compiler to drop and
the most expensive thing for a cache to keep moving.

This reports, per session, the token share held by the largest spans, so the claim "most of the
durable history is dead state" is a measurement rather than an impression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.g2_model_quality import classify
from benchmarks.g2_trace_index import _content, messages_from_row
import sys

# the repository root, so a direct `python benchmarks/<harness>.py` run works from a
# clone without an exported PYTHONPATH; the runner scripts set it as well
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ephemeralkv.index import DurableSpanIndex


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", default="data/sessions-64k.jsonl")
    p.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    p.add_argument("--max-sessions", type=int, default=8)
    p.add_argument("--min-history-tokens", type=int, default=32768)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731

    rows = []
    with open(args.jsonl) as fh:
        for line in fh:
            if len(rows) >= args.max_sessions:
                break
            try:
                messages = messages_from_row(json.loads(line))
            except ValueError:
                continue
            index = DurableSpanIndex()
            for turn, msg in enumerate(messages):
                text = _content(msg)
                index.append(turn=turn, role=str(msg.get("role", "?")), text=text,
                             token_estimate=max(1, counter(text)), kind=classify(text))
            sizes = sorted((s.token_estimate for s in index.spans), reverse=True)
            total = sum(sizes)
            if total < args.min_history_tokens:
                continue
            rows.append({
                "history_tokens": total,
                "spans": len(sizes),
                "largest_span": sizes[0],
                "top2_share": round(sum(sizes[:2]) / total, 3),
                "top5_share": round(sum(sizes[:5]) / total, 3),
                "tail_share": round(sum(sizes[5:]) / total, 3),
            })

    payload = {"schema": "ephemeral-kv-dead-state-v1", "kind": "structural_measurement",
               "interpretation": ("the share of a session's tokens held by its largest spans; the "
                                  "tail is what a turn-by-turn view actually walks"),
               "rows": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{'history':>9} {'spans':>6} {'largest':>8} {'top2':>7} {'top5':>7} {'tail':>7}")
    for row in rows:
        print(f"{row['history_tokens']:>9} {row['spans']:>6} {row['largest_span']:>8} "
              f"{100 * row['top2_share']:>6.0f}% {100 * row['top5_share']:>6.0f}% "
              f"{100 * row['tail_share']:>6.0f}%")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
