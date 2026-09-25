#!/usr/bin/env python
"""Does the evidence a query needs grow with the session?  (structural, no model)

The killer table measures the *state* and the *quality* at a fixed budget.  This measures the
quantity underneath them without a GPU: for each turn, how many tokens of the history actually
share terms with the query, and how large is the best-ranked span.  If those are flat in session
length then `|E_q|` is a property of the query rather than of the session's age, which is the
structural counterpart of `dM/dL ~ 0` and is measurable over hundreds of sessions instead of
dozens.

Cheap by construction: one pass per session, term overlap against the query, no model.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# the repository root, so a direct `python benchmarks/<harness>.py` run works from a
# clone without an exported PYTHONPATH; the runner scripts set it as well
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.g2_model_quality import classify
from benchmarks.g2_trace_index import _content, messages_from_row
import sys


from ephemeralkv.index import DurableSpanIndex, terms

BUCKETS = ((0, 32768, "<=32K"), (32768, 65536, "32K-64K"), (65536, 131072, "64K-128K"),
           (131072, 262144, "128K-256K"), (262144, 1 << 62, ">=256K"))


def median(xs):
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", default="data/sessions-64k.jsonl")
    p.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    p.add_argument("--max-sessions", type=int, default=300)
    p.add_argument("--min-history-tokens", type=int, default=32768)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    rows = []
    with Path(args.jsonl).open() as handle:
        for line in handle:
            if not line.strip() or len(rows) >= args.max_sessions * 8:
                break
            try:
                messages = messages_from_row(json.loads(line))
            except ValueError:
                continue
            index = DurableSpanIndex()
            history = []
            for turn, msg in enumerate(messages):
                role = str(msg.get("role", "?"))
                text = _content(msg)
                index.append(turn=turn, role=role, text=text,
                             token_estimate=max(1, counter(text)), kind=classify(text))
                history.append(msg)
                if role != "assistant" or len(index.spans) < 8:
                    continue
                query = ""
                for previous in reversed(history[:-1]):
                    if str(previous.get("role", "")) in {"user", "tool"}:
                        query = _content(previous)
                        break
                if not query:
                    continue
                query_terms = set(terms(query))
                if not query_terms:
                    continue
                history_tokens = sum(max(1, s.token_estimate) for s in index.spans)
                if history_tokens < args.min_history_tokens:
                    continue
                # rare-term overlap: a term that appears in most spans ("the file") carries no
                # information about which spans matter, so relevance is counted only through terms
                # whose document frequency is a minority of the session's spans
                span_terms = [set(terms(span.text)) for span in index.spans]
                document_frequency: dict[str, int] = {}
                for shared_terms in span_terms:
                    for term in shared_terms:
                        document_frequency[term] = document_frequency.get(term, 0) + 1
                cutoff = max(2, int(0.10 * len(span_terms)))
                rare_query_terms = {t for t in query_terms
                                    if document_frequency.get(t, 0) <= cutoff}
                relevant, strong = 0, 0
                best = 0
                for span, shared_terms in zip(index.spans, span_terms):
                    shared = len(shared_terms & rare_query_terms)
                    if shared:
                        relevant += max(1, span.token_estimate)
                    if shared >= 2:
                        strong += max(1, span.token_estimate)
                        best = max(best, max(1, span.token_estimate))
                rows.append({"history_tokens": history_tokens, "turns": len(history),
                             "query_terms": len(query_terms),
                             "rare_query_terms": len(rare_query_terms),
                             "relevant_tokens": relevant, "strongly_relevant_tokens": strong,
                             "best_span_tokens": best})
                if len(rows) >= args.max_sessions:
                    break
            if len(rows) >= args.max_sessions:
                break

    by_bucket: dict[str, list[dict]] = {}
    for row in rows:
        by_bucket.setdefault(bucket_of(row["history_tokens"]), []).append(row)
    table = {}
    for _low, _high, label in BUCKETS:
        group = by_bucket.get(label)
        if not group:
            table[label] = {"n": 0}
            continue
        table[label] = {
            "n": len(group),
            "turns_p50": int(median([r["turns"] for r in group])),
            "history_p50": int(median([r["history_tokens"] for r in group])),
            "relevant_tokens_p50": int(median([r["relevant_tokens"] for r in group])),
            "strongly_relevant_tokens_p50": int(median([r["strongly_relevant_tokens"]
                                                        for r in group])),
            "best_span_tokens_p50": int(median([r["best_span_tokens"] for r in group])),
        }
    payload = {"schema": "ephemeral-kv-evidence-mass-v1", "kind": "structural_measurement",
               "interpretation": ("tokens of the history that share terms with the query; if this "
                                  "is flat in session length, the evidence a turn needs is a "
                                  "property of the query rather than of the session's age"),
               "rows": len(rows), "by_history_bucket": table}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'bucket':<10} {'n':>4} {'turns':>6} {'history':>9} {'relevant':>9} "
          f"{'strong':>8} {'best span':>10}")
    for label, entry in table.items():
        if entry.get("n"):
            print(f"{label:<10} {entry['n']:>4} {entry['turns_p50']:>6} {entry['history_p50']:>9} "
                  f"{entry['relevant_tokens_p50']:>9} {entry['strongly_relevant_tokens_p50']:>8} "
                  f"{entry['best_span_tokens_p50']:>10}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
