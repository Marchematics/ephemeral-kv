#!/usr/bin/env python
"""G2 structural trace harness: can indexed lookup avoid a history scan?

This harness measures lookup structure only. It does **not** establish answer/task
quality.

It consumes JSONL trajectories with a messages field plus optional session metadata.
For each user/tool message, lookup runs against prior transcript spans and records:

* accumulated history tokens under the declared tokenizer;
* selected active-view tokens under a declared budget;
* postings visited versus stored spans;
* lookup wall time;
* official per-session max-ISL bucket when supplied by the dataset.

The public ThoughtWorks corpus is tokenized with cl100k_base. Use
`--tokenizer cl100k_base` for the calibrated run. The default regex counter exists only
for dependency-free unit tests and generic traces.

A favorable structural result advances only the lookup-complexity half of G2. G2 stays
open until a real model/task quality run validates the selected view.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

from ephemeralkv.index import DurableSpanIndex, terms


TokenCounter = Callable[[str], int]


def _content(msg) -> str:
    value = msg.get("content", "") if isinstance(msg, dict) else ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for x in value:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                text = x.get("text") or x.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(value)


def messages_from_row(row: dict) -> list[dict]:
    if isinstance(row.get("messages_json"), str):
        return json.loads(row["messages_json"])
    if isinstance(row.get("messages"), list):
        return row["messages"]
    if isinstance(row.get("trajectory"), list):
        return row["trajectory"]
    raise ValueError("row has no supported messages_json/messages/trajectory field")


def regex_token_estimate(text: str) -> int:
    return max(1, len(terms(text)))


# Backward-compatible name used by early tests/code.
approx_tokens = regex_token_estimate


def make_token_counter(name: str) -> TokenCounter:
    if name == "regex":
        return regex_token_estimate
    if name == "cl100k_base":
        try:
            import tiktoken
        except ImportError as e:
            raise SystemExit(
                "cl100k_base accounting requires optional dependency 'tiktoken'"
            ) from e
        enc = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            # encode_ordinary treats any special-looking strings in tool output as text.
            return max(1, len(enc.encode_ordinary(text)))

        return count
    raise ValueError(f"unknown tokenizer: {name}")


def percentile(xs: list[float], q: float):
    if not xs:
        return None
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def history_bucket(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    if tokens <= 8_000:
        return "<=8K"
    if tokens <= 16_000:
        return "8K-16K"
    if tokens <= 32_000:
        return "16K-32K"
    if tokens <= 64_000:
        return "32K-64K"
    if tokens <= 128_000:
        return "64K-128K"
    return ">128K"


# Backward-compatible alias.
bucket = history_bucket


def evaluate_session(
    messages: list[dict],
    token_budget: int,
    max_spans: int = 64,
    *,
    token_counter: TokenCounter = regex_token_estimate,
    session_max_isl: int | None = None,
):
    idx = DurableSpanIndex()
    history_tokens = 0
    rows = []

    for turn, msg in enumerate(messages):
        role = str(msg.get("role", "unknown"))
        text = _content(msg)
        tok = token_counter(text)

        if role in {"user", "tool"} and idx.spans:
            t0 = perf_counter()
            view, stats = idx.compile_view(
                text, token_budget=token_budget, max_spans=max_spans
            )
            dt = perf_counter() - t0
            selected = sum(s.token_estimate for s in view)
            rows.append(
                {
                    "turn": turn,
                    "role": role,
                    "history_token_estimate": history_tokens,
                    "history_spans": stats.total_spans,
                    "selected_token_estimate": selected,
                    "selected_fraction": (
                        selected / history_tokens if history_tokens else 0.0
                    ),
                    "postings_visited": stats.postings_visited,
                    "postings_per_span": (
                        stats.postings_visited / stats.total_spans
                        if stats.total_spans
                        else 0.0
                    ),
                    "candidate_spans": stats.candidate_spans,
                    "lookup_ms": dt * 1000.0,
                    "history_bucket": history_bucket(history_tokens),
                    "session_max_isl": session_max_isl,
                    "session_max_isl_bucket": history_bucket(session_max_isl),
                }
            )

        idx.append(turn=turn, role=role, text=text, token_estimate=tok)
        history_tokens += tok

    return rows


def summarize(rows: list[dict]) -> dict:
    by_history: dict[str, list[dict]] = {}
    by_session_max: dict[str, list[dict]] = {}
    for row in rows:
        by_history.setdefault(row["history_bucket"], []).append(row)
        by_session_max.setdefault(row["session_max_isl_bucket"], []).append(row)

    def agg(rs):
        selected = [r["selected_fraction"] for r in rs]
        scan = [r["postings_per_span"] for r in rs]
        postings = [float(r["postings_visited"]) for r in rs]
        lookup = [r["lookup_ms"] for r in rs]
        history = [float(r["history_token_estimate"]) for r in rs]
        return {
            "calls": len(rs),
            "history_tokens_p50": median(history) if history else None,
            "history_tokens_p95": percentile(history, 0.95),
            "selected_fraction_p50": median(selected) if selected else None,
            "selected_fraction_p95": percentile(selected, 0.95),
            "postings_visited_p50": median(postings) if postings else None,
            "postings_visited_p95": percentile(postings, 0.95),
            "postings_per_span_p50": median(scan) if scan else None,
            "postings_per_span_p95": percentile(scan, 0.95),
            "lookup_ms_p50": median(lookup) if lookup else None,
            "lookup_ms_p95": percentile(lookup, 0.95),
        }

    return {
        "calls": len(rows),
        "overall": agg(rows),
        "by_history_bucket": {k: agg(v) for k, v in sorted(by_history.items())},
        "by_session_max_isl_bucket": {
            k: agg(v) for k, v in sorted(by_session_max.items())
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--token-budget", type=int, default=4096)
    p.add_argument("--max-spans", type=int, default=64)
    p.add_argument("--max-sessions", type=int, default=0, help="0 means all rows")
    p.add_argument(
        "--tokenizer",
        choices=["regex", "cl100k_base"],
        default="regex",
        help="history/selected token accounting; index terms remain model-independent",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    counter = make_token_counter(args.tokenizer)
    all_rows = []
    sessions = 0
    source_counts: dict[str, int] = {}

    with Path(args.jsonl).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                messages = messages_from_row(row)
            except ValueError:
                continue

            max_isl = row.get("max_isl")
            if max_isl is not None:
                max_isl = int(max_isl)

            all_rows.extend(
                evaluate_session(
                    messages,
                    args.token_budget,
                    args.max_spans,
                    token_counter=counter,
                    session_max_isl=max_isl,
                )
            )
            source = str(row.get("source_dataset", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
            sessions += 1
            if args.max_sessions and sessions >= args.max_sessions:
                break

    payload = {
        "schema": "ephemeral-kv-g2-structural-v2",
        "kind": "structural_measurement",
        "input": args.jsonl,
        "sessions": sessions,
        "source_session_counts": source_counts,
        "token_budget": args.token_budget,
        "tokenizer": args.tokenizer,
        "summary": summarize(all_rows),
        "quality_status": "not_measured",
        "decision_rule": (
            "this harness can only advance the lookup-complexity half of G2; "
            "G2 remains open until real task/model quality is measured"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
