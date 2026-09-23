#!/usr/bin/env python
"""G2 structural trace harness: does indexed lookup avoid history scans?

This harness does NOT measure answer quality. It consumes JSONL trajectories and
reports only structural quantities needed before a model run:

* accumulated history size (word-token estimate),
* selected active-view size under a declared budget,
* index postings visited versus total stored spans,
* lookup wall time,
* scaling buckets by session age.

Supported rows:
* thoughtworks/agentic-coding-trajectories: messages_json string;
* common trajectory dumps: messages or trajectory list.

For each user/tool message, lookup is performed against prior transcript spans; then
the current message is appended. System/assistant messages are also indexed when seen.

A favorable result is only a complexity signal. G2 remains open until the selected view
is fed to a real model/task and quality is measured.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from ephemeralkv.index import DurableSpanIndex, terms


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


def approx_tokens(text: str) -> int:
    return max(1, len(terms(text)))


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


def bucket(history_tokens: int) -> str:
    for limit, name in (
        (8_000, "<=8K"),
        (32_000, "8K-32K"),
        (128_000, "32K-128K"),
        (512_000, "128K-512K"),
    ):
        if history_tokens <= limit:
            return name
    return ">512K"


def evaluate_session(messages: list[dict], token_budget: int, max_spans: int = 64):
    idx = DurableSpanIndex()
    history_tokens = 0
    rows = []

    for turn, msg in enumerate(messages):
        role = str(msg.get("role", "unknown"))
        text = _content(msg)
        tok = approx_tokens(text)

        if role in {"user", "tool"} and idx.spans:
            t0 = perf_counter()
            view, stats = idx.compile_view(
                text, token_budget=token_budget, max_spans=max_spans
            )
            dt = perf_counter() - t0
            selected = sum(s.token_estimate for s in view)
            rows.append({
                "turn": turn,
                "role": role,
                "history_token_estimate": history_tokens,
                "history_spans": stats.total_spans,
                "selected_token_estimate": selected,
                "selected_fraction": selected / history_tokens if history_tokens else 0.0,
                "postings_visited": stats.postings_visited,
                "postings_per_span": (
                    stats.postings_visited / stats.total_spans
                    if stats.total_spans else 0.0
                ),
                "candidate_spans": stats.candidate_spans,
                "lookup_ms": dt * 1000.0,
                "bucket": bucket(history_tokens),
            })

        idx.append(turn=turn, role=role, text=text, token_estimate=tok)
        history_tokens += tok

    return rows


def summarize(rows: list[dict]) -> dict:
    by_bucket: dict[str, list[dict]] = {}
    for row in rows:
        by_bucket.setdefault(row["bucket"], []).append(row)

    def agg(rs):
        selected = [r["selected_fraction"] for r in rs]
        scan = [r["postings_per_span"] for r in rs]
        lookup = [r["lookup_ms"] for r in rs]
        return {
            "calls": len(rs),
            "selected_fraction_p50": median(selected) if selected else None,
            "selected_fraction_p95": percentile(selected, .95),
            "postings_per_span_p50": median(scan) if scan else None,
            "postings_per_span_p95": percentile(scan, .95),
            "lookup_ms_p50": median(lookup) if lookup else None,
            "lookup_ms_p95": percentile(lookup, .95),
        }

    return {
        "calls": len(rows),
        "overall": agg(rows),
        "by_history_bucket": {k: agg(v) for k, v in sorted(by_bucket.items())},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--token-budget", type=int, default=4096)
    p.add_argument("--max-spans", type=int, default=64)
    p.add_argument("--max-sessions", type=int, default=0, help="0 means all rows")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    all_rows = []
    sessions = 0
    with Path(args.jsonl).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                messages = messages_from_row(row)
            except ValueError:
                continue
            all_rows.extend(evaluate_session(messages, args.token_budget, args.max_spans))
            sessions += 1
            if args.max_sessions and sessions >= args.max_sessions:
                break

    payload = {
        "schema": "ephemeral-kv-g2-structural-v1",
        "kind": "structural_measurement",
        "input": args.jsonl,
        "sessions": sessions,
        "token_budget": args.token_budget,
        "summary": summarize(all_rows),
        "quality_status": "not_measured",
        "decision_rule": (
            "this harness can only advance the lookup-complexity half of G2; "
            "G2 remains open until real task/model quality is measured"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
