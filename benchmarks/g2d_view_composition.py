#!/usr/bin/env python
"""What each arm's view actually shows the model, on the turns the action metric scored.

The mid-session action measurement found a split it could not explain: the bounded state emits an
executable action far more often than plain retrieval and less often than the full transcript.  This
counts the obvious candidate - how many prior *actions* each view contains, as opposed to how many
relevant spans.

It matters because the three arms differ in that count by an order of magnitude.  The full transcript
carries every action the session has taken (measured: 34-67 by mid-session); the bounded state
carries whatever the verbatim window happens to cover (6-8); and query-ranked retrieval carries
almost none (0-1), because its ranking selects *evidence* - tool output, file contents - and prior
assistant actions are not usually what the query terms match.  If a model needs to see the action
format to produce one, then a retrieval-only view is not a context-management design with a quality
cost, it is a context-management design that removes the thing being asked for.

The counts are read from the same view builder the measurement used, driven by the receipt's own
recorded configuration, so this describes the views that produced the numbers rather than a
reconstruction of them.

Usage:
    python benchmarks/g2d_view_composition.py \\
        --receipt artifacts/g2d-action-turns-state4096-v1.json \\
        --out artifacts/g2d-view-composition-v1.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import statistics
from pathlib import Path

from benchmarks.g2_model_quality import build_examples
from benchmarks.g2_trace_index import messages_from_row
from benchmarks.g2c_action_reproduction import parse_action
from benchmarks.g2d_action_turns import RETRIEVAL_COMPILER


def count_actions(text: str) -> int:
    """Tool invocations visible in a rendered context."""
    return sum(1 for block in text.split("<assistant>")[1:] if parse_action(block))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--receipt", default="artifacts/g2d-action-turns-state4096-v1.json")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    receipt = json.loads(Path(args.receipt).read_text())
    config = receipt["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    wanted: dict[str, set] = {}
    for row in receipt["rows"]:
        wanted.setdefault(row["session_id"], set()).add(row["turn_index"])

    rows = []
    with Path(config["jsonl"]).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            session_id = row.get("session_id")
            if session_id not in wanted:
                continue
            meta = json.loads(row.get("ground_truth_meta_json") or "{}")
            if not meta.get("patch_present"):
                continue
            messages = messages_from_row(row)
            shipped = build_examples(
                messages, session_id=session_id, token_budget=config["token_budget"],
                min_history_tokens=config["min_history_tokens"],
                compiler={"recency_spans": config["recency_spans"],
                          "recency_fraction": config["recency_fraction"],
                          "max_span_fraction": config["max_span_fraction"],
                          "provenance_terms": config["provenance_terms"],
                          "dedup": config["dedup_spans"], "snippet": config["snippet_spans"],
                          "consolidate": config["consolidate"],
                          "retrieve_multiplier": config["retrieve_multiplier"],
                          "collapse_paths": not config["no_collapse_paths"],
                          "compile_mode": config["compile_mode"],
                          "tail_fraction": config["tail_fraction"],
                          "tail_tokens": config["tail_tokens"], "summary_tokens": 512,
                          "summary_input_tokens": 8192, "summary_stride": 8,
                          "tail_cap": config["tail_cap"],
                          "far_compiler": config["far_compiler"]},
                token_counter=counter)
            retrieval = build_examples(
                messages, session_id=session_id, token_budget=config["token_budget"],
                min_history_tokens=config["min_history_tokens"],
                compiler=dict(RETRIEVAL_COMPILER), token_counter=counter)
            for index in sorted(wanted[session_id]):
                if index >= len(shipped) or index >= len(retrieval):
                    continue
                scored = next(r for r in receipt["rows"]
                              if r["session_id"] == session_id and r["turn_index"] == index)
                rows.append({
                    "session_id": session_id,
                    "turn_index": index,
                    "history_tokens": scored["history_tokens"],
                    "full_actions_in_view": count_actions(shipped[index].full_context),
                    "active_actions_in_view": count_actions(shipped[index].active_context),
                    "retrieval_actions_in_view": count_actions(retrieval[index].active_context),
                    "full_emits": scored["full"]["parseable"],
                    "active_emits": scored["active"]["parseable"],
                    "retrieval_emits": scored["retrieval"]["parseable"],
                })
                print(f"[g2d-view] {session_id[:40]} turn {index}: actions "
                      f"full={rows[-1]['full_actions_in_view']} "
                      f"state={rows[-1]['active_actions_in_view']} "
                      f"raw={rows[-1]['retrieval_actions_in_view']}", flush=True)

    def p50(key):
        values = [r[key] for r in rows]
        return statistics.median(values) if values else None

    summary = {
        "turns": len(rows),
        "action_examples_in_view_p50": {
            "full": p50("full_actions_in_view"),
            "active": p50("active_actions_in_view"),
            "retrieval": p50("retrieval_actions_in_view"),
        },
        "action_examples_in_view_min": {
            "full": min((r["full_actions_in_view"] for r in rows), default=None),
            "active": min((r["active_actions_in_view"] for r in rows), default=None),
            "retrieval": min((r["retrieval_actions_in_view"] for r in rows), default=None),
        },
        "turns_whose_retrieval_view_shows_no_action":
            sum(1 for r in rows if r["retrieval_actions_in_view"] == 0),
    }
    payload = {
        "schema": "ephemeral-kv-g2d-view-composition-v1",
        "kind": "derived_measurement",
        "source_receipt": args.receipt,
        "summary": summary,
        "rows": rows,
        "reading": (
            "the three arms differ in how many prior actions their view contains by an order of "
            "magnitude, which is the mechanism behind the action metric's split rather than a "
            "coincidence with it: a view assembled for *evidence* does not contain the format the "
            "continuation is supposed to be in"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
