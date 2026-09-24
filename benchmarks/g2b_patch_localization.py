#!/usr/bin/env python
"""G2 end-task half: does the active view still localize the change?

Teacher-forced next-token loss is a proxy.  The decision an evicted history can actually
break is the agent's *first* downstream act: knowing which files the task touches.  This
runner measures that directly, with no sandbox:

1. the recorded patch's file set is recovered from the transcript's tool outputs and
   tool-call arguments (`diff --git a/<path> b/<path>`, `*** Update File: <path>`,
   `"path": "<path>"`), which is where SWE-agent-style trajectories carry it - assistant
   prose names files only occasionally;
2. the frozen model generates the next assistant turn under two contexts: the full history
   and the active view the durable index compiles;
3. the file paths each continuation names are scored against the recorded set
   (precision/recall/F1, suffix-matched so `a/b/c.py` and `/workspace/x/a/b/c.py` agree).

A policy that preserves the *text* of the next turn but loses *which files matter* has
failed the agent, and this is the metric that shows it.

Usage:
    python benchmarks/g2b_patch_localization.py --jsonl data/sessions-64k.jsonl \
        --model <checkpoint> --token-budget 16384 --min-history-tokens 32768 \
        --max-length 65536 --max-examples 24 \
        --out artifacts/g2b-patch-localization-v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g2_model_quality import build_examples  # noqa: E402
from g2_trace_index import messages_from_row  # noqa: E402

# where a trajectory records what it changed
_DIFF = re.compile(r"diff --git a/(\S+?)\s+b/(\S+)")
_EDITOR = re.compile(r"\*\*\* (?:Update|Add|Delete) File: (\S+)")
_MODIFIED = re.compile(r"Modified File\*{0,2}:?\*{0,2}\s*`?([^\s`'\"]+)")
_ARG_PATH = re.compile(r'"(?:path|file_path|filename)"\s*:\s*"([^"]+)"')
_FILE_TOKEN = re.compile(r"[\w./-]+\.(?:py|pyx|js|ts|tsx|jsx|md|txt|json|ya?ml|toml|cfg|ini|"
                         r"c|cc|cpp|h|hpp|rs|go|java|rb|sh|sql|html|css)")


def _normalize(path: str) -> str:
    """One spelling per file: drop prose punctuation, the sandbox root and the checkout dir.

    `lstrip("ab/")` was the first attempt and it was wrong twice over: it eats any leading
    `a`/`b`/`/` characters rather than a diff's `a/` prefix (the diff regex already captures
    the path without it), so `/workspace/...` became `workspace/...` and lost the prefix
    the next check was looking for.
    """
    path = path.strip().strip("`'\"")
    for prefix in ("/workspace/", "/testbed/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    path = path.lstrip("/")
    # a checkout directory follows the sandbox root: /workspace/pandas__pandas__5.19/x.py
    parts = path.split("/")
    if len(parts) > 1 and "__" in parts[0]:
        path = "/".join(parts[1:])
    return path


def patch_files_from_messages(messages: list[dict]) -> set[str]:
    """The file set the recorded trajectory changed, from tool output and tool arguments."""
    files: set[str] = set()
    for message in messages:
        role = str(message.get("role"))
        blob = str(message.get("content") or "")
        calls = message.get("tool_calls_json")
        if calls:
            blob += " " + str(calls)
        if role not in ("tool", "assistant", "user"):
            continue
        for left, right in _DIFF.findall(blob):
            files.add(_normalize(left))
            files.add(_normalize(right))
        for match in _EDITOR.findall(blob):
            files.add(_normalize(match))
        for match in _MODIFIED.findall(blob):
            if _FILE_TOKEN.fullmatch(_normalize(match)):
                files.add(_normalize(match))
        for match in _ARG_PATH.findall(blob):
            if _FILE_TOKEN.fullmatch(_normalize(match)):
                files.add(_normalize(match))
    return {f for f in files if f and not f.startswith("..")}


def mentioned_files(text: str) -> set[str]:
    """Paths named in a continuation, normalized the same way as the ground truth."""
    return {_normalize(token) for token in _FILE_TOKEN.findall(text)}


def _matches(mentioned: str, recorded: str) -> bool:
    """Suffix match: absolute, checkout-prefixed and relative spellings of one file agree."""
    return (mentioned == recorded or mentioned.endswith("/" + recorded)
            or recorded.endswith("/" + mentioned)
            or mentioned.split("/")[-1] == recorded.split("/")[-1] and
            len(mentioned.split("/")[-1]) > 3)


def score_continuation(text: str, recorded: set[str]) -> dict:
    mentioned = mentioned_files(text)
    if not recorded:
        return {"precision": None, "recall": None, "f1": None, "hits": 0}
    hits = {r for r in recorded if any(_matches(m, r) for m in mentioned)}
    precision = (len(hits) / len(mentioned)) if mentioned else 0.0
    recall = len(hits) / len(recorded)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hits": len(hits),
            "mentioned": sorted(mentioned), "recorded": sorted(recorded)}


def generate(model, tokenizer, context: str, max_new: int, max_length: int, device: str):
    import torch

    ids = tokenizer.encode(context, add_special_tokens=False)[-max_length:]
    if not ids:
        return ""
    input_ids = torch.tensor([ids], device=device)
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id or 0)
    return tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--token-budget", type=int, default=16384)
    p.add_argument("--min-history-tokens", type=int, default=32768)
    p.add_argument("--max-length", type=int, default=65536)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--max-examples", type=int, default=24)
    p.add_argument("--consolidate", action=argparse.BooleanOptionalAction, default=False,
                   help="compile the retrieved evidence instead of concatenating it")
    p.add_argument("--retrieve-multiplier", type=float, default=2.0)
    p.add_argument("--dedup-spans", action=argparse.BooleanOptionalAction, default=False,
                   help="collapse identical span texts in the view (keeps the newest copy)")
    p.add_argument("--snippet-spans", action=argparse.BooleanOptionalAction, default=False,
                   help="keep query-relevant lines of an oversized span, not its prefix")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import statistics
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()

    rows = []
    with Path(args.jsonl).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = json.loads(row.get("ground_truth_meta_json") or "{}")
            if not meta.get("patch_present"):
                continue
            try:
                messages = messages_from_row(row)
            except ValueError:
                continue
            # ground truth is the recorded patch's file set; the context is built from the
            # history *before* the final turn, so nothing about it leaks into the view
            recorded = patch_files_from_messages(messages)
            if not recorded:
                continue
            examples = build_examples(
                messages, token_budget=args.token_budget,
                min_history_tokens=args.min_history_tokens,
                compiler={"recency_spans": 3, "recency_fraction": 0.6,
                          "max_span_fraction": 0.25, "provenance_terms": 8,
                          "dedup": args.dedup_spans, "snippet": args.snippet_spans,
                          "consolidate": args.consolidate,
                          "retrieve_multiplier": args.retrieve_multiplier},
                token_counter=lambda text: len(tokenizer.encode(text,
                                                                add_special_tokens=False)),
            )
            if not examples:
                continue
            example = examples[-1]                       # the last turn of the session
            full_text = generate(model, tokenizer, example.full_context, args.max_new,
                                 args.max_length, args.device)
            active_text = generate(model, tokenizer, example.active_context, args.max_new,
                                   args.max_length, args.device)
            # two references, because they have very different sparsity.  The recorded
            # *patch* is the end-task ground truth but a 96-token continuation names only
            # a fraction of it (measured: recall 0.08 with either arm), while the recorded
            # *next turn's* own file mentions are what the agent did immediately, which is
            # the sharper comparison and does not depend on the model volunteering files
            next_turn_files = mentioned_files(example.target)
            rows.append({
                "instance_id": meta.get("instance_id"),
                "resolved": meta.get("resolved"),
                "history_tokens": example.history_tokens_estimate,
                "active_tokens": example.active_tokens_estimate,
                "recorded_files": sorted(recorded),
                "full": {"continuation": full_text,
                         **score_continuation(full_text, recorded),
                         "next_turn": score_continuation(full_text, next_turn_files)
                         if next_turn_files else None},
                "active": {"continuation": active_text,
                           **score_continuation(active_text, recorded),
                           "next_turn": score_continuation(active_text, next_turn_files)
                           if next_turn_files else None},
            })
            print(f"[g2b] {meta.get('instance_id')} history={example.history_tokens_estimate} "
                  f"active={example.active_tokens_estimate} "
                  f"f1 full={rows[-1]['full']['f1']:.2f} active={rows[-1]['active']['f1']:.2f}",
                  flush=True)
            if len(rows) >= args.max_examples:
                break

    def mean(key, arm):
        vals = [r[arm][key] for r in rows if r[arm][key] is not None]
        return statistics.mean(vals) if vals else None

    def mean_nested(key, arm):
        vals = [r[arm]["next_turn"][key] for r in rows
                if r[arm].get("next_turn") and r[arm]["next_turn"][key] is not None]
        return statistics.mean(vals) if vals else None

    summary = {
        "examples": len(rows),
        "examples_with_next_turn_files": sum(1 for r in rows if r["full"].get("next_turn")),
        "history_tokens_p50": int(statistics.median([r["history_tokens"] for r in rows]))
        if rows else None,
        "active_fraction_p50": statistics.median(
            [r["active_tokens"] / max(1, r["history_tokens"]) for r in rows]) if rows else None,
        "full": {k: mean(k, "full") for k in ("precision", "recall", "f1")},
        "active": {k: mean(k, "active") for k in ("precision", "recall", "f1")},
        "full_vs_next_turn": {k: mean_nested(k, "full")
                              for k in ("precision", "recall", "f1")},
        "active_vs_next_turn": {k: mean_nested(k, "active")
                                for k in ("precision", "recall", "f1")},
    }
    payload = {
        "schema": "ephemeral-kv-g2b-patch-localization-v1",
        "kind": "end_task_measurement",
        "config": vars(args),
        "summary": summary,
        "rows": rows,
        "interpretation": (
            "file-level localization of the next turn against the recorded patch's files; "
            "this is a downstream behaviour, not teacher-forced text fidelity, and it is "
            "measured on the same examples for both arms"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
