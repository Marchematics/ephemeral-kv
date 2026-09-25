#!/usr/bin/env python
"""Mid-session next-action reproduction: does the state let the agent take its next real step?

The end task this project gates on asks whether a 96-token continuation *names* the files of the
recorded patch.  Naming a file is not doing the work, and the receipts that carry it generate at
`examples[-1]` - the last assistant turn of a session - which on a coding corpus is the agent
*submitting* (measured: 48 of 48, no file target; see `g2c_action_reproduction.py`).  So the
existing end task cannot say anything about the decision that actually matters: the next
**action** in the middle of a session, the editor call that changes a file.

This harness measures exactly that.  For each sampled assistant turn it builds the two contexts
the same way every other quality arm does - the full transcript, and the bounded state (a verbatim
window plus a consolidated far field at a fixed token budget) - generates greedily, parses the
continuation into a tool invocation, and compares it with the action the agent really took:

    tool     the same tool is invoked (editor against editor, shell against shell)
    target   the same file is addressed
    exact    tool, verb and target all agree

Mid-session turns need a long session to exist at all: with `min_history_tokens = 32,768` on a
32K-64K corpus almost every qualifying turn is the last one, which is why the earlier end task
only ever scored submissions.  The 64K-128K corpus is where the middle of a session has enough
history to be an example, so this harness runs there by default.

This is still not task success - the action is not executed against a repository - but it is the
executable form of the decision, which is what a runtime consumes.

Usage
-----
    python benchmarks/g2d_action_turns.py \\
        --jsonl data/sessions-patch-64k.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \\
        --min-history-tokens 32768 --actions-per-session 3 --max-sessions 24 \\
        --out artifacts/g2d-action-turns-state4096-v1.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks.g2_model_quality import build_examples
from benchmarks.g2_trace_index import messages_from_row
from benchmarks.g2b_patch_localization import generate
from benchmarks.g2c_action_reproduction import parse_action, score_action

# The comparison arm the gate names: plain lexical retrieval at the same budget, with no window
# and no consolidation.  It is cheap to add (a 4K context against the full arm's 33-49K) and it is
# the arm the "no regression, ideally better than raw retrieval" criterion is about.
RETRIEVAL_COMPILER = {"recency_spans": 0, "recency_fraction": 0.0, "max_span_fraction": 0.25,
                      "provenance_terms": 8, "dedup": False, "snippet": False,
                      "consolidate": False, "retrieve_multiplier": 1.0,
                      "collapse_paths": True, "compile_mode": "consolidate",
                      "tail_fraction": 0.6, "tail_tokens": 0, "summary_tokens": 512,
                      "summary_input_tokens": 8192, "summary_stride": 8, "tail_cap": 0.5,
                      "far_compiler": "consolidate"}


def action_turns(examples: list, per_session: int, max_history_tokens: int = 0) -> list:
    """The turns worth scoring: turns that take a file-addressing action.

    Spread evenly over the session rather than taken from the front, because the front of a
    session is the exploration the bounded view is supposed to be able to drop, and scoring only
    there would flatter it.

    A turn qualifies by *what it does*, not by where it sits: it must parse to a tool invocation
    that addresses a file.  That is also what excludes the session-ending submission, which
    addresses none - and it is why there is no "drop the last turn" rule.  There was one, and it
    was wrong: on a session that ends with a submission the last *action* turn is a legitimate
    edit, so the rule discarded a mid-session turn and, on a short session, the only one (measured
    while writing the tests: four turns with one edit and one submission yielded one example
    instead of the edit plus the second turn).
    """
    usable = []
    for index, example in enumerate(examples):
        # the history cap is applied *before* the spread, not after: filtering afterwards would
        # drop the later picks of every long session and quietly turn "evenly spread over the
        # session" into "the earliest turns that qualify", which is the opposite of mid-session
        if max_history_tokens and example.history_tokens_estimate > max_history_tokens:
            continue
        action = parse_action(example.target)
        if action is None or action["tool"] == "submit" or not action.get("target"):
            continue
        usable.append((index, example, action))
    if per_session and len(usable) > per_session:
        step = (len(usable) - 1) / (per_session - 1) if per_session > 1 else 0
        picked = [usable[round(i * step)] for i in range(per_session)] if per_session > 1 \
            else [usable[len(usable) // 2]]
        # keep them distinct: rounding can collide when there are barely more turns than slots
        seen, unique = set(), []
        for item in picked:
            if item[0] not in seen:
                seen.add(item[0])
                unique.append(item)
        usable = unique
    return usable


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def free_gpu_mib() -> int:
    """Free device memory, or a large number when the query fails (do not block on a bad query)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 1 << 20


def wait_for_gpu(need_mib: int, max_wait_s: int = 3600, label: str = "") -> None:
    """Block until the card has room.

    This measurement runs on a shared 24 GiB card and a co-tenant arrives without warning; two runs
    of it died mid-generation with `CUDA out of memory` after an hour of work.  Waiting is the
    difference between a measurement and a crashed one, and the checkpoint below is the difference
    between waiting and starting over.
    """
    waited = 0
    while True:
        free = free_gpu_mib()
        if free >= need_mib:
            return
        if waited >= max_wait_s:
            raise SystemExit(f"gave up waiting for {need_mib} MiB free (have {free}){label}")
        print(f"[g2d] waiting for {need_mib} MiB free, have {free}{label}", flush=True)
        time.sleep(20)
        waited += 20


def generate_guarded(model, tokenizer, context, max_new, max_length, device, need_mib, label=""):
    """`generate`, retried through a co-tenant's allocations rather than dying on them."""
    wait_for_gpu(need_mib, label=label)
    for attempt in range(40):
        try:
            return generate(model, tokenizer, context, max_new, max_length, device)
        except Exception as exc:                                     # torch.cuda.OutOfMemoryError
            if "out of memory" not in str(exc).lower():
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"[g2d] OOM on attempt {attempt + 1}{label}, waiting for room", flush=True)
            time.sleep(20)
            wait_for_gpu(need_mib, label=label)
    raise SystemExit(f"still out of memory after 40 attempts{label}")


def summarize(rows: list[dict]) -> dict:
    out = {"action_turns": len(rows),
           "sessions": len({r["session_id"] for r in rows}),
           "history_tokens_p50": (int(statistics.median([r["history_tokens"] for r in rows]))
                                  if rows else None)}
    for arm in ("full", "active", "retrieval"):
        present = [r for r in rows if arm in r]
        if not present:
            continue
        # only the keys the rows actually carry: this list was copied from the localisation harness,
        # which scores an F1 over mentioned files that an action comparison does not produce, and
        # the missing key crashed the payload write *after* every generation had completed
        out[arm] = {key: mean([r[arm][key] for r in present])
                    for key in ("parseable", "tool", "target", "exact")}
        out[arm]["n_tool"] = sum(1 for r in present if r[arm]["tool"] is not None)
        out[arm]["n_target"] = sum(1 for r in present if r[arm]["target"] is not None)
    return out


def bootstrap_ci(diffs: list[float], repeats: int = 20000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats))
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


def paired(rows: list[dict], key: str, base: str = "full", cand: str = "active") -> dict:
    diffs = [float(bool(r[cand][key])) - float(bool(r[base][key])) for r in rows
             if r.get(cand, {}).get(key) is not None and r.get(base, {}).get(key) is not None]
    if not diffs:
        return {"metric": key, "n": 0, "mean_diff": None, "ci95": None}
    lo, hi = bootstrap_ci(diffs)
    return {"metric": key, "n": len(diffs), "mean_diff": statistics.mean(diffs),
            "ci95": [round(lo, 4), round(hi, 4)],
            "wins": sum(1 for d in diffs if d > 0), "losses": sum(1 for d in diffs if d < 0),
            "includes_zero": lo <= 0 <= hi}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--token-budget", type=int, default=4096)
    p.add_argument("--min-history-tokens", type=int, default=32768)
    p.add_argument("--max-length", type=int, default=65536)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--max-sessions", type=int, default=24)
    p.add_argument("--actions-per-session", type=int, default=3)
    p.add_argument("--max-history-tokens", type=int, default=0,
                   help="skip turns whose history is longer than this (0 = no cap); the full arm "
                        "has to fit the model window, and `max_length` truncates it silently")
    p.add_argument("--recency-spans", type=int, default=3)
    p.add_argument("--recency-fraction", type=float, default=0.6)
    p.add_argument("--max-span-fraction", type=float, default=0.25)
    p.add_argument("--provenance-terms", type=int, default=8)
    p.add_argument("--retrieve-multiplier", type=float, default=2.0)
    p.add_argument("--tail-fraction", type=float, default=0.6)
    p.add_argument("--tail-tokens", type=int, default=3584)
    p.add_argument("--tail-cap", type=float, default=0.5)
    p.add_argument("--compile-mode", default="tail_state",
                   choices=("consolidate", "materialize", "state_first", "recency",
                            "tail_state", "tail_query", "compact", "action_window"))
    p.add_argument("--far-compiler", default="consolidate",
                   choices=("consolidate", "materialize", "raw"))
    p.add_argument("--consolidate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dedup-spans", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--snippet-spans", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-collapse-paths", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-free-mib", type=int, default=9000,
                   help="free device memory required before each generation; the card is shared, so "
                        "the harness waits for room instead of dying on a co-tenant's allocation")
    p.add_argument("--from-checkpoint", action="store_true",
                   help="assemble the receipt from an existing checkpoint without generating "
                        "anything: the generations are already recorded, and rebuilding every view "
                        "to re-derive them costs an hour of CPU for no new measurement")
    p.add_argument("--checkpoint", default="",
                   help="append each scored row here as it completes (default: <out>.partial.jsonl) "
                        "and resume from it, so an hour of generation is not lost to one OOM")
    args = p.parse_args(argv)

    checkpoint = Path(args.checkpoint or (args.out + ".partial.jsonl"))

    if args.from_checkpoint:
        if not checkpoint.exists():
            raise SystemExit(f"--from-checkpoint: {checkpoint} does not exist")
        rows = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: (str(r.get("session_id")), r.get("turn_index") or 0))
        return _write_payload(args, rows, checkpoint)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    wait_for_gpu(args.min_free_mib, label=" (model load)")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()

    done: set[tuple] = set()
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                prior = json.loads(line)
                done.add((prior.get("session_id"), prior.get("turn_index")))
        print(f"[g2d] resuming: {len(done)} rows already in {checkpoint}", flush=True)

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
            examples = build_examples(
                messages, session_id=row.get("session_id"), token_budget=args.token_budget,
                min_history_tokens=args.min_history_tokens,
                compiler={"recency_spans": args.recency_spans,
                          "recency_fraction": args.recency_fraction,
                          "max_span_fraction": args.max_span_fraction,
                          "provenance_terms": args.provenance_terms,
                          "dedup": args.dedup_spans, "snippet": args.snippet_spans,
                          "consolidate": args.consolidate,
                          "retrieve_multiplier": args.retrieve_multiplier,
                          "collapse_paths": not args.no_collapse_paths,
                          "compile_mode": args.compile_mode,
                          "tail_fraction": args.tail_fraction,
                          "tail_tokens": args.tail_tokens,
                          "summary_tokens": 512, "summary_input_tokens": 8192,
                          "summary_stride": 8, "tail_cap": args.tail_cap,
                          "far_compiler": args.far_compiler},
                token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            )
            if not examples:
                continue
            # The comparison context is built from the same history.  Which examples exist is
            # decided by `min_history_tokens` on the history alone, so the two lists line up
            # turn for turn; the assertion below is what makes that a checked fact rather than
            # an assumption, because a silent misalignment would pair an arm with another turn's
            # action and every number would be about a different question.
            retrieval_examples = build_examples(
                messages, session_id=row.get("session_id"), token_budget=args.token_budget,
                min_history_tokens=args.min_history_tokens,
                compiler=dict(RETRIEVAL_COMPILER),
                token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            )
            if [e.target for e in retrieval_examples] != [e.target for e in examples]:
                raise SystemExit(f"{meta.get('instance_id')}: the two contexts disagree about "
                                 "which turns are examples; refusing to pair them")
            picked = action_turns(examples, args.actions_per_session, args.max_history_tokens)
            if not picked:
                continue
            for index, example, recorded in picked:
                if (getattr(example, "session_id", None), index) in done:
                    continue
                tag = f" ({meta.get('instance_id')} turn {index})"
                full_text = generate_guarded(model, tokenizer, example.full_context, args.max_new,
                                             args.max_length, args.device, args.min_free_mib, tag)
                active_text = generate_guarded(model, tokenizer, example.active_context, args.max_new,
                                               args.max_length, args.device, args.min_free_mib, tag)
                retrieval_text = generate_guarded(model, tokenizer,
                                                  retrieval_examples[index].active_context,
                                                  args.max_new, args.max_length, args.device,
                                                  args.min_free_mib, tag)
                rows.append({
                    "session_id": getattr(example, "session_id", None),
                    "instance_id": meta.get("instance_id"),
                    "turn_index": index,
                    "turns_total": len(examples),
                    "history_tokens": example.history_tokens_estimate,
                    "active_tokens": example.active_tokens_estimate,
                    "retrieval_tokens": retrieval_examples[index].active_tokens_estimate,
                    "recorded_action": recorded,
                    "recorded_text": example.target[:400],
                    "full": {**score_action(full_text, recorded), "continuation": full_text},
                    "active": {**score_action(active_text, recorded), "continuation": active_text},
                    "retrieval": {**score_action(retrieval_text, recorded),
                                  "continuation": retrieval_text},
                })
                with checkpoint.open("a") as handle:
                    handle.write(json.dumps(rows[-1]) + "\n")
                print(f"[g2d] {meta.get('instance_id')} turn {index}/{len(examples)} "
                      f"hist={example.history_tokens_estimate} "
                      f"action={recorded['tool']}:{recorded.get('target')} "
                      f"full={rows[-1]['full']['tool']}/{rows[-1]['full']['target']} "
                      f"state={rows[-1]['active']['tool']}/{rows[-1]['active']['target']} "
                      f"raw={rows[-1]['retrieval']['tool']}/{rows[-1]['retrieval']['target']}",
                      flush=True)
            if len({r["session_id"] for r in rows}) >= args.max_sessions:
                break

    if done:
        # the checkpoint is the record of what was actually generated; a resumed run has to report
        # the whole sample, not only the rows it produced this time
        seen = {(r.get("session_id"), r.get("turn_index")) for r in rows}
        for line in checkpoint.read_text().splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            if (prior.get("session_id"), prior.get("turn_index")) not in seen:
                rows.append(prior)
    rows.sort(key=lambda r: (str(r.get("session_id")), r.get("turn_index") or 0))
    return _write_payload(args, rows, checkpoint)


def _write_payload(args, rows, checkpoint) -> int:
    summary = summarize(rows)
    payload = {
        "schema": "ephemeral-kv-g2d-action-turns-v1",
        "kind": "executable_action_measurement",
        "config": vars(args),
        "summary": summary,
        "paired": {**{f"active_vs_full_{k}": paired(rows, k)
                      for k in ("tool", "target", "exact")},
                   **{f"active_vs_retrieval_{k}": paired(rows, k, "retrieval", "active")
                      for k in ("tool", "target", "exact")}},
        "rows": rows,
        "interpretation": (
            "mid-session assistant turns whose recorded action addresses a file; three arms are "
            "generated greedily from the same turn - the full transcript, the shipped bounded "
            "state (3,584-token window + consolidated far field at 4,096), and plain lexical "
            "retrieval at the same 4,096 budget - and each continuation is parsed into a tool "
            "invocation.  `target` is scored only where the recorded action addresses a file, "
            "which is every row here by construction.  Not task success: the action is not "
            "executed against a repository"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(json.dumps(payload["paired"], indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
