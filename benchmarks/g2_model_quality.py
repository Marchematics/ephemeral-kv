#!/usr/bin/env python
"""G2 model-quality gate: does a non-QCC active view preserve the next turn?

This runner uses a frozen causal LM and public multi-turn trajectories. For every
assistant turn with enough preceding history it compares teacher-forced target loss
under:

* full-history context (truncated only by an explicitly declared model window);
* EphemeralKV lexical/provenance active view compiled from prior spans.

This is a model-level quality gate, not an end-task success result. It exists to kill
the project early if bounded retrieval destroys the information needed for the next
assistant action before we build a cluster runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from ephemeralkv.index import DurableSpanIndex, Span
from benchmarks.g2_trace_index import _content, messages_from_row


@dataclass(frozen=True)
class Example:
    full_context: str
    active_context: str
    target: str
    history_tokens_estimate: int
    active_tokens_estimate: int


def render_span(span: Span) -> str:
    return f"<{span.role}>\n{span.text}\n"


def render_message(msg: dict) -> str:
    return f"<{msg.get('role', 'unknown')}>\n{_content(msg)}\n"


def build_examples(
    messages: list[dict],
    *,
    token_budget: int,
    max_spans: int = 64,
    min_history_spans: int = 8,
    token_counter=None,
) -> list[Example]:
    """Build assistant-target examples without consulting the target during retrieval."""
    idx = DurableSpanIndex()
    history: list[dict] = []
    if token_counter is None:
        token_counter = lambda text: max(1, len(text.split()))
    out: list[Example] = []

    for turn, msg in enumerate(messages):
        role = str(msg.get("role", "unknown"))
        text = _content(msg)

        if role == "assistant" and text.strip() and len(idx.spans) >= min_history_spans:
            # Query retrieval from the most recent user/tool event only. This prevents
            # target leakage and approximates the information visible when scheduling
            # the next assistant action.
            query = ""
            for prev in reversed(history):
                if str(prev.get("role", "")) in {"user", "tool"}:
                    query = _content(prev)
                    break
            if query:
                view, _ = idx.compile_view(
                    query, token_budget=token_budget, max_spans=max_spans
                )
                active = "".join(render_span(s) for s in view)
                full = "".join(render_message(x) for x in history)
                out.append(
                    Example(
                        full_context=full,
                        active_context=active,
                        target=text,
                        history_tokens_estimate=sum(
                            max(1, s.token_estimate) for s in idx.spans
                        ),
                        active_tokens_estimate=sum(
                            max(1, s.token_estimate) for s in view
                        ),
                    )
                )

        # Index every completed transcript event after constructing the example.
        tok_est = int(token_counter(text))
        idx.append(turn=turn, role=role, text=text, token_estimate=max(1, tok_est))
        history.append(msg)

    return out


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


def _encode_with_target(tokenizer, context: str, target: str, max_length: int):
    """Return input ids and a loss mask while preserving the target suffix."""
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids or max_length < 2:
        return None, None
    target_ids = target_ids[: max_length - 1]
    prefix_ids = tokenizer.encode(context, add_special_tokens=False)
    room = max_length - len(target_ids)
    prefix_ids = prefix_ids[-room:]
    ids = prefix_ids + target_ids
    target_start = len(prefix_ids)
    return ids, target_start


def score_target(model, tokenizer, context: str, target: str, max_length: int, device: str):
    import torch

    ids, target_start = _encode_with_target(tokenizer, context, target, max_length)
    if ids is None or target_start <= 0 or target_start >= len(ids):
        return None

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :target_start] = -100

    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        logits = out.logits[:, :-1]
        next_ids = input_ids[:, 1:]
        mask = labels[:, 1:] != -100
        pred = logits.argmax(dim=-1)
        correct = ((pred == next_ids) & mask).sum().item()
        count = mask.sum().item()

    return {
        "nll": float(out.loss.item()),
        "target_tokens": int(count),
        "token_accuracy": float(correct / count) if count else None,
        "context_tokens": int(target_start),
    }


# the same history buckets the structural harness reports, so the two halves of G2 can
# be read side by side: active fraction from the index, quality from the model
HISTORY_BUCKETS = ((0, 8192, "<=8K"), (8192, 32768, "8K-32K"),
                   (32768, 131072, "32K-128K"), (131072, None, ">=128K"))

# the plan's G2 advance rule, in the units this harness measures: at worst a two
# percentage point loss of next-token accuracy, and an active fraction that does not
# grow with history
ACCURACY_TOLERANCE = -0.02
MIN_BUCKET_EXAMPLES = 8


def bucket_of(history_tokens: int) -> str:
    for low, high, label in HISTORY_BUCKETS:
        if history_tokens >= low and (high is None or history_tokens < high):
            return label
    return HISTORY_BUCKETS[-1][2]


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("full") and r.get("active")]
    if not valid:
        return {"examples": 0}

    nll_delta = [r["active"]["nll"] - r["full"]["nll"] for r in valid]
    acc_delta = [
        r["active"]["token_accuracy"] - r["full"]["token_accuracy"]
        for r in valid
        if r["active"]["token_accuracy"] is not None
        and r["full"]["token_accuracy"] is not None
    ]
    fractions = [
        r["active_tokens_estimate"] / max(1, r["history_tokens_estimate"])
        for r in valid
    ]
    return {
        "examples": len(valid),
        "active_fraction_p50": median(fractions),
        "active_fraction_p95": percentile(fractions, 0.95),
        "nll_delta_active_minus_full_p50": median(nll_delta),
        "nll_delta_active_minus_full_p95": percentile(nll_delta, 0.95),
        "token_accuracy_delta_p50": median(acc_delta) if acc_delta else None,
        "token_accuracy_delta_p05": percentile(acc_delta, 0.05) if acc_delta else None,
    }


def summarize_by_bucket(rows: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(bucket_of(int(row.get("history_tokens_estimate") or 0)),
                           []).append(row)
    return {label: summarize(buckets[label]) for _low, _high, label in HISTORY_BUCKETS
            if label in buckets}


def verdict_by_bucket(by_bucket: dict) -> dict:
    """The checkable form of the G2 advance rule.

    `advance` needs every populated bucket to be both large enough to read and inside
    the accuracy tolerance, and the longest bucket's active fraction to be no larger
    than the shortest bucket's - i.e. the working set must not track history.
    """
    usable = {label: stats for label, stats in by_bucket.items()
              if stats.get("examples", 0) >= MIN_BUCKET_EXAMPLES}
    if len(usable) < 2:
        return {"verdict": "inconclusive", "reason": "fewer than two readable buckets",
                "readable_buckets": sorted(usable)}

    def order(label):
        return [i for i, (_l, _h, name) in enumerate(HISTORY_BUCKETS) if name == label][0]

    labels = sorted(usable, key=order)
    losses = {label: usable[label]["token_accuracy_delta_p50"] for label in labels}
    fractions = {label: usable[label]["active_fraction_p50"] for label in labels}
    worst_loss = min((v for v in losses.values() if v is not None), default=None)
    grows = fractions[labels[-1]] > fractions[labels[0]] + 1e-9
    inside = worst_loss is not None and worst_loss >= ACCURACY_TOLERANCE
    return {
        "verdict": "advance" if (inside and not grows) else "kill",
        "reason": ("active fraction and accuracy both hold across buckets" if inside and not grows
                   else ("accuracy loss beyond tolerance" if not inside
                         else "active fraction grows with history")),
        "readable_buckets": labels,
        "active_fraction_p50_by_bucket": fractions,
        "token_accuracy_delta_p50_by_bucket": losses,
        "worst_token_accuracy_delta_p50": worst_loss,
        "accuracy_tolerance": ACCURACY_TOLERANCE,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--token-budget", type=int, default=4096)
    p.add_argument("--max-spans", type=int, default=64)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-sessions", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device)
    model.eval()

    rows = []
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
            examples = build_examples(
                messages,
                token_budget=args.token_budget,
                max_spans=args.max_spans,
                token_counter=lambda text: len(
                    tokenizer.encode(text, add_special_tokens=False)
                ),
            )
            for ex in examples:
                full = score_target(
                    model, tokenizer, ex.full_context, ex.target,
                    args.max_length, args.device
                )
                active = score_target(
                    model, tokenizer, ex.active_context, ex.target,
                    args.max_length, args.device
                )
                rows.append(
                    {
                        "history_tokens_estimate": ex.history_tokens_estimate,
                        "active_tokens_estimate": ex.active_tokens_estimate,
                        "full": full,
                        "active": active,
                    }
                )
                if args.max_examples and len(rows) >= args.max_examples:
                    break
            sessions += 1
            if (args.max_sessions and sessions >= args.max_sessions) or (
                args.max_examples and len(rows) >= args.max_examples
            ):
                break

    payload = {
        "schema": "ephemeral-kv-g2-model-quality-v1",
        "kind": "model_quality_measurement",
        "model": args.model,
        "token_budget": args.token_budget,
        "max_length": args.max_length,
        "sessions": sessions,
        "summary": summarize(rows),
        "by_history_bucket": summarize_by_bucket(rows),
        "verdict_by_bucket": verdict_by_bucket(summarize_by_bucket(rows)),
        "advance_rule": (
            f"every readable history bucket keeps token-accuracy p50 >= "
            f"{ACCURACY_TOLERANCE} and the longest bucket's active fraction does not "
            f"exceed the shortest bucket's; buckets need >= {MIN_BUCKET_EXAMPLES} "
            f"examples to be read"
        ),
        "rows": rows,
        "interpretation": (
            "teacher-forced next-assistant continuation quality; this is a model-level "
            "G2 gate and does not substitute for end-task agent success"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print("by bucket:", json.dumps(payload["by_history_bucket"], indent=2))
    print("verdict:", json.dumps(payload["verdict_by_bucket"], indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
