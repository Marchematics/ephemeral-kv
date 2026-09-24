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
    min_history_tokens: int = 0,
    compiler: dict | None = None,
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
                    query, token_budget=token_budget, max_spans=max_spans,
                    **(compiler or {}),
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

    if min_history_tokens:
        # the gate is about *long* sessions.  A session that reaches 64K tokens does so at
        # its end, so without this filter almost every example comes from an early turn
        # whose history is a few thousand tokens (measured: sessions with max_isl >= 64K
        # still produced examples with 4K-16K histories, leaving the >=32K bucket empty).
        out = [ex for ex in out if ex.history_tokens_estimate >= min_history_tokens]
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


def _encode_with_target(tokenizer, context: str, target: str, max_length: int,
                        max_target_tokens: int = 512):
    """Return input ids and a loss mask while preserving the target suffix.

    Both the target and the context are bounded.  A coding trajectory's assistant turn
    can itself be tens of thousands of tokens (a whole file write), and scoring all of it
    needs logits of `target x vocabulary` - 30K x 128K x 4B is 15 GiB, which OOMed this
    gate on a 24 GiB card even after the head was sliced.  The scored prefix of the
    target is the part that tests whether retrieval preserved what the turn needed, and
    both arms get the same truncation.
    """
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids or max_length < 2:
        return None, None
    limit = max(1, min(int(max_target_tokens), max_length - 1))
    target_ids = target_ids[:limit]
    prefix_ids = tokenizer.encode(context, add_special_tokens=False)
    room = max_length - len(target_ids)
    prefix_ids = prefix_ids[-room:]
    ids = prefix_ids + target_ids
    target_start = len(prefix_ids)
    return ids, target_start


def score_target(model, tokenizer, context: str, target: str, max_length: int, device: str,
                 max_target_tokens: int = 512):
    """Teacher-forced loss and next-token accuracy on the target suffix.

    The head is applied only to the target positions.  Asking the causal LM for `labels`
    materialises logits for *every* context position - at 32K and a 128K vocabulary that
    is ~17 GiB, which OOMed this gate on a 24 GiB card (measured: a single 8.21 GiB
    cross-entropy allocation) - while the gate only needs the target's tokens.  Running
    the backbone and slicing its last hidden state gives the same numbers for a few
    hundred target positions (checked against the `labels=` path in
    tests/test_g2_model_quality.py).
    """
    import torch
    import torch.nn.functional as F

    ids, target_start = _encode_with_target(tokenizer, context, target, max_length,
                                            max_target_tokens)
    if ids is None or target_start <= 0 or target_start >= len(ids):
        return None

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    backbone = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    with torch.inference_mode():
        if backbone is None or head is None:                 # pragma: no cover
            labels = input_ids.clone()
            labels[:, :target_start] = -100
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            logits = out.logits[:, target_start - 1:-1]
            loss = float(out.loss.item())
        else:
            hidden = backbone(input_ids=input_ids, use_cache=False).last_hidden_state
            logits = head(hidden[:, target_start - 1:-1].to(hidden.dtype)).float()
            labels = input_ids[:, target_start:]
            loss = float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).item())
        labels = input_ids[:, target_start:]
        count = int(labels.numel())
        correct = int((logits.argmax(dim=-1) == labels).sum().item())

    return {
        "nll": loss,
        "target_tokens": count,
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
    p.add_argument("--recency-spans", type=int, default=0,
                   help="always keep the last N spans (a serving stack never evicts the "
                        "current turn); 0 reproduces the lexical-only receipt")
    p.add_argument("--recency-fraction", type=float, default=0.5)
    p.add_argument("--max-span-fraction", type=float, default=1.0,
                   help="truncate an oversized span to this share of the budget instead "
                        "of dropping it")
    p.add_argument("--dedup-spans", action=argparse.BooleanOptionalAction, default=False,
                   help="collapse identical span texts in the compiled view, keeping the "
                        "most recent copy (measured: 17.5%% of view spans are duplicates)")
    p.add_argument("--provenance-terms", type=int, default=0,
                   help="also pull spans sharing identifiers with the top lexical hits")
    p.add_argument("--min-history-tokens", type=int, default=0,
                   help="only score turns whose preceding history reaches this size; "
                        "without it examples come from the early turns of long sessions")
    p.add_argument("--max-target-tokens", type=int, default=512,
                   help="scored prefix of the assistant target; the logits for a whole "
                        "long turn do not fit (see _encode_with_target)")
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
                min_history_tokens=args.min_history_tokens,
                compiler={"recency_spans": args.recency_spans,
                          "recency_fraction": args.recency_fraction,
                          "max_span_fraction": args.max_span_fraction,
                          "provenance_terms": args.provenance_terms,
                          "dedup": args.dedup_spans},
                token_counter=lambda text: len(
                    tokenizer.encode(text, add_special_tokens=False)
                ),
            )
            for ex in examples:
                full = score_target(
                    model, tokenizer, ex.full_context, ex.target,
                    args.max_length, args.device, args.max_target_tokens
                )
                active = score_target(
                    model, tokenizer, ex.active_context, ex.target,
                    args.max_length, args.device, args.max_target_tokens
                )
                rows.append(
                    {
                        "history_tokens_estimate": ex.history_tokens_estimate,
                        "active_tokens_estimate": ex.active_tokens_estimate,
                        "full": full,
                        "active": active,
                    }
                )
                if len(rows) % 16 == 0:
                    # progress plus an incremental artifact: this runs on a shared GPU,
                    # and a long trace pass should not lose everything to one OOM
                    print(f"[g2] {len(rows)} examples, {sessions} sessions, "
                          f"history {ex.history_tokens_estimate} tokens, "
                          f"active fraction "
                          f"{ex.active_tokens_estimate / max(1, ex.history_tokens_estimate):.3f}",
                          flush=True)
                    Path(args.out).write_text(json.dumps(
                        {"schema": "ephemeral-kv-g2-model-quality-v1", "partial": True,
                         "model": args.model, "token_budget": args.token_budget,
                         "max_length": args.max_length, "sessions": sessions,
                         "rows": rows}, indent=2) + "\n")
                if args.max_examples and len(rows) >= args.max_examples:
                    break
            sessions += 1
            if (args.max_sessions and sessions >= args.max_sessions) or (
                args.max_examples and len(rows) >= args.max_examples
            ):
                break
            if args.max_examples and not examples and sessions > 4000:
                # a filter can leave whole sessions with nothing to score
                print(f"[g2] stopping after {sessions} sessions with no scoreable turn",
                      flush=True)
                break

    payload = {
        "schema": "ephemeral-kv-g2-model-quality-v1",
        "kind": "model_quality_measurement",
        "model": args.model,
        "token_budget": args.token_budget,
        "max_length": args.max_length,
        "compiler": {"recency_spans": args.recency_spans,
                     "recency_fraction": args.recency_fraction,
                     "max_span_fraction": args.max_span_fraction,
                     "provenance_terms": args.provenance_terms,
                     "dedup": args.dedup_spans},
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
