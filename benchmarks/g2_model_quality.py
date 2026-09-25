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
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Iterable

from ephemeralkv.consolidate import compile_units, consolidate, render
from ephemeralkv.index import terms
from ephemeralkv.statecompile import classify, compile_executable_state, state_first_units
from ephemeralkv.index import DurableSpanIndex, Span
from benchmarks.g2_trace_index import _content, messages_from_row


@dataclass(frozen=True)
class Example:
    full_context: str
    active_context: str
    target: str
    history_tokens_estimate: int
    active_tokens_estimate: int
    # recorded so receipts can be aggregated by session length: the killer table is a statement
    # about what happens as a session grows, not only about how many tokens it holds
    turns: int = 0


def render_span(span: Span) -> str:
    return f"<{span.role}>\n{span.text}\n"


def suffix_within(text: str, budget: int, token_counter) -> str:
    """Largest suffix of `text` that fits `budget` tokens, marked when it is cut.

    Only used when a single span is larger than the whole view budget - a whole-file dump can
    be - where the alternative is either dropping the turn the model is answering or letting
    the view exceed the budget it claims.  The cut is from the front: the end of a tool result
    is what the next turn continues from.
    """
    if token_counter(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_counter(text[-mid:]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    cut = text[-lo:] if lo else ""
    return "[... earlier lines elided ...]\n" + cut


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
    summarizer=None,
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
                options = dict(compiler or {})
                consolidate_view = bool(options.pop("consolidate", False))
                retrieve_multiplier = float(options.pop("retrieve_multiplier", 2.0))
                collapse_paths = bool(options.pop("collapse_paths", True))
                compile_mode = str(options.pop("compile_mode", "consolidate"))
                keep_earlier_verbatim = bool(options.pop("keep_earlier_verbatim", False))
                tail_fraction = float(options.pop("tail_fraction", 0.6))
                # an absolute window size is the right parameterisation: what the surface needs is
                # a *fixed* amount of verbatim recent context (~2.9K measured), so every token
                # beyond it should go to ranked evidence.  A fraction grows the window with the
                # budget, which is why the windowed arms scored the same 0.16 decision at 8,192
                # and at 12,288 - the split stayed proportional.
                tail_tokens = int(options.pop("tail_tokens", 0))
                tail_cap = float(options.pop("tail_cap", 0.5))
                far_compiler = str(options.pop("far_compiler", "consolidate"))
                if compile_mode == "recency":
                    # feasibility control: the last N tokens of history, with no selection and
                    # no compilation.  If 8K of plain recency is far from full history while
                    # 16K is close, the gate is about token volume rather than about which
                    # spans get chosen - and then no extractive compiler can pass it by
                    # re-selecting the same text
                    budget = token_budget
                    taken = []
                    for span in sorted(idx.spans, key=lambda sp: -sp.turn):
                        cost = max(1, int(token_counter(span.text)))
                        if budget - cost < 0:
                            # skip rather than stop: the newest span is often a whole-file
                            # dump that cannot fit, and stopping there leaves an empty view
                            continue
                        budget -= cost
                        taken.append(span)
                    taken.sort(key=lambda sp: sp.turn)
                    active = "".join(f"<{sp.role}>\n{sp.text}\n" for sp in taken)
                    view = taken
                elif consolidate_view:
                    # retrieve generously, consolidate the evidence, then compile down: the
                    # point is that the *representation* changes, not that the ranking does
                    spans, _ = idx.compile_view(
                        query, token_budget=max(token_budget + 1,
                                                int(token_budget * retrieve_multiplier)),
                        max_spans=max_spans, **options)
                    if compile_mode == "compact":
                        # the de facto baseline for context overflow: keep the newest evidence
                        # verbatim, summarise everything older with the model itself, and pack both
                        # into the same budget.  `summarizer` is a text->text callable; without one
                        # this mode degrades to recency rather than pretending to summarise.
                        tail_budget = (tail_tokens if tail_tokens > 0
                                       else int(token_budget * tail_fraction))
                        tail, tail_used, tail_ids = [], 0, set()
                        for span in sorted(idx.spans, key=lambda sp: -sp.turn):
                            cost = max(1, int(token_counter(span.text)))
                            limit = max(tail_budget, cost)
                            if tail_used + cost > limit:
                                continue
                            tail_used += cost
                            tail.append(span)
                            tail_ids.add(span.span_id)
                        tail.sort(key=lambda sp: sp.turn)
                        older = [sp for sp in idx.spans if sp.span_id not in tail_ids]
                        summary_budget = max(64, token_budget - tail_used)
                        older_text = "".join(render_span(sp) for sp in older)
                        if summarizer is not None and older_text.strip():
                            summary = (summarizer(older_text, summary_budget) or "").strip()
                        else:
                            summary = ""
                        active = (f"<summary of earlier history>\n{summary}\n" if summary else "") + \
                            "".join(render_span(sp) for sp in tail)
                        view = tail
                    elif compile_mode == "tail_state":
                        # The tail is the working set.  Keep the most recent spans whole and in
                        # order, then spend what is left on the compiled far field.  This is the
                        # arm the controls point at: surface fidelity is conditioned on a
                        # verbatim tail (8,192 tokens of plain recency sits within 0.3 pp of
                        # full history, while every arm that cut or re-ranked the tail lost
                        # 5-15 pp), and the decision is conditioned on the compressed far field
                        # (a compiled 8K view beats raw evidence on patch localisation).  It
                        # tries to hold both at once, which is the only way both gate halves
                        # pass together.
                        tail_budget = (tail_tokens if tail_tokens > 0
                                       else int(token_budget * tail_fraction))
                        tail, tail_used, tail_ids = [], 0, set()
                        newest = max(idx.spans, key=lambda sp: sp.turn) if idx.spans else None
                        for span in sorted(idx.spans, key=lambda sp: -sp.turn):
                            cost = max(1, int(token_counter(span.text)))
                            limit = tail_budget
                            if span is newest:
                                # the current turn is not optional.  Admit the newest span whole
                                # even when it exceeds the tail share - the share governs how much
                                # *older* context is protected, not whether the turn the model is
                                # answering is visible - and if it cannot fit the whole budget,
                                # keep its suffix rather than dropping the only evidence the new
                                # turn has.
                                limit = max(tail_budget, cost)
                                if cost > token_budget:
                                    # never rebind `text`: it is the loop's message text and it
                                    # becomes both the example target and the next indexed span,
                                    # so shadowing it here corrupts the target and the durable
                                    # index for every later turn (measured: histories grew by
                                    # 8,087 tokens per corrupted span)
                                    cut = suffix_within(span.text, token_budget, token_counter)
                                    tail.append(replace(span, text=cut,
                                                        token_estimate=token_budget))
                                    tail_ids.add(span.span_id)
                                    tail_used = token_budget
                                    break
                            if tail_used + cost > limit:
                                continue
                            tail_used += cost
                            tail.append(span)
                            tail_ids.add(span.span_id)
                        tail.sort(key=lambda sp: sp.turn)
                        far_budget = max(0, token_budget - tail_used)
                        far_options = dict(options)
                        # only the *window* is handled separately: the far field must get the
                        # same treatment the compiled arm gives its evidence (truncate or
                        # snippet an oversized span rather than dropping it), otherwise this arm
                        # is not "window + compiler" but "window + whatever survived", and the
                        # comparison against the compiled arm was measuring the difference in
                        # span treatment, not the difference in what the budget is spent on
                        far_options["recency_spans"] = 0
                        far_spans, _ = idx.compile_view(
                            query, token_budget=far_budget, max_spans=max_spans,
                            **far_options)
                        far_spans = [sp for sp in far_spans if sp.span_id not in tail_ids]
                        if far_compiler == "raw":
                            # no compiler at all: the window carries the surface, plain retrieval
                            # carries the decision.  This is the arm a system would ship if the
                            # compiler's stages cannot be shown to pay (and on this corpus they
                            # cannot: plain retrieval at 4,096 beats the compiled 8,192 view by
                            # 0.104 F1 paired on 48 sessions).
                            active = "".join(render_span(sp) for sp in far_spans) + \
                                "".join(render_span(sp) for sp in tail)
                            view = far_spans + tail
                        elif far_compiler == "materialize":
                            # the far field is the *executable state* (replay the log: dumps set a
                            # file's content, diffs and search/replace blocks apply to it, repeated
                            # commands collapse to their latest result), not a shortened transcript.
                            # The recent window stays verbatim because the surface needs it; the
                            # far field is state because the decision needs it.
                            far_units, far_stats = compile_executable_state(
                                far_spans, far_budget, token_counter,
                                query_terms=set(terms(query)))
                        else:
                            far_units, far_stats = compile_units(
                                consolidate(far_spans, collapse_paths=collapse_paths,
                                            keep_earlier_verbatim=keep_earlier_verbatim),
                                far_budget, token_counter)
                        active = render(far_units) + "".join(render_span(sp) for sp in tail)
                        view = far_spans + tail
                    elif compile_mode == "tail_query":
                        # The current turn verbatim, the query's evidence for everything else.
                        #
                        # The two metrics want different things and both are conditional on the
                        # newest span: fidelity is at parity when the newest span is kept whole
                        # (8,192 tokens of recency, +0.29 pp) and loses 5-15 pp when it is
                        # truncated or rewritten, while the decision is carried by query-focused
                        # evidence (a compiled 8K view scores 0.292 against 0.237 raw and 0.154
                        # for recency).  tail_state bought fidelity with a 60% tail and lost the
                        # decision (0.104), because *generic* recency padding spends budget on
                        # older spans the decision does not use.  This arm spends nothing on
                        # generic recency: the newest span, then the compiler.
                        tail, tail_used, tail_ids = [], 0, set()
                        newest = max(idx.spans, key=lambda sp: sp.turn) if idx.spans else None
                        if newest is not None:
                            cost = max(1, int(token_counter(newest.text)))
                            share = max(1, int(token_budget * tail_cap))
                            if cost > share:
                                # a newest span larger than its share is kept from the end: the
                                # alternative is dropping the turn the model is answering or
                                # leaving no budget for the evidence the decision needs
                                cut = suffix_within(newest.text, share, token_counter)
                                tail.append(replace(newest, text=cut,
                                                    token_estimate=share))
                                tail_used = share
                            else:
                                tail.append(newest)
                                tail_used = cost
                            tail_ids.add(newest.span_id)
                        far_budget = max(0, token_budget - tail_used)
                        far_options = dict(options)
                        far_options["recency_spans"] = 0
                        far_options["max_span_fraction"] = 1.0
                        far_spans, _ = idx.compile_view(
                            query, token_budget=far_budget, max_spans=max_spans,
                            **far_options)
                        far_spans = [sp for sp in far_spans if sp.span_id not in tail_ids]
                        far_units, _ = compile_units(
                            consolidate(far_spans, collapse_paths=collapse_paths,
                                        keep_earlier_verbatim=keep_earlier_verbatim),
                            far_budget, token_counter)
                        active = render(far_units) + "".join(render_span(sp) for sp in tail)
                        view = far_spans + tail
                    elif compile_mode == "state_first":
                        # selection, not compression: the view is built from state-carrying
                        # evidence (all materialised file states, the latest result of each
                        # command) and only then from loose text, instead of taking whatever
                        # the lexical ranking happened to return
                        units, _stats = state_first_units(idx, query, token_budget,
                                                          token_counter)
                    elif compile_mode == "materialize":
                        # the executable-state compiler: replay the log's file events and keep
                        # the materialised current state, not the latest view of it
                        units, _stats = compile_executable_state(
                            spans, token_budget, token_counter,
                            query_terms=set(terms(query)))
                    else:
                        units, _stats = compile_units(
                            consolidate(spans, collapse_paths=collapse_paths,
                                        keep_earlier_verbatim=keep_earlier_verbatim),
                            token_budget, token_counter)
                        active = render(units)
                        view = spans
                else:
                    view, _ = idx.compile_view(
                        query, token_budget=token_budget, max_spans=max_spans, **options)
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
                        active_tokens_estimate=(
                            max(1, int(token_counter(active))) if consolidate_view
                            else sum(max(1, s.token_estimate) for s in view)
                        ),
                        turns=len(history),
                    )
                )

        # Index every completed transcript event after constructing the example.
        tok_est = int(token_counter(text))
        idx.append(turn=turn, role=role, text=text, token_estimate=max(1, tok_est),
                   kind=classify(text))
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
    p.add_argument("--snippet-spans", action=argparse.BooleanOptionalAction, default=False,
                   help="keep query-relevant lines of an oversized span instead of its "
                        "prefix (tool outputs are mostly whole-file dumps)")
    p.add_argument("--consolidate", action=argparse.BooleanOptionalAction, default=False,
                   help="compile the retrieved evidence instead of concatenating it: "
                        "collapse superseded file states and repeated output, then keep "
                        "executable lines verbatim (the G2 gap is in the representation, "
                        "not in the ranking)")
    p.add_argument("--retrieve-multiplier", type=float, default=2.0,
                   help="how much evidence to retrieve before consolidating it down")
    p.add_argument("--tail-fraction", type=float, default=0.6,
                   help="share of the budget kept as an untruncated verbatim tail "
                        "(tail_state mode)")
    p.add_argument("--far-compiler", default="consolidate",
                   choices=("consolidate", "materialize", "raw"),
                   help="how the far field of tail_state is compiled: `raw` keeps the retrieved "
                        "spans as they are, `consolidate` keeps the "
                        "newest view of each piece of state, `materialize` replays the log into "
                        "the current executable state")
    p.add_argument("--tail-tokens", type=int, default=0,
                   help="absolute size of the verbatim window in tail_state mode; 0 uses "
                        "--tail-fraction x budget.  The surface needs a fixed amount of verbatim "
                        "context, so an absolute window is the right knob: a fraction grows the "
                        "window with the budget and leaves the ranked share unchanged")
    p.add_argument("--tail-cap", type=float, default=0.5,
                   help="share of the budget the newest span may take before it is kept from "
                        "the end instead (tail_query mode)")
    p.add_argument("--compile-mode", default="consolidate",
                   choices=("consolidate", "materialize", "state_first", "recency",
                            "tail_state", "tail_query", "compact"),
                   help="`consolidate` picks among the retrieved views; `materialize` replays "
                        "the log's file events and keeps the current state")
    p.add_argument("--no-collapse-paths", action="store_true",
                   help="ablation: keep every view of a file instead of its latest state")
    p.add_argument("--keep-earlier-verbatim", action="store_true",
                   help="union of evidence: latest state plus the verbatim lines of the "
                        "views it replaced")
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
            def _summarise(text: str, budget: int, _model=model, _tok=tokenizer) -> str:
                """The model summarises its own older history, greedily and within the budget."""
                import torch

                prompt = ("Summarise the earlier part of this coding session for the next turn. "
                          "Keep file paths, decisions, errors and open problems; drop chatter.\n\n"
                          + text)
                keep = max(256, args.max_length - 512)
                ids = _tok.encode(prompt, add_special_tokens=False)[-keep:]
                input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
                with torch.inference_mode():
                    out = _model.generate(input_ids=input_ids, max_new_tokens=int(budget),
                                          do_sample=False,
                                          pad_token_id=_tok.eos_token_id)
                return _tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

            examples = build_examples(
                messages,
                token_budget=args.token_budget,
                summarizer=_summarise if args.compile_mode == "compact" else None,
                max_spans=args.max_spans,
                min_history_tokens=args.min_history_tokens,
                compiler={"recency_spans": args.recency_spans,
                          "recency_fraction": args.recency_fraction,
                          "max_span_fraction": args.max_span_fraction,
                          "provenance_terms": args.provenance_terms,
                          "dedup": args.dedup_spans,
                          "snippet": args.snippet_spans,
                          "consolidate": args.consolidate,
                          "retrieve_multiplier": args.retrieve_multiplier,
                          "collapse_paths": not args.no_collapse_paths,
                          "compile_mode": args.compile_mode,
                          "tail_fraction": args.tail_fraction,
                          "tail_tokens": args.tail_tokens,
                          "tail_cap": args.tail_cap,
                          "far_compiler": args.far_compiler,
                          "keep_earlier_verbatim": args.keep_earlier_verbatim},
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
                        "turns": ex.turns,
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
                     "dedup": args.dedup_spans,
                     "snippet": args.snippet_spans},
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
