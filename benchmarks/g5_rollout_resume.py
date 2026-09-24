#!/usr/bin/env python
"""G5: a worker or a model can be replaced without moving history-sized state.

The claim the other gates point at is that session state has two parts with different
lifetimes:

* the **durable** part - the transcript, and a model-independent index over it;
* the **ephemeral** part - the per-worker KV cache, which is an optimisation.

Two things follow, and this runner measures both:

1. **Resume cost after a model rollout.**  KV is model-specific: after a rollout the old
   cache cannot be used by the new model, so the only full-KV option is re-prefilling the
   history on the new model.  The ephemeral route needs the transcript and the index,
   which are model-independent, plus a prefill of the active set on the new model.
2. **Cross-model fidelity of the durable view.**  The same index and the same active view
   are scored with a *different* frozen model than the one the view was built for; if the
   view is a property of the session rather than of the model, the next turn stays intact.

The runner reuses the G2 model-quality machinery (example construction without target
leakage, suffix-scored teacher-forced fidelity) and reads the prefill constants from the
G3 receipts where a measurement is not repeated here.

Usage:
    python benchmarks/g5_rollout_resume.py \
        --jsonl data/sessions-64k.jsonl --models <model-a> <model-b> \
        --token-budget 16384 --min-history-tokens 32768 --max-length 65536 \
        --out artifacts/g5-rollout-v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g2_model_quality import build_examples, score_target  # noqa: E402
from g2_trace_index import messages_from_row  # noqa: E402

# measured in G3 (Qwen2.5-0.5B, 12,288 B/token): full re-prefill at 32K and 128K, and
# "infeasible" past 512K on a 24 GiB card
G3_FULL_PREFILL_S = {32768: 1.59, 131072: 14.32, 524288: None}
G3_LOOKUP_S = {8192: 0.000087, 32768: 0.00015, 131072: 0.000246}


def recompile_costs(history_tokens: int) -> dict:
    """Cost of the only full-KV option after a model change: re-prefill the history.

    Where the G3 measurement itself says infeasible for a length at or below
    `history_tokens`, so does this - an infeasible route is reported, never extrapolated.
    Otherwise the cost is scaled from the largest measured length at or below the history,
    and flagged as optimistic, because the measured curve is superlinear (4x the history
    cost 9x the time).
    """
    infeasible_at = [tokens for tokens, seconds in G3_FULL_PREFILL_S.items()
                     if seconds is None]
    if infeasible_at and history_tokens >= min(infeasible_at):
        return {"status": "infeasible",
                "reason": f"a {min(infeasible_at)}-token prefill is infeasible on this "
                          f"card, so a {history_tokens}-token one is too"}
    feasible = [(tokens, seconds) for tokens, seconds in sorted(G3_FULL_PREFILL_S.items())
                if seconds is not None and tokens <= history_tokens]
    if not feasible:
        return {"status": "infeasible",
                "reason": "no measured prefill at or below this history"}
    tokens, seconds = feasible[-1]
    return {"status": "estimated", "seconds": round(seconds * (history_tokens / tokens), 3),
            "scaled_from_tokens": tokens, "optimistic": "linear scaling of a superlinear "
                                                        "measured curve"}


def lookup_cost(history_tokens: int) -> float:
    bucket = next((b for b in sorted(G3_LOOKUP_S) if history_tokens <= b),
                  max(G3_LOOKUP_S))
    return G3_LOOKUP_S[bucket]


def load_model(name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    return model, tokenizer


def measure_active_prefill(model, tokenizer, tokens: int, device: str) -> float:
    """Seconds to prefill `tokens` tokens of this model: the ephemeral resume cost."""
    import time
    import torch

    vocab = int(model.config.vocab_size)
    ids = (torch.arange(tokens, device=device, dtype=torch.long) % max(2, vocab - 1)) \
        .unsqueeze(0)
    with torch.inference_mode():
        model(input_ids=ids, use_cache=True)                 # warm-up
        torch.cuda.synchronize()
        started = time.perf_counter()
        model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
    return time.perf_counter() - started


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="two or more frozen checkpoints; the first builds the views, the "
                        "rest stand in for a rollout")
    p.add_argument("--token-budget", type=int, default=16384)
    p.add_argument("--min-history-tokens", type=int, default=32768)
    p.add_argument("--max-length", type=int, default=65536)
    p.add_argument("--max-examples", type=int, default=24)
    p.add_argument("--max-spans", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import statistics

    # one example set, built once with the first model's tokenizer: the *view* must not
    # depend on which model will consume it, and the token budget is the caller's
    primary, primary_tokenizer = load_model(args.models[0], args.device)
    examples = []
    histories = []
    with Path(args.jsonl).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                messages = messages_from_row(row)
            except ValueError:
                continue
            built = build_examples(
                messages, token_budget=args.token_budget, max_spans=args.max_spans,
                min_history_tokens=args.min_history_tokens,
                compiler={"recency_spans": 3, "recency_fraction": 0.6,
                          "max_span_fraction": 0.25, "provenance_terms": 8},
                token_counter=lambda text: len(
                    primary_tokenizer.encode(text, add_special_tokens=False)),
            )
            for example in built:
                examples.append(example)
                histories.append(example.history_tokens_estimate)
                if len(examples) >= args.max_examples:
                    break
            if len(examples) >= args.max_examples:
                break
    del primary
    import torch
    torch.cuda.empty_cache()

    per_model = {}
    for name in args.models:
        model, tokenizer = load_model(name, args.device)
        rescored = []
        for example in examples:
            full = score_target(model, tokenizer, example.full_context, example.target,
                                args.max_length, args.device)
            active = score_target(model, tokenizer, example.active_context,
                                  example.target, args.max_length, args.device)
            if full and active:
                rescored.append({"full": full, "active": active,
                                 "history_tokens": example.history_tokens_estimate,
                                 "active_tokens": example.active_tokens_estimate})
        prefill_s = measure_active_prefill(model, tokenizer, args.token_budget, args.device)
        per_model[name] = {
            "examples": len(rescored),
            "nll_delta_p50": statistics.median(
                [r["active"]["nll"] - r["full"]["nll"] for r in rescored]) if rescored else None,
            "token_accuracy_delta_p50": statistics.median(
                [r["active"]["token_accuracy"] - r["full"]["token_accuracy"]
                 for r in rescored]) if rescored else None,
            "active_prefill_s": prefill_s,
            "rows": rescored,
        }
        del model
        torch.cuda.empty_cache()

    median_history = int(statistics.median(histories)) if histories else 0
    resume = {
        "history_tokens_p50": median_history,
        "ephemeral": {
            "lookup_s": lookup_cost(median_history),
            "active_prefill_s": {name: per_model[name]["active_prefill_s"]
                                 for name in per_model},
        },
        "full_kv_after_rollout": recompile_costs(median_history),
        "note": ("KV is model-specific, so a rollout cannot reuse a cache from the previous "
                 "model: the full-KV route is a re-prefill of the history on the new model, "
                 "and the ephemeral route is a prefill of the active set"),
        "caveats": [
            "the active view's token budget is applied with the primary model's tokenizer; "
            "another tokenizer renders the same text in slightly more or fewer tokens",
            "the full-KV re-prefill figure is scaled linearly from a measured length and "
            "flagged optimistic - the measured curve is superlinear",
            "fidelity here is teacher-forced next-turn scoring, not end-task success",
        ],
    }
    payload = {
        "schema": "ephemeral-kv-g5-rollout-v1",
        "kind": "measurement_and_composition",
        "config": vars(args),
        "examples": len(examples),
        "resume": resume,
        "per_model": {name: {k: v for k, v in stats.items() if k != "rows"}
                      for name, stats in per_model.items()},
        "per_model_rows": per_model,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"resume": resume,
                      "per_model": payload["per_model"]}, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
