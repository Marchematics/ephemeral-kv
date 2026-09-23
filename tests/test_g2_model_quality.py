from __future__ import annotations

import pytest

from benchmarks.g2_model_quality import build_examples, summarize


def _trajectory():
    msgs = [{"role": "system", "content": "coding agent"}]
    msgs += [
        {"role": "user", "content": "fix src/cache.py eviction bug"},
        {"role": "assistant", "content": "I will inspect src/cache.py"},
        {"role": "tool", "content": "src/cache.py contains an LRU eviction path"},
        {"role": "assistant", "content": "The eviction path is suspicious"},
    ]
    for i in range(12):
        msgs += [
            {"role": "user", "content": f"inspect unrelated module {i}"},
            {"role": "tool", "content": f"module_{i}.py telemetry nominal"},
            {"role": "assistant", "content": f"module {i} is unrelated"},
        ]
    msgs += [
        {"role": "user", "content": "what should we change in src/cache.py?"},
        {"role": "assistant", "content": "change the LRU eviction path in src/cache.py"},
    ]
    return msgs


def test_examples_do_not_include_target_in_context():
    examples = build_examples(_trajectory(), token_budget=64, min_history_spans=8)
    assert examples
    ex = examples[-1]
    assert ex.target == "change the LRU eviction path in src/cache.py"
    assert ex.target not in ex.full_context
    assert ex.target not in ex.active_context


def test_active_view_is_bounded():
    examples = build_examples(_trajectory(), token_budget=16, min_history_spans=8)
    assert examples
    assert max(x.active_tokens_estimate for x in examples) <= 16


def test_summary_reports_active_vs_full_deltas():
    rows = [
        {
            "history_tokens_estimate": 1000,
            "active_tokens_estimate": 100,
            "full": {"nll": 1.0, "token_accuracy": 0.5},
            "active": {"nll": 1.2, "token_accuracy": 0.45},
        },
        {
            "history_tokens_estimate": 2000,
            "active_tokens_estimate": 100,
            "full": {"nll": 1.1, "token_accuracy": 0.6},
            "active": {"nll": 1.0, "token_accuracy": 0.62},
        },
    ]
    s = summarize(rows)
    assert s["examples"] == 2
    assert s["active_fraction_p50"] == 0.07500000000000001
    assert s["nll_delta_active_minus_full_p50"] == 0.04999999999999993


def _row(history, active, acc_delta=0.0, nll_delta=0.0):
    return {
        "history_tokens_estimate": history,
        "active_tokens_estimate": active,
        "full": {"nll": 1.0, "token_accuracy": 0.5},
        "active": {"nll": 1.0 + nll_delta, "token_accuracy": 0.5 + acc_delta},
    }


def test_bucket_of_uses_the_structural_harness_boundaries():
    from benchmarks.g2_model_quality import bucket_of
    assert bucket_of(0) == "<=8K"
    assert bucket_of(8191) == "<=8K"
    assert bucket_of(8192) == "8K-32K"          # inclusive low bound
    assert bucket_of(32767) == "8K-32K"
    assert bucket_of(32768) == "32K-128K"
    assert bucket_of(131072) == ">=128K"        # the top bucket has no upper bound
    assert bucket_of(10 ** 7) == ">=128K"


def test_verdict_advances_when_the_working_set_stops_tracking_history():
    from benchmarks.g2_model_quality import summarize_by_bucket, verdict_by_bucket
    rows = [_row(5000, 3000) for _ in range(10)]
    rows += [_row(40000, 4000) for _ in range(10)]
    verdict = verdict_by_bucket(summarize_by_bucket(rows))
    assert verdict["verdict"] == "advance"
    assert verdict["readable_buckets"] == ["<=8K", "32K-128K"]


def test_verdict_kills_on_accuracy_loss_beyond_tolerance():
    from benchmarks.g2_model_quality import summarize_by_bucket, verdict_by_bucket
    rows = [_row(5000, 3000) for _ in range(10)]
    rows += [_row(40000, 4000, acc_delta=-0.05) for _ in range(10)]
    verdict = verdict_by_bucket(summarize_by_bucket(rows))
    assert verdict["verdict"] == "kill"
    assert verdict["worst_token_accuracy_delta_p50"] == pytest.approx(-0.05)


def test_verdict_kills_when_the_active_fraction_grows_with_history():
    from benchmarks.g2_model_quality import summarize_by_bucket, verdict_by_bucket
    rows = [_row(5000, 2000) for _ in range(10)]
    rows += [_row(40000, 30000) for _ in range(10)]     # 75% of a longer history
    verdict = verdict_by_bucket(summarize_by_bucket(rows))
    assert verdict["verdict"] == "kill"
    assert verdict["reason"] == "active fraction grows with history"


def test_verdict_is_inconclusive_with_one_readable_bucket():
    from benchmarks.g2_model_quality import summarize_by_bucket, verdict_by_bucket
    rows = [_row(5000, 3000) for _ in range(10)]
    rows += [_row(40000, 4000) for _ in range(3)]       # below the readable minimum
    verdict = verdict_by_bucket(summarize_by_bucket(rows))
    assert verdict["verdict"] == "inconclusive"
    assert verdict["readable_buckets"] == ["<=8K"]


def test_target_scoring_matches_the_labels_path_on_a_tiny_model():
    """The sliced-head path must agree with the standard `labels=` computation.

    The gate needs the suffix-scored form because the full-logit form OOMs at 32K on a
    24 GiB card; this pins the two to the same numbers on a small model.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from benchmarks.g2_model_quality import score_target

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=1024)
    model = LlamaForCausalLM(config).eval()

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [min(255, (ord(ch) % 250) + 1) for ch in text]

    tokenizer = Tokenizer()
    context = "the cache holds a bounded working set " * 6
    target = "the answer is bounded context"
    got = score_target(model, tokenizer, context, target, max_length=256, device="cpu")

    ids, target_start = None, None
    from benchmarks.g2_model_quality import _encode_with_target
    ids, target_start = _encode_with_target(tokenizer, context, target, 256)
    input_ids = torch.tensor([ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :target_start] = -100
    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        reference_nll = float(out.loss.item())
        reference_acc = float((out.logits[:, target_start - 1:-1].argmax(-1)
                               == input_ids[:, target_start:]).float().mean().item())

    assert got["target_tokens"] == len(ids) - target_start
    assert got["nll"] == pytest.approx(reference_nll, rel=1e-4, abs=1e-4)
    assert got["token_accuracy"] == pytest.approx(reference_acc, abs=1e-6)


def test_min_history_tokens_keeps_only_long_turns():
    from benchmarks.g2_model_quality import build_examples

    msgs = [{"role": "system", "content": "agent"}]
    for i in range(12):
        msgs += [
            {"role": "user", "content": f"inspect module_{i}.py " * 40},
            {"role": "assistant", "content": f"module_{i} looks fine " * 20},
        ]
    unfiltered = build_examples(msgs, token_budget=64, min_history_spans=4)
    filtered = build_examples(msgs, token_budget=64, min_history_spans=4,
                              min_history_tokens=600)
    assert unfiltered and filtered
    assert len(filtered) < len(unfiltered)
    assert all(ex.history_tokens_estimate >= 600 for ex in filtered)
    assert min(ex.history_tokens_estimate for ex in unfiltered) < 600
