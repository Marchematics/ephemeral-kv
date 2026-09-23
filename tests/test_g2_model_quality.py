from __future__ import annotations

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
