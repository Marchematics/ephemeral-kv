from __future__ import annotations

from benchmarks.g2_model_quality import build_examples

NEWEST = "NEWEST TOOL DUMP " + "line of the file just read " * 60


def _trajectory():
    """A long session whose newest tool result is far too big for the budget."""
    msgs = [{"role": "system", "content": "coding agent"}]
    msgs += [
        {"role": "user", "content": "fix src/cache.py eviction bug"},
        {"role": "tool", "content": "src/cache.py contains an LRU eviction path"},
        {"role": "assistant", "content": "I will inspect src/cache.py"},
    ]
    for i in range(12):
        msgs += [
            {"role": "user", "content": f"inspect unrelated module {i}"},
            {"role": "tool", "content": f"module_{i}.py telemetry nominal " * 8},
            {"role": "assistant", "content": f"module {i} is unrelated"},
        ]
    msgs += [
        {"role": "user", "content": "what should we change in src/cache.py?"},
        {"role": "tool", "content": NEWEST},
        {"role": "assistant", "content": "change the LRU eviction path in src/cache.py"},
    ]
    return msgs


def _tail_state(budget=400, fraction=0.6, **extra):
    compiler = {"compile_mode": "tail_state", "tail_fraction": fraction,
                "consolidate": True, "retrieve_multiplier": 2, "dedup": True,
                "provenance_terms": 4}
    compiler.update(extra)
    return build_examples(_trajectory(), token_budget=budget, min_history_spans=8,
                          compiler=compiler)


def test_newest_span_is_admitted_whole_and_last():
    """The current turn is not optional, even when it is bigger than the tail share.

    The tail share governs how much *older* context is protected; it must not decide whether
    the turn being answered is visible.  The newest span therefore lands whole at the end of
    the view, byte-for-byte, even though it costs more than 60% of this budget.
    """
    examples = _tail_state()
    assert examples
    ex = examples[-1]
    assert NEWEST in ex.active_context          # byte-for-byte, not summarised
    assert ex.active_context.rstrip().endswith(NEWEST.rstrip())
    assert ex.active_context.count(NEWEST) == 1


def test_the_query_turn_is_the_last_span_when_nothing_follows_it():
    """With a small newest span the view ends at the newest indexed span, in order."""
    msgs = _trajectory()[:-2] + [
        {"role": "user", "content": "what should we change in src/cache.py?"},
        {"role": "assistant", "content": "change the LRU eviction path in src/cache.py"},
    ]
    examples = build_examples(msgs, token_budget=400, min_history_spans=8,
                              compiler={"compile_mode": "tail_state", "tail_fraction": 0.6,
                                        "consolidate": True, "dedup": True})
    ex = examples[-1]
    assert ex.active_context.rstrip().endswith("what should we change in src/cache.py?")


def test_span_larger_than_the_whole_budget_keeps_its_suffix():
    """A span bigger than the entire view must not be dropped or blow the budget."""
    huge = "HUGE NEWEST OUTPUT " + " ".join(f"w{i}" for i in range(5000))
    msgs = _trajectory()[:-2] + [
        {"role": "user", "content": "what should we change in src/cache.py?"},
        {"role": "tool", "content": huge},
        {"role": "assistant", "content": "change the LRU eviction path in src/cache.py"},
    ]
    examples = build_examples(msgs, token_budget=400, min_history_spans=8,
                              compiler={"compile_mode": "tail_state", "tail_fraction": 0.6,
                                        "consolidate": True, "dedup": True})
    assert examples
    ex = examples[-1]
    assert ex.active_context.strip()                    # not empty
    assert "earlier lines elided" in ex.active_context  # the cut is marked
    assert "w4999" in ex.active_context                 # the end of the turn is what survives
    assert ex.active_tokens_estimate <= 400 * 1.1       # and the budget still holds


def test_far_field_comes_before_the_tail_and_excludes_it():
    """Order matters: the compiled far field precedes the verbatim working set."""
    small_newest = "SMALL NEWEST OUTPUT about src/cache.py"
    msgs = [{"role": "system", "content": "coding agent"},
            {"role": "user", "content": "fix src/cache.py eviction bug"},
            {"role": "tool", "content": "src/cache.py contains an LRU eviction path"},
            {"role": "assistant", "content": "I will inspect src/cache.py"}]
    for i in range(12):
        msgs += [
            {"role": "user", "content": f"inspect unrelated module {i}"},
            {"role": "tool", "content": f"module_{i}.py telemetry nominal"},
            {"role": "assistant", "content": f"module {i} is unrelated"},
        ]
    msgs += [
        {"role": "user", "content": "what should we change in src/cache.py?"},
        {"role": "tool", "content": small_newest},
        {"role": "assistant", "content": "change the LRU eviction path in src/cache.py"},
    ]
    examples = build_examples(msgs, token_budget=600, min_history_spans=8,
                              compiler={"compile_mode": "tail_state", "tail_fraction": 0.5,
                                        "consolidate": True, "dedup": True,
                                        "provenance_terms": 4, "retrieve_multiplier": 2})
    ex = examples[-1]
    assert small_newest in ex.active_context
    head = ex.active_context[:ex.active_context.index(small_newest)]
    assert head.strip()                          # there is a far field at this budget
    assert "eviction" in head.lower()            # the far-field query match is in it
    assert small_newest not in head              # and the tail is not duplicated into it


def test_tail_budget_is_respected_and_bounded():
    for fraction in (0.0, 0.25, 0.6, 1.0):
        examples = _tail_state(fraction=fraction)
        ex = examples[-1]
        assert ex.active_tokens_estimate <= 400 * 1.1      # estimate vs tokenizer drift
        if fraction == 0.0:
            # the current turn is still not optional - only *older* context stops being
            # protected, so everything else in the view was retrieved by the far field
            assert ex.active_context.rstrip().endswith(NEWEST.rstrip())


def test_turns_are_recorded_on_every_example():
    """Session length is a first-class axis: the killer table is a claim about it."""
    examples = _tail_state()
    assert examples
    assert all(ex.turns >= 8 for ex in examples)
    assert examples[-1].turns > examples[0].turns


def test_recency_mode_keeps_the_newest_spans_whole():
    """The control arm must not truncate or rank: it takes whole spans, newest first."""
    examples = build_examples(_trajectory(), token_budget=400, min_history_spans=8,
                              compiler={"compile_mode": "recency"})
    assert examples
    ex = examples[-1]
    assert NEWEST in ex.active_context
    assert ex.active_context.rstrip().endswith(NEWEST.rstrip())
    assert ex.active_tokens_estimate <= 400 * 1.1       # whole spans, within budget


def test_truncating_the_newest_span_does_not_corrupt_the_target_or_the_index():
    """A compile branch must not rebind the loop's message text.

    `text` becomes both the example target and the next span appended to the durable index, so
    a branch that rebinds it (as the first version of the oversized-span truncation did) scores
    the model against a truncated dump and grows every later history by the size of that dump.
    """
    huge = "HUGE NEWEST OUTPUT " + " ".join(f"w{i}" for i in range(4000))
    msgs = _trajectory()[:-2] + [
        {"role": "user", "content": "what should we change in src/cache.py?"},
        {"role": "tool", "content": huge},
        {"role": "assistant", "content": "change the LRU eviction path in src/cache.py"},
        {"role": "user", "content": "and what about the metrics?"},
        {"role": "assistant", "content": "the metrics stay in src/cache.py"},
    ]
    examples = build_examples(msgs, token_budget=300, min_history_spans=8,
                              compiler={"compile_mode": "tail_state", "tail_fraction": 0.6,
                                        "consolidate": True, "dedup": True})
    targets = [ex.target for ex in examples]
    assert "the metrics stay in src/cache.py" in targets
    assert "change the LRU eviction path in src/cache.py" in targets
    assert not any(t.startswith("HUGE NEWEST OUTPUT") for t in targets)
    # the index must keep the whole tool result, not the truncated view of it: a rebind costs
    # every later history the size of that span
    assert examples[-1].history_tokens_estimate >= 4000
    assert examples[-1].history_tokens_estimate > examples[-2].history_tokens_estimate
