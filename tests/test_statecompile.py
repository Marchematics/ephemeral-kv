"""The executable-state compiler: replay the log, keep the state, drop what was superseded.

The first compiler iteration kept a file's *latest view*, which the measurement rejected
(-13.90 pp at 8K against -6.94 pp for plain truncation) because a later view is usually a diff
or one region rather than the file.  These tests pin the replacement rule: full dumps set state,
diffs and search/replace blocks apply to it, a command's repeated runs collapse to its latest
result, and nothing executable is ever paraphrased.
"""

from __future__ import annotations

from ephemeralkv.statecompile import (compile_executable_state, materialise,
                                      _apply_editor, _apply_hunks, _dump_lines)


def _span(turn, text, role="tool"):
    return type("S", (), {"turn": turn, "span_id": turn, "text": text, "role": role,
                          "token_estimate": max(1, len(text.split()))})()


DUMP = ("cat -n pkg/cache.py\n"
        "     1\tclass Cache:\n"
        "     2\t    def evict(self):\n"
        "     3\t        return self._lru.pop()\n"
        "     4\t\n"
        "     5\tCACHE = Cache()\n")


def test_a_full_dump_sets_the_file_state():
    assert _dump_lines(DUMP) == ["class Cache:", "    def evict(self):",
                                "        return self._lru.pop()", "", "CACHE = Cache()"]
    files, results, loose = materialise([_span(1, DUMP)])
    state = files["pkg/cache.py"]
    assert state.text.startswith("class Cache:")
    assert state.turns == [1]


def test_a_unified_diff_applies_to_the_current_state():
    diff = ("diff --git a/pkg/cache.py b/pkg/cache.py\n"
            "--- a/pkg/cache.py\n"
            "+++ b/pkg/cache.py\n"
            "@@ -1,3 +1,3 @@\n"
            " class Cache:\n"
            "     def evict(self):\n"
            "-        return self._lru.pop()\n"
            "+        return self._lru.popitem()\n")
    files, _, _ = materialise([_span(1, DUMP), _span(2, diff)])
    state = files["pkg/cache.py"]
    assert "popitem" in state.text and "pop()" not in state.text
    assert state.turns == [1, 2] and state.events == 2


def test_a_search_replace_block_applies_to_the_current_state():
    block = ("*** Update File: pkg/cache.py\n"
             "<<<<<<< SEARCH\n"
             "        return self._lru.pop()\n"
             "=======\n"
             "        return self._lru.popitem(last=False)\n"
             ">>>>>>> REPLACE\n")
    files, _, _ = materialise([_span(1, DUMP), _span(2, block)])
    assert "popitem(last=False)" in files["pkg/cache.py"].text


def test_hunk_application_survives_a_moved_region():
    base = ["a", "b", "c", "d", "e"]
    diff = "@@ -2,2 +2,2 @@\n b\n-c\n+C\n"
    out, applied = _apply_hunks(base, diff)
    assert applied == 1 and out == ["a", "b", "C", "d", "e"]


def test_repeated_command_runs_collapse_to_the_latest_result():
    first = "pytest -q tests/test_cache.py\n1 failed, 3 passed"
    second = "pytest -q tests/test_cache.py\n4 passed"
    files, results, _ = materialise([_span(1, first), _span(7, second)])
    assert len(results) == 1
    result = next(iter(results.values()))
    assert result.text == second and result.turn == 7
    assert result.superseded == [1]


def test_repeated_reasoning_keeps_only_the_newest_occurrence():
    ack = "Let me check the cache implementation once more."
    _, _, loose = materialise([_span(1, ack), _span(5, ack)])
    assert len(loose) == 1 and loose[0].turns == [1, 5]


def test_compiled_state_fits_the_budget_and_keeps_the_current_file():
    spans = [_span(1, DUMP)]
    for turn in range(2, 40):
        spans.append(_span(turn, f"pytest -q\n{failure(turn)}"))
        spans.append(_span(turn, DUMP.replace("pop()", f"pop{turn}()")))
    units, stats = compile_executable_state(spans, token_budget=120,
                                            tokens_of=lambda t: max(1, len(t.split())))
    assert stats["tokens_used"] <= 120
    text = "\n".join(u.text for u in units)
    assert "pop39()" in text                      # the newest materialised state survives
    assert stats["superseded_events"] >= 1        # earlier states were recorded as replaced


def failure(turn):
    return f"{turn} failed, 2 passed" if turn % 2 else f"{turn} passed"


def test_classify_separates_state_from_chatter():
    from ephemeralkv.statecompile import classify

    assert classify(DUMP) == "file_event"
    assert classify("diff --git a/pkg/cache.py b/pkg/cache.py\n@@ -1 +1 @@\n-a\n+b") == "file_event"
    assert classify("pytest -q tests/test_cache.py\n4 passed") == "command_result"
    assert classify("Let me look at the cache one more time.") == "loose"


def test_state_first_selection_prefers_state_over_chatter():
    """The measured failure mode: lexical top-k returns 74% chatter, so state never fits."""
    from benchmarks.g2_model_quality import DurableSpanIndex
    from ephemeralkv.statecompile import state_first_units

    idx = DurableSpanIndex()
    for turn in range(20):
        idx.append(turn=turn, role="assistant",
                   text=f"cache eviction chatter number {turn} about the cache and eviction",
                   token_estimate=20, kind="loose")
    idx.append(turn=20, role="tool", text=DUMP, token_estimate=40, kind="file_event")
    idx.append(turn=21, role="tool", text="pytest -q\n4 passed", token_estimate=10,
               kind="command_result")
    units, stats = state_first_units(idx, "fix the cache eviction", token_budget=80,
                                     tokens_of=lambda t: max(1, len(t.split())))
    text = "\n".join(u.text for u in units)
    assert "class Cache:" in text                      # the file state is in the view
    assert "4 passed" in text                          # and so is the latest result
    assert stats["tokens_used"] <= 80
    assert stats["kind_file_spans"] == 1 and stats["kind_loose"] >= 20
