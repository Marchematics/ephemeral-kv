"""The consolidator: one unit per state, provenance preserved, code never paraphrased.

The G2 receipts put the remaining gap here rather than in retrieval (re-ranking the same text
was within noise of plain truncation; removing duplicated content halved the budget), so the
tests are about the two properties that make this different from both retrieval and
summarisation: repeated views of one file collapse to its *latest* state with the turns
recorded, and whatever is kept of an executable line is kept byte-for-byte.
"""

from __future__ import annotations

from ephemeralkv.consolidate import (Unit, compile_units, consolidate,
                                     content_fingerprint, must_stay_verbatim, path_of,
                                     render, select_verbatim)
from ephemeralkv.index import DurableSpanIndex


def _span(turn, text, tokens=None, role="tool"):
    return type("S", (), {"turn": turn, "span_id": turn, "text": text, "role": role,
                          "token_estimate": tokens or max(1, len(text.split()))})()


FILLER = "The committee reviewed the record and noted the programme continues on schedule."


def test_path_of_finds_the_file_a_span_is_about():
    assert path_of("cat -n src/cache.py\n   1\timport os") == "src/cache.py"
    assert path_of("no file here, just prose") is None
    assert path_of("diff --git a/pkg/mod.py b/pkg/mod.py\n@@ -1 +1 @@") in ("pkg/mod.py",)


def test_verbatim_classification_protects_executable_lines():
    assert must_stay_verbatim("    return self._evict(OrderedDict())")
    assert must_stay_verbatim("Traceback (most recent call last):")
    assert must_stay_verbatim("AssertionError: 3 != 4")
    assert must_stay_verbatim("see pkg/mod.py line 42")
    assert not must_stay_verbatim(FILLER)


def test_superseded_file_views_collapse_to_the_latest_state():
    """Four views of one file must cost one view plus a provenance line."""
    spans = [_span(12, "cat -n cache.py\n   1\tLRU list implementation"),
             _span(19, "cat -n cache.py\n   1\tdict of deques implementation"),
             _span(27, "cat -n cache.py\n   1\tOrderedDict implementation"),
             _span(35, "cat -n cache.py\n   1\tOrderedDict with maxsize") ]
    units = consolidate(spans)
    assert len(units) == 1
    unit = units[0]
    assert "OrderedDict with maxsize" in unit.text          # the latest state
    assert unit.superseded == [12, 19, 27]
    assert unit.turns == [12, 19, 27, 35]
    assert "superseded: 12,19,27" in unit.provenance


def test_identical_text_from_different_turns_is_merged():
    spans = [_span(1, FILLER), _span(5, FILLER), _span(9, FILLER + " Changed.")]
    units = consolidate(spans)
    assert len(units) == 2
    assert [u.turns for u in units][0] == [1, 5]


def test_compile_prefers_recent_state_and_fits_the_budget():
    spans = [_span(t, f"cat -n pkg/mod{i}.py\n   1\tstate of module {i} " * 20)
             for i, t in enumerate(range(0, 400, 20))]
    units = consolidate(spans)
    out, stats = compile_units(units, token_budget=200,
                               tokens_of=lambda text: max(1, len(text.split())))
    assert stats["tokens_used"] <= 200
    assert out, "something must be kept"
    assert max(max(u.turns) for u in out) == 380            # the newest state survives


def test_select_verbatim_drops_prose_before_code():
    code = ["def f(x):", "    return x + 1", "assert f(1) == 2"]
    prose = [FILLER for _ in range(40)]
    text = "\n".join(code + prose)
    kept, verbatim = select_verbatim(text, keep_tokens=20,
                                     tokens_of=lambda line: max(1, len(line.split())))
    assert "def f(x):" in kept and "return x + 1" in kept
    assert kept.count(FILLER) < 40                          # prose was cut, code was not
    assert "... [compiled:" in kept


def test_render_attaches_provenance():
    unit = Unit(text="state", tokens=1, verbatim=True, turns=[3, 7], superseded=[3])
    assert render([unit]).endswith("[turns 3,7] [superseded: 3]")


def test_consolidation_over_a_real_index_shrinks_the_view():
    """End to end on the real index: fewer tokens for the same latest state."""
    idx = DurableSpanIndex()
    for turn in range(30):
        idx.append(turn=turn, role="tool",
                   text=f"cat -n src/cache.py\n   1\trevision {turn} of the eviction path",
                   token_estimate=12)
    spans, _ = idx.lookup("cache.py eviction path", max_spans=32)
    units = consolidate(spans)
    raw = sum(s.token_estimate for s in spans)
    compiled = sum(u.tokens for u in units)
    assert compiled < raw
    assert any("revision 29" in u.text for u in units)
