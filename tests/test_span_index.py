from __future__ import annotations

from ephemeralkv.index import DurableSpanIndex


def test_irrelevant_history_does_not_turn_lookup_into_a_history_scan():
    idx = DurableSpanIndex()
    idx.append(turn=0, role="tool", text="pytest failure in src/cache.py eviction path")
    _, before = idx.lookup("cache.py eviction")

    for i in range(1000):
        idx.append(turn=i + 1, role="tool",
                   text=f"unrelated telemetry shard_{i} heartbeat nominal")

    hits, after = idx.lookup("cache.py eviction")
    assert [h.turn for h in hits] == [0]
    assert after.total_spans == 1001
    assert after.postings_visited == before.postings_visited


def test_compile_view_obeys_budget_and_restores_chronological_order():
    idx = DurableSpanIndex()
    idx.append(turn=1, role="tool", text="alpha cache.py first evidence", token_estimate=5)
    idx.append(turn=7, role="tool", text="alpha cache.py later evidence", token_estimate=5)
    idx.append(turn=8, role="assistant", text="alpha cache.py latest note", token_estimate=5)

    view, _ = idx.compile_view("alpha cache.py", token_budget=10)
    assert sum(s.token_estimate for s in view) <= 10
    assert [s.turn for s in view] == sorted(s.turn for s in view)
    assert len(view) == 2


def test_serialized_index_is_model_independent_and_round_trips():
    idx = DurableSpanIndex()
    idx.append(turn=2, role="tool", text="/repo/foo.py ValueError stack trace", token_estimate=9)
    raw = idx.dumps()
    assert "model_id" not in raw
    assert "checkpoint" not in raw.lower()
    assert "adapter_id" not in raw.lower()
    assert "dtype" not in raw.lower()
    assert "rope" not in raw.lower()

    restored = DurableSpanIndex.loads(raw)
    hits, stats = restored.lookup("foo.py ValueError")
    assert hits[0].turn == 2
    assert stats.total_spans == 1


def test_query_cost_is_postings_based_not_span_count():
    idx = DurableSpanIndex()
    for i in range(200):
        idx.append(turn=i, role="tool", text=f"file_{i}.py unique_{i}")
    _, stats = idx.lookup("unique_73")
    assert stats.total_spans == 200
    assert stats.postings_visited == 1
    assert stats.candidate_spans == 1


def test_dedup_keeps_the_most_recent_copy_and_frees_budget():
    """Trajectories repeat themselves; the view should hold each distinct text once."""
    from ephemeralkv.index import DurableSpanIndex

    idx = DurableSpanIndex()
    idx.append(turn=0, role="tool", text="cache eviction path " * 30, token_estimate=90)
    idx.append(turn=1, role="user", text="what should change in the cache?", token_estimate=8)
    idx.append(turn=2, role="tool", text="cache eviction path " * 30, token_estimate=90)
    query = "what should change in the cache?"

    plain, _ = idx.compile_view(query, token_budget=400, max_spans=8)
    deduped, _ = idx.compile_view(query, token_budget=400, max_spans=8, dedup=True)
    assert sum(s.token_estimate for s in deduped) < sum(s.token_estimate for s in plain)
    copies = [s for s in deduped if "cache eviction path" in s.text]
    assert len(copies) == 1
    assert copies[0].turn == 2                      # the newer revision, not the older one


def test_dedup_distinguishes_genuinely_different_text():
    from ephemeralkv.index import DurableSpanIndex

    idx = DurableSpanIndex()
    idx.append(turn=0, role="tool", text="module alpha uses an LRU cache", token_estimate=8)
    idx.append(turn=1, role="tool", text="module beta uses a FIFO queue", token_estimate=8)
    kept, _ = idx.compile_view("module cache queue", token_budget=100, max_spans=8,
                               dedup=True)
    assert len(kept) == 2


def test_snippet_keeps_the_line_the_query_needs_and_its_context():
    """Prefix truncation keeps the head of a dump; the task is usually further down."""
    from ephemeralkv.index import DurableSpanIndex

    body = [f"line {i} of an unrelated module" for i in range(200)]
    body[150] = "the eviction policy in cache.py drops the wrong entry"
    idx = DurableSpanIndex()
    idx.append(turn=0, role="tool", text="\n".join(body), token_estimate=2000)

    prefix, _ = idx.compile_view("fix the eviction policy in cache.py", token_budget=200,
                                 max_spans=8, max_span_fraction=0.25)
    snippet, _ = idx.compile_view("fix the eviction policy in cache.py", token_budget=200,
                                  max_spans=8, max_span_fraction=0.25, snippet=True)
    target = "the eviction policy in cache.py drops the wrong entry"
    assert target in snippet[0].text
    assert target not in prefix[0].text
    assert "... [truncated] ..." in snippet[0].text
    assert "line 149" in snippet[0].text and "line 151" in snippet[0].text   # context
    assert snippet[0].text.index("line 149") < snippet[0].text.index(target)


def test_snippet_leaves_small_spans_alone():
    from ephemeralkv.index import DurableSpanIndex

    idx = DurableSpanIndex()
    idx.append(turn=0, role="tool", text="short output about cache.py", token_estimate=8)
    kept, _ = idx.compile_view("cache.py", token_budget=100, max_spans=4, snippet=True)
    assert kept[0].text == "short output about cache.py"


def test_snippet_selection_never_exceeds_the_room_it_was_given():
    """A line budget is not a size budget.

    `_snippet_text` keeps the query-relevant lines plus two neighbours of context each, so the
    selected text can be several times the room the caller derived from the token budget - and
    because the caller records `room` as the span's size, the view then exceeds the budget it
    reports (measured: a 27,199-token view claiming 8,192 before this bound existed).
    """
    from ephemeralkv.index import DurableSpanIndex

    idx = DurableSpanIndex()
    # one oversized tool output whose lines all match the query, so every kept line is "relevant"
    body = "\n".join(f"src/cache.py line {i} eviction path" for i in range(4000))
    idx.append(turn=0, role="tool", text=body, token_estimate=4000)
    kept, _ = idx.compile_view("cache eviction", token_budget=1000, max_spans=8,
                               max_span_fraction=0.25, snippet=True)
    assert kept
    selected = sum(len(s.text) for s in kept)
    assert selected <= 0.30 * len(body)          # 25% of the characters, plus the markers
    assert all(s.token_estimate <= 1000 for s in kept)
