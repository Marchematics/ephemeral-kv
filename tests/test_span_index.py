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
