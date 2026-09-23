from __future__ import annotations

import json

from benchmarks.g2_trace_index import evaluate_session, main, messages_from_row, summarize


def _session(n_irrelevant=100):
    msgs = [
        {"role": "system", "content": "coding agent"},
        {"role": "user", "content": "fix src/cache.py eviction bug"},
        {"role": "assistant", "content": "I will inspect the cache implementation"},
        {"role": "tool", "content": "src/cache.py contains def evict and LRU logic"},
    ]
    for i in range(n_irrelevant):
        msgs += [
            {"role": "assistant", "content": f"checking unrelated subsystem {i}"},
            {"role": "tool", "content": f"telemetry_{i}.log heartbeat nominal"},
        ]
    msgs.append({"role": "tool", "content": "cache.py eviction still raises ValueError"})
    return msgs


def test_thoughtworks_messages_json_is_supported():
    row = {"messages_json": json.dumps([{"role": "user", "content": "hello"}])}
    assert messages_from_row(row)[0]["content"] == "hello"


def test_structural_harness_records_index_work_separately_from_history():
    rows = evaluate_session(_session(500), token_budget=128)
    last = rows[-1]
    assert last["history_spans"] > 900
    assert last["postings_visited"] < last["history_spans"] / 10
    assert 0 <= last["selected_fraction"] < 0.1


def test_summary_has_history_buckets():
    rows = evaluate_session(_session(20), token_budget=64)
    s = summarize(rows)
    assert s["calls"] == len(rows)
    assert s["overall"]["postings_per_span_p50"] is not None


def test_cli_marks_quality_as_unmeasured(tmp_path):
    inp = tmp_path / "sample.jsonl"
    inp.write_text(json.dumps({"messages": _session(5)}) + "\n")
    out = tmp_path / "out.json"
    assert main(["--jsonl", str(inp), "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["sessions"] == 1
    assert payload["quality_status"] == "not_measured"
    assert "quality" in payload["decision_rule"]


def test_official_session_max_isl_gets_its_own_bucket():
    rows = evaluate_session(_session(10), token_budget=64, session_max_isl=70_000)
    s = summarize(rows)
    assert "64K-128K" in s["by_session_max_isl_bucket"]
    assert s["by_session_max_isl_bucket"]["64K-128K"]["calls"] == len(rows)


def test_regex_counter_remains_dependency_free():
    from benchmarks.g2_trace_index import make_token_counter

    counter = make_token_counter("regex")
    assert counter("alpha beta gamma") == 3
