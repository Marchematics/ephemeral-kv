"""CPU tests for the G1 accounting layer.

The numbers are assumptions, so what is pinned here is the *structure*: every policy
appears per turn count, the transfer policies move KV twice (out and back), the
recompile policy's cost tracks the working set rather than the history, and the
accounting is labelled as accounting rather than as a measurement.
"""

from __future__ import annotations

import json

from benchmarks.kill_gate_crossover import accounting, crossover, main

POLICIES = {"keep_hbm", "keep_dram", "keep_nvme", "recompute_full", "recompile_active"}


def test_every_policy_is_accounted_once_per_turn_count():
    rows = accounting(turns=32, turn_tokens=512, request_tokens=2048,
                      working_set=4096, decode_tokens=128)
    assert {row["policy"] for row in rows} == POLICIES
    assert all(row["history_tokens"] == 32 * 512 for row in rows)
    by_policy = {row["policy"]: row for row in rows}
    assert by_policy["keep_dram"]["bytes_moved"] > 0
    assert by_policy["keep_hbm"]["bytes_moved"] == 0
    assert by_policy["recompile_active"]["resident_bytes"] < by_policy["keep_hbm"]["resident_bytes"]


def test_recompile_cost_tracks_the_working_set_not_the_history():
    short = accounting(8, 512, 2048, 4096, 128)
    long = accounting(512, 512, 2048, 4096, 128)
    active = {r["history_tokens"]: r["seconds"] for r in short if r["policy"] == "recompile_active"}
    active_long = {r["history_tokens"]: r["seconds"] for r in long if r["policy"] == "recompile_active"}
    # the index scan grows with history, but nothing else does
    growth = list(active_long.values())[0] / list(active.values())[0]
    assert 1.0 < growth < 4.0
    full = {r["history_tokens"]: r["seconds"] for r in long if r["policy"] == "recompute_full"}
    assert list(full.values())[0] / list(active_long.values())[0] > 5


def test_crossover_is_reported_only_when_recompile_wins():
    rows = accounting(512, 512, 2048, 4096, 128)
    crossings = crossover(rows)
    # Under the declared cost model, recompiling beats a full re-prefill from the
    # first tested session length ...
    assert crossings["recompute_full"] == 512 * 512   # wins at the longest length too
    # ... but it does NOT beat moving a resident KV in from DRAM/NVMe by 512 turns:
    # 8.6 GB over a 25 GB/s link is cheaper than scanning a 256K-token index.  That is
    # the gate signal worth recording - the latency case for EphemeralKV is not made
    # by tiered-storage comparisons; capacity and model-agnosticism are where it has
    # to be made (gates G3-G5).
    assert crossings["keep_dram"] is None
    assert crossings["keep_hbm"] is None      # resident KV is free at request time
    tiny = accounting(1, 8, 2048, 4096, 128)
    assert crossover(tiny)["keep_dram"] is None    # empty session: nothing to beat


def test_cli_labels_the_output_as_accounting(tmp_path):
    out = tmp_path / "g1.json"
    assert main(["--turns", "8", "32", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["kind"] == "accounting"
    assert set(payload["crossover_history_tokens"]) >= {"keep_hbm", "keep_dram"}
    assert "measured" in payload["decision_rule"]
    assert payload["cost_model"]["kv_bytes_per_token"] == 32 * 1024
