from __future__ import annotations

import json

from benchmarks.kill_gate_mobility import (inversion, main, route_costs, scaling, transfer_crossover_history)


def test_full_state_mobility_tracks_history_but_ephemeral_tracks_working_set():
    rows = [route_costs(h, 4096) for h in (32768, 131072, 524288, 1048576)]
    s = scaling(rows)
    assert s["history_growth_x"] == 32
    assert s["full_kv_transfer_growth_x"] == 32
    assert s["full_reprefill_growth_x"] == 32
    assert s["ephemeral_rematerialize_growth_x"] == 1
    assert s["ephemeral_resident_growth_x"] == 1


def test_longer_history_can_be_cheaper_to_move_if_active_set_is_smaller():
    x = inversion()
    assert x["history_ratio_long_over_short"] == 32
    assert x["resident_inversion"] is True
    assert x["route_cost_inversion"] is True
    assert x["ephemeral_resident_ratio_long_over_short"] == 0.125
    assert x["full_kv_route_cost_ratio_long_over_short"] == 32


def test_ephemeral_queue_threshold_depends_on_active_set():
    same_history = 1_048_576
    small = route_costs(same_history, 2048)
    large = route_costs(same_history, 16384)
    assert small["full_kv_transfer_s"] == large["full_kv_transfer_s"]
    assert small["ephemeral_rematerialize_s"] < large["ephemeral_rematerialize_s"]


def test_cli_labels_accounting_not_measurement(tmp_path):
    out = tmp_path / "g3.json"
    assert main(["--out", str(out)]) == 0
    p = json.loads(out.read_text())
    assert p["kind"] == "accounting"
    assert "assumptions" in p["honesty"]
    assert "measured" in p["decision_rule"]


def test_declared_full_kv_transfer_crossover_is_long_context():
    cross = transfer_crossover_history(4096)
    assert 700_000 < cross < 900_000
