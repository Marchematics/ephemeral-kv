"""G4 replays measured costs; the tests pin the costs and the gate arithmetic.

The simulator is only as good as its inputs, so the tests check that (a) the cost model
reproduces the measured primitives, (b) the ephemeral tax does not grow with history while
a full-KV move does, and (c) the routing policies actually behave differently - sticky
pins a session, a KV mover migrates, and an empty tier forces a recompute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.g4_routing_replay import (Costs, KvTier, Turn, build_turns, load_costs,
                                          route)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def _costs() -> Costs:
    return Costs(kv_bytes_per_token=12288, bandwidth_gbs=23.3,
                 lookup_s={8192: 0.000087, 32768: 0.00015, 131072: 0.000246},
                 active_prefill_s={2048: 0.053, 4096: 0.095, 8192: 0.203, 16384: 0.485},
                 full_prefill_s={32768: 1.59, 131072: 14.32, 524288: None})


def test_measured_costs_round_trip_from_the_g3_receipts():
    for name in ("g3-hardware-primitives-qwen05b-v1.json",
                 "g3-active-prefill-warm-qwen05b-v1.json",
                 "g3-fullprefill-qwen05b-v1.json"):
        if not (ARTIFACTS / name).exists():
            pytest.skip(f"{name} not present")
    costs = load_costs(ARTIFACTS / "g3-hardware-primitives-qwen05b-v1.json",
                       ARTIFACTS / "g3-active-prefill-warm-qwen05b-v1.json",
                       ARTIFACTS / "g3-fullprefill-qwen05b-v1.json")
    assert costs.kv_bytes_per_token == 12288
    assert 20 < costs.bandwidth_gbs < 26                  # measured 23.2-23.4 GB/s
    assert costs.active_prefill_s[16384] == pytest.approx(0.485, abs=0.01)
    assert costs.full_prefill_s[524288] is None           # infeasible, not estimated


def test_ephemeral_tax_is_flat_in_history_while_a_move_is_not():
    costs = _costs()
    short, long = costs.ephemeral(32768, 2048), costs.ephemeral(1048576, 2048)
    # the prefill term is identical; only the lookup moves, and it is a fraction of a ms
    assert long - short < 0.001
    assert long < 1.05 * short
    assert costs.move_s(1048576) > 20 * costs.move_s(32768)


def test_history_grows_within_a_session():
    turns = build_turns(sessions=4, turns_per_session=6, seed=0, gap_model="bursty",
                        active_tokens=2048)
    first = min(turns, key=lambda t: (t.session, t.arrival))
    assert first.history <= 4096                          # a session starts short
    by_session: dict[int, list[int]] = {}
    for turn in turns:
        by_session.setdefault(turn.session, []).append(turn.history)
    assert all(histories == sorted(histories) for histories in by_session.values())
    assert max(turns, key=lambda t: t.history).history >= 32768


def test_sticky_pins_a_session_and_a_mover_does_not():
    costs = _costs()
    turns = [Turn(0, 0.0, 32768, 2048), Turn(0, 5.0, 32768, 2048),
             Turn(1, 0.5, 32768, 2048), Turn(1, 5.5, 32768, 2048)]
    sticky = route(turns, costs, workers=2, policy="strict_sticky", warm_capacity=2,
                   tier_capacity=4)
    move = route(turns, costs, workers=2, policy="full_kv_move", warm_capacity=2,
                 tier_capacity=4)
    assert sticky["migrations"] == 2                      # one placement per session
    assert move["migrations"] >= sticky["migrations"]
    assert sticky["unserved"] == 0 and move["unserved"] == 0


def test_an_empty_tier_turns_a_move_into_a_recompute():
    costs = _costs()
    # one worker with room for one warm session: every turn of the other session is cold,
    # so the tier decides whether it is a move or a recompute
    turns = [Turn(0, 0.0, 262144, 2048), Turn(1, 0.5, 262144, 2048),
             Turn(0, 1.0, 262144, 2048), Turn(1, 1.5, 262144, 2048),
             Turn(0, 2.0, 262144, 2048), Turn(1, 2.5, 262144, 2048)]
    no_tier = route(turns, costs, workers=1, policy="full_kv_move", warm_capacity=1,
                    tier_capacity=0)
    with_tier = route(turns, costs, workers=1, policy="full_kv_move", warm_capacity=1,
                      tier_capacity=4)
    # the 262K row is infeasible to re-prefill on this stack, so a mover without a tier
    # cannot serve those turns at all
    assert no_tier["unserved"] > 0 or no_tier["p50_s"] > with_tier["p50_s"]


def test_kv_tier_is_lru_and_bounded():
    tier = KvTier(capacity=2)
    for session in (0, 1, 2):
        tier.store(session)
    assert not tier.holds(0) and tier.holds(1) and tier.holds(2)
    tier.store(1)
    tier.store(3)
    assert tier.holds(1) and not tier.holds(2) and tier.holds(3)
