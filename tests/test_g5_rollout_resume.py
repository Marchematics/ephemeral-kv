"""G5 compositions: resume costs and the model-independence of the durable view.

The runner itself needs a GPU and a trace; what is testable on CPU is the arithmetic that
decides the claim - an infeasible full-KV route must stay infeasible (never scaled into a
number), and the ephemeral route must be a function of the active set, not the history.
"""

from __future__ import annotations

import pytest

from benchmarks.g5_rollout_resume import (G3_FULL_PREFILL_S, lookup_cost,
                                          recompile_costs)


def test_infeasible_lengths_are_never_extrapolated():
    verdict = recompile_costs(1048576)
    assert verdict["status"] == "infeasible"
    assert "524288" in verdict["reason"] or "infeasible" in verdict["reason"]
    verdict = recompile_costs(524288)
    assert verdict["status"] == "infeasible"
    assert "seconds" not in verdict


def test_feasible_lengths_scale_and_say_they_are_optimistic():
    verdict = recompile_costs(131072)
    assert verdict["status"] == "estimated"
    assert verdict["seconds"] == pytest.approx(G3_FULL_PREFILL_S[131072])
    assert verdict["optimistic"]
    scaled = recompile_costs(65536)
    assert scaled["seconds"] > G3_FULL_PREFILL_S[32768]
    assert scaled["seconds"] < G3_FULL_PREFILL_S[131072]


def test_lookup_grows_sublinearly_with_the_history_bucket():
    assert lookup_cost(8192) < lookup_cost(32768) < lookup_cost(131072)
    # a 16x larger history costs 2.8x the lookup, which is the sublinearity the plan asks
    # for and which the structural receipt measured
    assert lookup_cost(131072) / lookup_cost(8192) < 4
