"""CPU tests for the kill-gate harness: the gate's contract, not its numbers."""

from __future__ import annotations

import json

from benchmarks.kill_gate_crossover import main


def test_gate_writes_a_ledger_shaped_artifact(tmp_path):
    out = tmp_path / "gate.json"
    assert main(["--model", "dummy", "--turns", "8", "32", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["schema"].startswith("ephemeral-kv-kill-gate")
    assert payload["status"] == "not-implemented"      # until the harness is filled in
    assert payload["config"]["turns"] == [8, 32]
    assert payload["rows"] == []
    assert "recompile_active" in payload["decision_rule"]
