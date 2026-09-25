#!/usr/bin/env python
"""Re-derive the paper's headline numbers from the receipts and assert them.

A paper whose numbers live only in its own prose cannot be checked.  This script reads the
artifacts and recomputes every number the manuscript quotes, printing PASS or FAIL per claim with
the value found.  Run it before submitting; if a receipt is regenerated and a number moves, this
is where it shows up.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

TOL = 5e-3          # absolute tolerance for accuracy-like quantities
TOL_PP = 0.05       # percentage points


def load(name: str) -> dict:
    return json.loads(Path(name).read_text())


def assert_complete(path: str) -> None:
    """A killed run leaves a partial artifact at its final path; never quote one as evidence."""
    payload = load(path)
    if payload.get("partial"):
        raise SystemExit(f"{path} is a PARTIAL receipt ({len(payload.get('rows', []))} rows); "
                         f"re-run the arm before quoting it")


def bucket_stats(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("g2_model_quality",
                                                  "benchmarks/g2_model_quality.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g2_model_quality"] = module
    spec.loader.exec_module(module)
    payload = load(path)
    buckets = module.summarize_by_bucket(payload["rows"])
    return buckets


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true", help="emit the numbers as JSON as well")
    args = p.parse_args(argv)

    results: list[tuple[str, bool, str]] = []

    def check(claim: str, got, want, tol=TOL):
        ok = got is not None and abs(float(got) - float(want)) <= tol
        results.append((claim, ok, f"got {got}, want {want}"))

    # --- C1: the state is bounded and the fidelity delta does not move with age
    killer = load("artifacts/g2-killer-table-v4.json")
    turns = killer["fidelity_by_turns"]["g2-compiler-tailstate-tf0.6-b8192-v3.json"]
    check("turns 24-48 fidelity pp", turns["24-48 turns"]["acc_delta_pp_p50"], -1.43, TOL_PP)
    check("turns 48-96 fidelity pp", turns["48-96 turns"]["acc_delta_pp_p50"], 0.0, TOL_PP)
    check("turns 48-96 state p50", turns["48-96 turns"]["state_p50"], 7887, 1)

    # --- C2/C3: the budget floor and the joint arm
    for path in ("artifacts/g2-compiler-window3k-b4096-v1.json",
                 "artifacts/g2-compiler-window3k-b6144-v1.json",
                 "artifacts/g2-compiler-window3k-b8192-v1.json",
                 "artifacts/g2-compiler-window3kraw-b12288-v1.json"):
        assert_complete(path)
    for path, want in (("artifacts/g2-compiler-window3k-b4096-v1.json", -2.40),
                       ("artifacts/g2-compiler-window3k-b6144-v1.json", 0.00),
                       ("artifacts/g2-compiler-window3k-b8192-v1.json", 0.00),
                       ("artifacts/g2-compiler-window3kraw-b12288-v1.json", -6.07)):
        stats = bucket_stats(path).get("32K-128K") or {}
        check(f"{Path(path).stem} fidelity pp", round(100 * (stats.get("token_accuracy_delta_p50") or 0), 2), want, TOL_PP)

    joint = load("artifacts/g2b-patch-localization-windowcompiler-b8192-n96.json")["summary"]
    raw = load("artifacts/g2b-patch-localization-raw-b4096-n96.json")["summary"]
    check("joint arm decision (n=96)", round(joint["active"]["f1"], 3), 0.165, TOL)
    check("raw @4096 decision (n=96)", round(raw["active"]["f1"], 3), 0.169, TOL)
    check("full history decision (n=96)", round(joint["full"]["f1"], 3), 0.089, TOL)

    # --- C4: the compiler attribution
    dedup = load("artifacts/g2b-patch-localization-deduponly-b4096-n48.json")["summary"]
    check("dedup-only decision", round(dedup["active"]["f1"], 3), 0.239, TOL)
    consolidate = load("artifacts/g2-compiler-consolidate-b8192-nopath-v1.json")
    collapse = bucket_stats("artifacts/g2-compiler-consolidate-b8192-v1.json").get("32K-128K") or {}
    keep = bucket_stats("artifacts/g2-compiler-consolidate-b8192-nopath-v1.json").get("32K-128K") or {}
    check("path collapse fidelity pp",
          round(100 * (collapse.get("token_accuracy_delta_p50") or 0), 2), -13.90, TOL_PP)
    check("no-collapse fidelity pp",
          round(100 * (keep.get("token_accuracy_delta_p50") or 0), 2), -5.44, TOL_PP)

    # --- C5/C6: the law and the inversion
    inversion = {row["history"]: row for row in killer["inversion"]}
    check("1M/4K mobility s", inversion[1048576]["mobility_s"], 0.0954, 1e-3)
    check("32K/16K mobility s", inversion[32768]["mobility_s"], 0.4855, 1e-3)

    # --- C7: the routing region
    join = load("artifacts/g4-quality-join-v2.json")
    row8192 = next(r for r in join["rows"] if r["active_tokens"] == 8192)
    check("routing cells at 8192", row8192["cells"], 294, 0)
    check("advancing cells at 8192 (fidelity-admissible)", row8192["advance"], 52, 0)
    check("retrieval-parity cells", join["summary"]["advancing_cells_decision_parity"], 0, 0)

    # --- C8: capacity
    capacity = load("artifacts/g5-capacity-planning-v1.json")["rows"]
    eight_b_state = next(r for r in capacity
                         if r["geometry"] == "8B-class" and r["resident"] == "compiled state"
                         and r["tokens"] == 8192)
    check("8B capacity, state 8192", eight_b_state["sessions_per_worker"], 20.0, TOL)

    # --- C9: failover
    failover = load("artifacts/g5-failover-two-workers-samemodel-v1.json")["rows"]
    check("failover hits", sum(1 for r in failover if r["b_hit"]), 4, 0)
    check("failover rebuild s p50",
          round(statistics.median(r["state_rebuild_s"] for r in failover), 3), 1.387, 1e-2)

    # --- receipts must do what their name claims, not merely contain the expected value
    # (the compaction arm once reported a plausible number while never calling its summariser)
    for name in ("artifacts/g2-compiler-compact-b8192-v1.json",
                 "artifacts/g2-compiler-compact-strongsummary-b8192-v1.json"):
        path = Path(name)
        if not path.exists():
            continue
        rows = load(name)["rows"]
        calls = sum(int(r.get("summariser_calls") or 0) for r in rows)
        results.append((f"{path.stem} actually summarised",
                        calls > 0 and len(rows) > 0,
                        f"{calls} summariser calls over {len(rows)} rows"))

    # --- C4b: the compaction baseline
    compact = bucket_stats("artifacts/g2-compiler-compact-b8192-v1.json").get("32K-128K") or {}
    check("compaction fidelity pp",
          round(100 * (compact.get("token_accuracy_delta_p50") or 0), 2), -23.86, TOL_PP)
    compaction_arm = load("artifacts/g2b-patch-localization-compact-b8192-n48.json")["summary"]
    check("compaction decision", round(compaction_arm["active"]["f1"], 3), 0.191, TOL)

    # --- C3 (stricter end task): the action-level rescoring must stay a bound, not a win
    action = load("artifacts/g2b-action-metric-v1.json")["rows"]
    joint_action = next(r for r in action if r["arm"].startswith("windowcompiler-b8192-n96"))
    check("action-level paired delta (joint arm)", joint_action["paired_action_delta"], 0.010, TOL)
    ci_low, ci_high = joint_action["ci95"]
    results.append(("action-level interval includes zero",
                    ci_low <= 0 <= ci_high, f"ci95 [{ci_low}, {ci_high}]"))

    # --- C1 (structural): dead state
    dead = load("artifacts/g2-dead-state-v1.json")["rows"]
    top5 = sorted(r["top5_share"] for r in dead)
    check("dead state: min top-5 share", round(top5[0], 3), 0.73, TOL)
    check("dead state: max top-5 share", round(top5[-1], 3), 0.96, TOL)

    failures = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<48} {detail}")
    print(f"\n{len(results) - len(failures)}/{len(results)} checks pass")
    if args.json:
        print(json.dumps({name: {"ok": ok, "detail": detail} for name, ok, detail in results},
                         indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
