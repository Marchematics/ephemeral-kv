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
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# the repository root: some headline numbers are re-derived through the harnesses' own code, which
# imports the `ephemeralkv` package, so a direct run must not depend on an exported PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOL = 5e-3          # absolute tolerance for accuracy-like quantities
TOL_PP = 0.05       # percentage points


def load(name: str) -> dict:
    return json.loads(Path(name).read_text())


def read_csv_rows(path: Path) -> list[dict]:
    import csv
    with path.open() as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


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


def paired_delta(path_a: str, path_b: str, repeats: int = 20000, seed: int = 0):
    """The paired mean difference and 95% interval that `paired_f1.py` reports.

    Recomputed here rather than read from a receipt because the window-axis comparisons are between
    arms that were run separately: the pairing is over the instances the two receipts share.
    """
    def by_instance(name: str) -> dict:
        return {row["instance_id"]: row["active"]["f1"]
                for row in load(name)["rows"] if "active" in row}

    a, b = by_instance(path_a), by_instance(path_b)
    shared = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in shared]
    rng = random.Random(seed)
    means = []
    for _ in range(repeats):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (sum(diffs) / len(diffs), means[int(0.025 * len(means))],
            means[int(0.975 * len(means)) - 1], len(shared))


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
    # C11: the region's width is waiting on the 4,096-token state, so the count in that column and
    # the distance to admissibility are the numbers the paper's "what would widen this" paragraph
    # quotes - and the fidelity figure it calls 0.4 pp outside the allowance
    row4096 = next(r for r in join["rows"] if r["active_tokens"] == 4096)
    check("routing cells at 4096", row4096["cells"], 198, 0)
    check("advancing cells at 4096", row4096["advance"], 63, 0)
    check("region if 4096 were admissible (8192 + 4096 columns)",
          row8192["advance"] + row4096["advance"], 115, 0)
    window3k = bucket_stats("artifacts/g2-compiler-window3k-b4096-v1.json").get("32K-128K") or {}
    check("best 4096-token state fidelity pp",
          100 * (window3k.get("token_accuracy_delta_p50") or 0.0), -2.40, TOL_PP)
    # the join's 4,096 point must be that arm, not an arm that replaces the window: the first
    # version of the points file scored this column at -25 pp, which overstated the distance to
    # admissibility by a factor of ten and contradicted C11
    check("join 4096 fidelity point is the best measured state", row4096["fidelity_pp"], -2.4, TOL)
    # and the no-window retrieval row of Table 9, whose fidelity and decision come from the
    # retrieval-only arm at that budget rather than from an arm that caps the newest span
    lexical = bucket_stats("artifacts/g2-compiler-lexical-b4096-n48.json").get("32K-128K") or {}
    check("plain retrieval, no window, 4096 fidelity pp",
          100 * (lexical.get("token_accuracy_delta_p50") or 0.0), -22.84, TOL_PP)
    raw_rows = {r["instance_id"]: r for r in load("artifacts/g2b-patch-localization-raw-b4096-n48.json")["rows"]}
    raw_f1 = [r["active"]["f1"] for r in raw_rows.values() if r.get("recorded_files")]
    check("plain retrieval, no window, 4096 decision",
          round(statistics.mean(raw_f1), 3), 0.243, TOL)
    results.append(("plain retrieval, no window, 4096 instances",
                    len(raw_f1) == 48, f"got {len(raw_f1)}, want 48"))

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

    # --- receipts must do what their name claims, not merely contain the expected value: the
    # fidelity arm of the compaction baseline once reported a plausible number while never calling
    # its summariser, and the decision arm used to not record whether it had called one at all
    for name, view_proves_summary in (
            ("artifacts/g2-compiler-compact-b8192-v1.json", False),
            ("artifacts/g2b-patch-localization-compact-b8192-n48.json", True)):
        path = Path(name)
        if not path.exists():
            results.append((f"{path.stem} present", False, "receipt absent"))
            continue
        payload = load(name)
        rows = payload["rows"]
        recorded = all("summariser_calls" in row for row in rows)
        calls = sum(int(row.get("summariser_calls") or 0) for row in rows)
        results.append((f"{path.stem} records and makes summariser calls",
                        recorded and calls > 0 and bool(rows),
                        f"{calls} calls over {len(rows)} rows"
                        + ("" if recorded else " (field absent: the arm ran without the fix)")))
        window = (payload.get("config") or {}).get("tail_tokens")
        view_key = ("active_tokens_estimate" if "active_tokens_estimate" in rows[0]
                    else "active_tokens")
        view_max = max(row[view_key] for row in rows)
        if view_proves_summary:
            # a compacted view larger than the verbatim window the arm may keep is a summary *in*
            # the view; the fidelity arm's window packs to just under its cap, so its evidence is
            # the call count above rather than the view size
            results.append((f"{path.stem} view holds the summary",
                            bool(window) and view_max > window,
                            f"max view {view_max} vs window {window}"))

    # --- C4b: the compaction baseline, asserted in the bucket the paper quotes rather than
    # reported from whatever the receipt happens to hold (this arm was withdrawn once)
    compact_path = Path("artifacts/g2-compiler-compact-b8192-v1.json")
    if compact_path.exists():
        compact = bucket_stats(str(compact_path)).get("32K-128K") or {}
        check("compaction fidelity pp (32K-128K)",
              100 * (compact.get("token_accuracy_delta_p50") or 0.0), 0.0, TOL_PP)
        check("compaction NLL delta p50 (32K-128K)",
              round(compact.get("nll_delta_active_minus_full_p50") or 0.0, 3), -0.020, TOL)
        compact_view = statistics.median(row["active_tokens_estimate"]
                                         for row in load(str(compact_path))["rows"])
        check("compaction view tokens p50", round(compact_view), 6650, 5)
    else:
        results.append(("compaction baseline", False, "receipt absent - the paper quotes it"))

    # --- the decision column of the same table, and the reference it is read against: the two arms
    # are scored on the same instances, so their full-history means must agree
    for name, want in (("artifacts/g2b-patch-localization-compact-b8192-n48.json", 0.122),
                       ("artifacts/g2b-patch-localization-window6656-b8192-n48.json", 0.089)):
        path = Path(name)
        if not path.exists():
            results.append((f"{path.stem} decision", False, "receipt absent"))
            continue
        rows = [row for row in load(name)["rows"] if row.get("recorded_files")]
        check(f"{path.stem} mean F1", round(statistics.mean(r["active"]["f1"] for r in rows), 3),
              want, TOL)
        check(f"{path.stem} full-history reference",
              round(statistics.mean(r["full"]["f1"] for r in rows), 3), 0.089, TOL)

    # --- the figures must carry the same numbers as the receipts (data path: receipt -> CSV -> SVG)
    fig3 = Path("figures/fig3_compiler_ablation.csv")
    if fig3.exists():
        rows = read_csv_rows(fig3)
        shipped = next((r for r in rows if r["arm"] == "window + compiler (n=96)"), None)
        if shipped:
            check("figure 3 carries the shipped view's decision",
                  float(shipped["active_f1"]), 0.165, TOL)
        raw96 = next((r for r in rows if r["arm"] == "raw retrieval (n=96)"), None)
        if raw96:
            check("figure 3 carries the retrieval arm's decision",
                  float(raw96["active_f1"]), 0.169, TOL)

    # --- the figures must be *reproducible*, not merely present: regenerate every one of them from
    # the receipts in a temporary directory and require the tracked files to be identical.  This is
    # the data path asserted rather than documented - and because make_figures fails on a missing
    # input, it also asserts that every figure's inputs are in the repository.
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = subprocess.run([sys.executable, "benchmarks/make_figures.py", "--outdir", tmp],
                                 capture_output=True, text=True, check=False)
        drawn = subprocess.run([sys.executable, "benchmarks/make_svg_figures.py", "--dir", tmp],
                               capture_output=True, text=True, check=False)
        tracked = sorted(Path("figures").glob("*"))
        drifted = [path.name for path in tracked
                   if not (Path(tmp) / path.name).exists()
                   or (Path(tmp) / path.name).read_bytes() != path.read_bytes()]
        ok = rebuilt.returncode == 0 and drawn.returncode == 0 and not drifted
        detail = f"{len(tracked)} files, drift: {drifted or 'none'}"
        if not ok and not drifted:
            detail += f" (make_figures exit {rebuilt.returncode}, svg exit {drawn.returncode})"
        results.append(("figures regenerate byte-identically from the receipts", ok, detail))

    # --- C3 (stricter end task): the action-level rescoring must stay a bound, not a win
    action = load("artifacts/g2b-action-metric-v1.json")["rows"]
    joint_action = next(r for r in action if r["arm"].startswith("windowcompiler-b8192-n96"))
    check("action-level paired delta (joint arm)", joint_action["paired_action_delta"], 0.010, TOL)
    ci_low, ci_high = joint_action["ci95"]
    results.append(("action-level interval includes zero",
                    ci_low <= 0 <= ci_high, f"ci95 [{ci_low}, {ci_high}]"))

    # --- the window axis at a fixed 8,192-token budget (Table 9): every step towards a smaller
    # window is positive on the decision, none is resolvable on its own, and all three points hold
    # fidelity - which is what makes the window/budget split the lever rather than the summary
    axis = (("6,656 -> 4,096", "artifacts/g2b-patch-localization-window6656-b8192-n48.json",
             "artifacts/g2b-patch-localization-windowcompiler-b8192-n96.json", 0.072, 48),
            ("6,656 -> 2,867", "artifacts/g2b-patch-localization-window6656-b8192-n48.json",
             "artifacts/g2b-patch-localization-tailstate-compiledfar-tf0.35-b8192-v1.json", 0.069, 24),
            ("4,096 -> 2,867", "artifacts/g2b-patch-localization-windowcompiler-b8192-n96.json",
             "artifacts/g2b-patch-localization-tailstate-compiledfar-tf0.35-b8192-v1.json", 0.040, 24))
    for label, path_a, path_b, want, want_n in axis:
        if not (Path(path_a).exists() and Path(path_b).exists()):
            results.append((f"window axis {label}", False, "receipt absent"))
            continue
        mean, low, high, n = paired_delta(path_a, path_b)
        check(f"window axis {label} paired delta", round(mean, 3), want, TOL)
        results.append((f"window axis {label} interval includes zero",
                        low <= 0 <= high, f"ci95 [{low:.3f}, {high:.3f}]"))
        results.append((f"window axis {label} shared instances", n == want_n,
                        f"got {n}, want {want_n}"))

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
