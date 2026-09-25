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
    # gate-5 magnitudes at the admissible column: how much, not just where.  The advance rule is
    # `goodput >= 1.5x or p99 gain >= 30%`, so these are the numbers behind the paper's sentence
    advancing = [r for r in load("artifacts/g4-all-phase-summary-v2.json")["rows"]
                 if r["active_tokens"] == 8192 and r["decision"] == "advance"]
    finite = [r["goodput_ratio"] for r in advancing
              if r["goodput_ratio"] not in (None, float("inf"))]
    unbounded = [r for r in advancing if r["goodput_ratio"] == float("inf")]
    balanced = [r["goodput_ratio"] for r in advancing
                if r["regime"] == "balanced" and r["goodput_ratio"] not in (None, float("inf"))]
    check("advancing cells clearing the goodput bar with a finite ratio",
          sum(1 for g in finite if g >= 1.5), 26, 0)
    check("advancing cells whose strongest baseline completes no work", len(unbounded), 26, 0)
    check("balanced goodput ratio p50 (finite cells)",
          round(statistics.median(balanced), 2), 2.15, 1e-2)
    check("cells advancing via the p99 route",
          sum(1 for r in advancing if (r["p99_reduction"] or -9) >= 0.30), 0, 0)
    check("best p99 reduction in the admissible column",
          round(max((r["p99_reduction"] or -9) for r in advancing), 2), 0.26, 1e-2)
    # the admissible region now covers every size with a measured point, including 16,384, whose
    # retrieval-only point is inside tolerance - the join used to report that size as having no
    # quality point and silently dropped its two advancing cells
    row16384 = next(r for r in join["rows"] if r["active_tokens"] == 16384)
    check("admissible region across sizes",
          join["summary"]["advancing_cells_fidelity_admissible"], 117, 0)
    results.append(("the 4,096 column is admissible at the measured floor",
                    bool(row4096.get("fidelity_admissible")),
                    f"fidelity point {row4096.get('fidelity_pp')} pp"))
    check("advancing cells at 16384 (fidelity-admissible)", row16384["advance"], 2, 0)
    results.append(("every active size has a quality point",
                    not join["summary"]["sizes_without_quality_points"],
                    f"missing: {join['summary']['sizes_without_quality_points'] or 'none'}"))
    # gate-5 magnitudes for the region as it now stands: every admissible cell clears the bar, and
    # the 4,096 column is a throughput result rather than a tail-latency one
    admissible = [r for r in load("artifacts/g4-all-phase-summary-v2.json")["rows"]
                  if r["active_tokens"] in (4096, 8192, 16384) and r["decision"] == "advance"]
    finite = [r["goodput_ratio"] for r in admissible
              if r["goodput_ratio"] not in (None, float("inf"))]
    check("admissible cells clearing the goodput bar", len(finite), 79, 0)
    check("admissible cells facing a baseline that completes nothing",
          len(admissible) - len(finite), 38, 0)
    check("admissible goodput ratio p50 (finite)",
          round(statistics.median(finite), 2), 1.61, 1e-2)
    results.append(("every admissible cell clears 1.5x", all(g >= 1.5 for g in finite),
                    f"min {min(finite):.2f}x"))
    cells_4096 = [r for r in load("artifacts/g4-all-phase-summary-v2.json")["rows"]
                  if r["active_tokens"] == 4096 and r["decision"] == "advance"]
    finite_4096 = [r["goodput_ratio"] for r in cells_4096
                   if r["goodput_ratio"] not in (None, float("inf"))]
    check("4,096 column finite-ratio cells", len(finite_4096), 53, 0)
    check("4,096 column goodput ratio p50 (finite)",
          round(statistics.median(finite_4096), 2), 1.57, 1e-2)

    check("region if 4096 were admissible (8192 + 4096 columns)",
          row8192["advance"] + row4096["advance"], 115, 0)
    # the measured state floor: a 3,584-token window inside a 4,608-token total holds the surface,
    # while the same 3,072-token window at a smaller total does not - which is what makes the
    # headline state 4.6-8K rather than 6-8K
    # the measured floor: the same 4,096-token total holds the surface with a 3,584-token window
    # (-0.96 pp) and fails it with a 3,072-token one (-2.40 pp) - so the window is what carries it
    floor = bucket_stats("artifacts/g2-compiler-window3584-b4096-v1.json").get("32K-128K") or {}
    check("4,096-token state (window 3,584) fidelity pp",
          100 * (floor.get("token_accuracy_delta_p50") or 0.0), -0.96, TOL_PP)
    floor_view = statistics.median(row["active_tokens_estimate"]
                                   for row in load("artifacts/g2-compiler-window3584-b4096-v1.json")["rows"])
    check("4,096-token state view p50", round(floor_view), 4004, 20)
    wider = bucket_stats("artifacts/g2-compiler-window3584-b4608-v1.json").get("32K-128K") or {}
    check("4,608-token state (window 3,584) fidelity pp",
          100 * (wider.get("token_accuracy_delta_p50") or 0.0), 0.0, TOL_PP)
    narrow = bucket_stats("artifacts/g2-compiler-window3k-b4096-v1.json").get("32K-128K") or {}
    results.append(("the window, not the total, carries the surface at 4,096",
                    100 * (floor.get("token_accuracy_delta_p50") or 0.0)
                    > 100 * (narrow.get("token_accuracy_delta_p50") or 0.0) + 1.0,
                    f"window 3,584: {100 * (floor.get('token_accuracy_delta_p50') or 0.0):+.2f} pp vs "
                    f"window 3,072: {100 * (narrow.get('token_accuracy_delta_p50') or 0.0):+.2f} pp"))
    # the compiler question at the size the system runs at: materialised state does not rescue the
    # floor, and the window does - the paper's negative result, now measured where it matters most
    materialised = bucket_stats("artifacts/g2-compiler-window3k-farmaterialize-b4096-v1.json").get("32K-128K") or {}
    check("materialised far field at the 4,096 floor pp",
          100 * (materialised.get("token_accuracy_delta_p50") or 0.0), -2.48, TOL_PP)
    narrow_window = bucket_stats("artifacts/g2-compiler-window3k-b4096-v1.json").get("32K-128K") or {}
    window_pp = 100 * (narrow_window.get("token_accuracy_delta_p50") or 0.0)
    materialised_pp = 100 * (materialised.get("token_accuracy_delta_p50") or 0.0)
    results.append(("consolidation beats materialisation at the floor",
                    window_pp > materialised_pp,
                    f"consolidation {window_pp:+.2f} pp vs materialisation {materialised_pp:+.2f} pp "
                    f"at a 4,096-token total"))

    window3k = bucket_stats("artifacts/g2-compiler-window3k-b4096-v1.json").get("32K-128K") or {}
    check("best 4096-token state fidelity pp",
          100 * (window3k.get("token_accuracy_delta_p50") or 0.0), -2.40, TOL_PP)
    # the join's 4,096 point must be that arm, not an arm that replaces the window: the first
    # version of the points file scored this column at -25 pp, which overstated the distance to
    # admissibility by a factor of ten and contradicted C11
    check("join 4096 fidelity point is the best measured state", row4096["fidelity_pp"], -0.96, TOL)
    floor_decision = [r["active"]["f1"] for r in
                      load("artifacts/g2b-patch-localization-window3584-b4096-n48.json")["rows"]
                      if r.get("recorded_files")]
    check("4,096-token floor decision (n=48)", round(statistics.mean(floor_decision), 3), 0.134, TOL)
    check("best 4096-token state with the wider window",
          100 * (bucket_stats("artifacts/g2-compiler-window3584-b4096-v1.json")
                 .get("32K-128K", {}).get("token_accuracy_delta_p50") or 0.0), -0.96, TOL_PP)
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
    # and at the measured floor, which is what the paper's capacity claim now quotes
    eight_b_floor = next(r for r in capacity
                         if r["geometry"].startswith("8B") and r["resident"] == "compiled state"
                         and r["tokens"] == 4096)
    check("8B capacity, state 4096 (the measured floor)",
          eight_b_floor["sessions_per_worker"], 40.0, TOL)
    half_b_floor = next(r for r in capacity
                        if r["geometry"].startswith("Qwen2.5-0.5B") and r["resident"] == "compiled state"
                        and r["tokens"] == 4096)
    half_b_history = next(r for r in capacity
                          if r["geometry"].startswith("Qwen2.5-0.5B") and r["resident"] == "history"
                          and r["tokens"] == 131072)
    check("capacity ratio at the floor (0.5B, 128K history)",
          round(half_b_floor["sessions_per_worker"] / half_b_history["sessions_per_worker"]), 32, 0)

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
        # Calls are not enough: the summary has to be *in the scored context*.  A gate that skips
        # an oversized span left the summary empty on exactly the long-history turns these arms
        # keep, so the arm was a window-only view while its calls, its view size and its numbers
        # all looked right.  This is the check that would have caught it.
        in_view = [int(row.get("summary_tokens_in_view") or 0) for row in rows]
        results.append((f"{path.stem} summary is in the scored context",
                        bool(rows) and all(value > 0 for value in in_view),
                        f"min {min(in_view) if in_view else 0} tokens in view over {len(in_view)} rows"))
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
              100 * (compact.get("token_accuracy_delta_p50") or 0.0), 0.13, TOL_PP)
        check("compaction NLL delta p50 (32K-128K)",
              round(compact.get("nll_delta_active_minus_full_p50") or 0.0, 3), -0.032, TOL)
        compact_view = statistics.median(row["active_tokens_estimate"]
                                         for row in load(str(compact_path))["rows"])
        check("compaction view tokens p50", round(compact_view), 7159, 20)
    else:
        results.append(("compaction baseline", False, "receipt absent - the paper quotes it"))

    # --- the decision column of the same table, and the reference it is read against: the two arms
    # are scored on the same instances, so their full-history means must agree
    # the compaction baseline's paired comparisons, with its summary in the view: both intervals
    # include zero, which is the paper's "equivalent, not worse" claim
    for label, path_a, path_b, want in (
            ("compaction vs same-window retrieval",
             "artifacts/g2b-patch-localization-window6656-b8192-n48.json",
             "artifacts/g2b-patch-localization-compact-b8192-n48.json", 0.042),
            ("compaction vs the shipped view",
             "artifacts/g2b-patch-localization-windowcompiler-b8192-n48.json",
             "artifacts/g2b-patch-localization-compact-b8192-n48.json", -0.030)):
        mean, low, high, n = paired_delta(path_a, path_b)
        check(f"{label}: paired delta", round(mean, 3), want, TOL)
        results.append((f"{label}: interval includes zero", low <= 0 <= high,
                        f"ci95 [{low:.3f}, {high:.3f}] over {n} paired instances"))

    for name, want in (("artifacts/g2b-patch-localization-compact-b8192-n48.json", 0.131),
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

    # --- the killer table past the corpus: across 8.9x of raw history the state does not move.
    # These are the numbers behind Table 7b, and they are the paper's sharpest claim
    transfer = []
    for name in ("real64k", "composed-128k", "composed-256k", "composed-512k", "composed-1m"):
        rows = load(f"artifacts/g2-state-size-{name}-v1.json")["rows"]
        transfer.append((name,
                         round(statistics.median(r["history_tokens"] for r in rows)),
                         round(statistics.median(r["turns"] or 0 for r in rows)),
                         round(statistics.median(r["state_tokens"] for r in rows)),
                         round(statistics.median(r["state_bytes"] for r in rows) / 1024, 1)))
    want = (("real64k", 111084, 42, 8203, 18.1), ("composed-128k", 140666, 291, 6598, 40.1),
            ("composed-256k", 284518, 614, 7174, 25.9), ("composed-512k", 514211, 963, 7010, 25.4),
            ("composed-1m", 983692, 1879, 7039, 24.8))
    for got, expected in zip(transfer, want):
        label = got[0]
        check(f"{label} raw history p50", got[1], expected[1], 2)
        check(f"{label} state tokens p50", got[3], expected[3], 60)
        check(f"{label} state KB p50", got[4], expected[4], 1.0)
    history_growth = transfer[-1][1] / transfer[1][1]
    state_growth = transfer[-1][3] / transfer[1][3]
    results.append(("history grows 7x while the state moves under 10%",
                    history_growth > 6.5 and abs(state_growth - 1.0) < 0.15,
                    f"history x{history_growth:.1f}, state x{state_growth:.2f}"))

    # --- the killer table's quality side: paired against a resident 131K-window reference.  Rows
    # are paired by *index* and the pairing is verified against the per-row history tokens, because
    # the harnesses record no session id - pairing by a missing key compares arbitrary rows, which is
    # how this check first passed with numbers the data does not show
    for bucket, want in (("128k", -0.38), ("256k", -3.26), ("512k", -4.91), ("1m", -3.35)):
        state = load(f"artifacts/g2-composed-{bucket}-state8192-v1.json")["rows"]
        reference = load(f"artifacts/g2-composed-{bucket}-reference131k-v1.json")["rows"]
        assert [r["history_tokens_estimate"] for r in state] == \
               [r["history_tokens_estimate"] for r in reference], f"{bucket}: rows do not correspond"
        gaps = [100 * (s["active"]["token_accuracy"] - r["active"]["token_accuracy"])
                for s, r in zip(state, reference)]
        check(f"composed {bucket}: state vs resident reference pp",
              round(statistics.median(gaps), 2), want, 0.05)
        results.append((f"composed {bucket}: gap spread reported",
                        min(gaps) < 0 and max(gaps) > min(gaps),
                        f"{min(gaps):.1f} to {max(gaps):.1f} pp"))

    # --- the evidence-mass caveat, measured at the long end: more than half of a 764K-token history
    # still shares terms with the query, so the bound is a design choice rather than a ceiling
    mass_long = load("artifacts/g2-evidence-mass-composed-1m-v1.json")["by_history_bucket"]
    long_bucket = next((v for k, v in mass_long.items() if v.get("n")), None)
    check("long-history bucket: history tokens p50", round(long_bucket["history_p50"]), 763515, 2000)
    check("long-history bucket: relevant tokens p50",
          round(long_bucket["relevant_tokens_p50"]), 394863, 2000)
    results.append(("more than half a 764K history still matches the query",
                    long_bucket["relevant_tokens_p50"] / long_bucket["history_p50"] > 0.5,
                    f"{100 * long_bucket['relevant_tokens_p50'] / long_bucket['history_p50']:.0f}%"))

    # --- C5's lookup term: measured on composed histories, and query-driven rather than
    # history-driven - which is the abstraction's own claim, so it is asserted rather than narrated
    lookup_1m = load("artifacts/g2-index-lookup-composed-1m-v1.json")["rows"]
    check("1M composed history, spans", round(statistics.median(r["spans"] for r in lookup_1m)), 2055, 400)
    lookup_p50 = [r["lookup_ms_p50"] for r in lookup_1m]
    results.append(("1M lookup under 5 ms p50 on every example",
                    max(lookup_p50) < 5.0, f"max {max(lookup_p50):.2f} ms"))
    small = [r["lookup_ms_p50"] for r in lookup_1m if r["query_tokens"] < 100]
    large = [r["lookup_ms_p50"] for r in lookup_1m if r["query_tokens"] >= 300]
    results.append(("lookup cost tracks the query, not the history",
                    bool(small) and bool(large) and max(small) < min(large),
                    f"small queries {small}, large queries {large}"))

    # --- the arm frontier: the bounded state is the best *fidelity-admissible* decision, and every
    # arm above it pays in fidelity.  These are the numbers the paper's frontier sentence quotes
    def arm_decision(path):
        rows = [r for r in load(path)["rows"] if r.get("recorded_files")]
        return statistics.mean(r["active"]["f1"] for r in rows), len(rows)

    top, top_n = arm_decision(
        "artifacts/g2b-patch-localization-tailstate-compiledfar-tf0.35-b8192-v1.json")
    check("best fidelity-admissible decision (tf0.35, n=24)", round(top, 3), 0.179, TOL)
    results.append(("... and its instance count", top_n == 24, f"got {top_n}"))
    # fidelity and decision are asserted *per arm*, because reading them from two different arms is
    # how this paragraph was first written wrong: the 0.292 decision belongs to the no-collapse arm
    # (-5.44 pp), not to the collapsing one (-13.90 pp)
    for name, decision_receipt, want_fid, want_dec in (
            ("artifacts/g2-compiler-dedup-b8192-v1.json",
             "artifacts/g2b-patch-localization-raw-b8192-n48.json", -6.94, 0.191),
            ("artifacts/g2-compiler-consolidate-b8192-nopath-v1.json",
             "artifacts/g2b-patch-localization-consolidate-b8192-v1.json", -5.44, 0.292),
            ("artifacts/g2-compiler-statefirst-b8192-v1.json",
             "artifacts/g2b-patch-localization-statefirst-b8192-v1.json", -15.38, 0.242),
            ("artifacts/g2-compiler-consolidate-b8192-v1.json",
             "artifacts/g2b-patch-localization-b8192-v1.json", -13.90, 0.237),
            ("artifacts/g2-compiler-materialize-b8192-v1.json",
             "artifacts/g2b-patch-localization-materialize-b8192-v1.json", -7.21, 0.231),
            ("artifacts/g2-compiler-protectwhole-b8192-v1.json",
             "artifacts/g2b-patch-localization-protectwhole-b8192-v1.json", -20.33, 0.211)):
        stats_ = bucket_stats(name).get("32K-128K") or {}
        check(f"{Path(name).stem[:30]} fidelity pp",
              100 * (stats_.get("token_accuracy_delta_p50") or 0.0), want_fid, TOL_PP)
        if decision_receipt:
            dec, n = arm_decision(decision_receipt)
            check(f"{Path(decision_receipt).stem[:30]} decision", round(dec, 3), want_dec, TOL)

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
    # the concentration does not extrapolate to composed lengths, which is why the paper says the
    # bound is a design choice: at a million tokens the top five spans hold a few percent
    composed = load("artifacts/g2-dead-state-composed-1m-v1.json")["rows"]
    composed_top5 = [100 * row["top5_share"] for row in composed]
    check("composed 1M: top-5 share p50", round(statistics.median(composed_top5)), 3, 1)
    results.append(("composed 1M: top-5 share stays under 15%",
                    max(composed_top5) < 15, f"max {max(composed_top5):.1f}%"))
    check("composed 1M: spans p50",
          round(statistics.median(row["spans"] for row in composed)), 2601, 500)

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
