#!/usr/bin/env python
"""Emit the paper's figures as CSV from the receipts, so a figure cannot drift from its claim.

Each figure writes `figures/<name>.csv` next to a header line naming the receipts it was built
from.  Nothing is interpolated: every row is read from an artifact, and a missing receipt is
reported rather than silently skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# the repository root: this tool loads the harness modules to reuse their bucket statistics, and
# those import the `ephemeralkv` package, so a direct run must not depend on an exported
# PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIGURES = {
    # figure -> (receipts, columns)
    "fig1_state_vs_age": {
        "receipts": ["artifacts/g2-killer-table-v4.json"],
        "columns": ["bucket", "raw_history_p50", "state_p50", "fidelity_delta_pp", "n"],
        "source": "g2_killer_table by history and by turns",
    },
    "fig2_two_metrics": {
        # each arm needs *both* metrics, so each row names the two receipts it comes from; the
        # instance counts differ between halves and are carried in the CSV rather than averaged away
        "arms": [
            ("plain recency (8,192)",
             "artifacts/g2-compiler-recency-b8192-v1.json",
             "artifacts/g2b-patch-localization-recency-b8192-v1.json"),
            ("window 60% + compiled far (8,192)",
             "artifacts/g2-compiler-tailstate-tf0.6-b8192-v3.json",
             "artifacts/g2b-patch-localization-tailstate-tf0.6-b8192-v3.json"),
            ("window 50% + compiled far (8,192, n=96)",
             "artifacts/g2-compiler-tailstate-compiledfar-tf0.5-b8192-v1.json",
             "artifacts/g2b-patch-localization-windowcompiler-b8192-n96.json"),
            ("evidence consolidation (8,192, n=96)",
             "artifacts/g2-compiler-consolidate-b8192-nopath-v1.json",
             "artifacts/g2b-patch-localization-consolidate-b8192-v1.json"),
            ("raw retrieval (4,096, n=96)",
             "artifacts/g2-compiler-tailquery-cap0.75-b4096-v1.json",
             "artifacts/g2b-patch-localization-raw-b4096-n96.json"),
            ("raw retrieval (8,192, n=48)",
             "artifacts/g2-compiler-dedup-b8192-v1.json",
             "artifacts/g2b-patch-localization-raw-b8192-n48.json"),
            ("log replay into state (8,192)",
             "artifacts/g2-compiler-materialize-b8192-v1.json",
             "artifacts/g2b-patch-localization-materialize-b8192-v1.json"),
            ("state-first selection (8,192)",
             "artifacts/g2-compiler-statefirst-b8192-v1.json",
             "artifacts/g2b-patch-localization-statefirst-b8192-v1.json"),
            ("protected spans whole (8,192)",
             "artifacts/g2-compiler-protectwhole-b8192-v1.json",
             "artifacts/g2b-patch-localization-protectwhole-b8192-v1.json"),
            ("newest span only (8,192)",
             "artifacts/g2-compiler-tailquery-cap0.5-b8192-v1.json",
             "artifacts/g2b-patch-localization-tailquery-cap0.5-b8192-v1.json"),
        ],
        "columns": ["arm", "fidelity_pp", "decision_f1", "n_fidelity", "n_decision"],
        "source": "the compiler ladder with each arm's two receipts named",
    },
    "fig3_compiler_ablation": {
        "receipts": [
            "artifacts/g2b-patch-localization-raw-b4096-n48.json",
            "artifacts/g2b-patch-localization-deduponly-b4096-n48.json",
            "artifacts/g2b-patch-localization-deduponly-b8192-n48.json",
            "artifacts/g2b-patch-localization-compiled-b8192-n48.json",
            "artifacts/g2b-patch-localization-compiled-b12288-n48.json",
            "artifacts/g2b-patch-localization-windowcompiler-b8192-n48.json",
            "artifacts/g2b-patch-localization-windowraw-tf0.5-b8192-n48.json",
            "artifacts/g2b-patch-localization-window3kraw-b8192-n48.json",
        ],
        "columns": ["arm", "budget", "full_f1", "active_f1", "paired_f1"],
        "source": "the end-task ladder, paired on the same instances",
    },
    "fig4_phase_diagram": {
        "receipts": ["artifacts/g4-quality-join-v2.json"],
        "columns": ["active_tokens", "cells", "advance", "fidelity_pp", "decision_f1",
                    "fidelity_admissible", "decision_admissible"],
        "source": "the routing replay joined with the measured quality points",
    },
    "fig5_capacity_recovery": {
        "receipts": ["artifacts/g5-capacity-planning-v1.json",
                     "artifacts/g5-failover-two-workers-samemodel-v1.json"],
        "columns": ["row", "value"],
        "source": "sessions per worker and the process-level failover",
    },
    "fig6_metric_lesson": {
        "receipts": [
            "artifacts/g2-compiler-recency-b8192-v1.json",
            "artifacts/g2-compiler-tailstate-compiledfar-tf0.5-b8192-v1.json",
            "artifacts/g2-compiler-window3kraw-b8192-v1.json",
            "artifacts/g2-compiler-protectwhole-b8192-v1.json",
            "artifacts/g2-compiler-tailquery-cap0.5-b8192-v1.json",
        ],
        "columns": ["arm", "rendering", "fidelity_pp", "n"],
        "source": "fidelity against how the newest evidence is rendered",
    },
}

RENDERING = {
    "artifacts/g2-compiler-recency-b8192-v1.json": "newest spans whole, nothing else",
    "artifacts/g2-compiler-tailstate-compiledfar-tf0.5-b8192-v1.json": "3-4K window whole + consolidated far field",
    "artifacts/g2-compiler-window3kraw-b8192-v1.json": "3K window whole + raw far field",
    "artifacts/g2-compiler-protectwhole-b8192-v1.json": "protected spans whole, ranked rest",
    "artifacts/g2-compiler-tailquery-cap0.5-b8192-v1.json": "newest span only + compiled far field",
}

ARMS = {
    "artifacts/g2-compiler-recency-b8192-v1.json": "plain recency",
    "artifacts/g2-compiler-tailstate-tf0.6-b8192-v3.json": "window 60% + compiled far",
    "artifacts/g2-compiler-tailstate-compiledfar-tf0.5-b8192-v1.json": "window 50% + compiled far",
    "artifacts/g2-compiler-consolidate-b8192-nopath-v1.json": "evidence consolidation",
    "artifacts/g2-compiler-dedup-b8192-v1.json": "raw retrieval (dedup + snippet)",
    "artifacts/g2-compiler-materialize-b8192-v1.json": "log replay into state",
    "artifacts/g2-compiler-statefirst-b8192-v1.json": "state-first selection",
    "artifacts/g2-compiler-protectwhole-b8192-v1.json": "protected spans whole",
    "artifacts/g2-compiler-tailquery-cap0.5-b8192-v1.json": "newest span only",
    "artifacts/g2-compiler-window3kraw-b8192-v1.json": "3K window + raw far",
}


def bucket_stats(path: Path, label: str) -> dict:
    """32K-128K bucket statistics from a model-quality receipt, using the harness summariser."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("g2_model_quality",
                                                  "benchmarks/g2_model_quality.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g2_model_quality"] = module
    spec.loader.exec_module(module)
    payload = json.loads(path.read_text())
    buckets = module.summarize_by_bucket(payload["rows"])
    stats = buckets.get("32K-128K") or next(iter(buckets.values()))
    return {"arm": label, "fidelity_pp": round(100 * (stats["token_accuracy_delta_p50"] or 0), 2),
            "nll_delta": round(stats["nll_delta_active_minus_full_p50"], 3),
            "active_fraction_p50": round(stats.get("active_fraction_p50", 0), 3),
            "n": stats["examples"]}


def end_task_row(path: Path, arm: str, budget: int) -> dict:
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    return {"arm": arm, "budget": budget,
            "full_f1": round(summary["full"]["f1"], 3),
            "active_f1": round(summary["active"]["f1"], 3),
            "paired_f1": round(summary["active"]["f1"] - summary["full"]["f1"], 3)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", default="figures")
    args = p.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written, missing = [], []

    def emit(name: str, rows: list[dict], columns: list[str], source: str, receipts: list[str]):
        path = outdir / f"{name}.csv"
        with path.open("w", newline="") as handle:
            handle.write(f"# source: {source}\n")
            handle.write(f"# receipts: {', '.join(Path(r).name for r in receipts)}\n")
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        written.append(path)

    killer = Path("artifacts/g2-killer-table-v4.json")
    if killer.exists():
        payload = json.loads(killer.read_text())
        rows = []
        for receipt, buckets in payload.get("fidelity_by_turns", {}).items():
            for label, entry in buckets.items():
                if entry.get("n"):
                    rows.append({"raw_history_p50": entry["raw_history_p50"],
                                 "state_p50": entry["state_p50"],
                                 "fidelity_delta_pp": entry["acc_delta_pp_p50"],
                                 "n": entry["n"], "bucket": label})
        emit("fig1_state_vs_age", rows, FIGURES["fig1_state_vs_age"]["columns"],
             FIGURES["fig1_state_vs_age"]["source"], FIGURES["fig1_state_vs_age"]["receipts"])
    else:
        missing.append(str(killer))

    rows = []
    for arm, fidelity_receipt, decision_receipt in FIGURES["fig2_two_metrics"]["arms"]:
        fpath, dpath = Path(fidelity_receipt), Path(decision_receipt)
        if not fpath.exists() or not dpath.exists():
            missing.append(fidelity_receipt if not fpath.exists() else decision_receipt)
            continue
        stats = bucket_stats(fpath, arm)
        decision = json.loads(dpath.read_text())["summary"]
        rows.append({"arm": arm, "fidelity_pp": stats["fidelity_pp"],
                     "decision_f1": round(decision["active"]["f1"], 3),
                     "n_fidelity": stats["n"],
                     "n_decision": decision["examples_with_next_turn_files"]})
    emit("fig2_two_metrics", rows, FIGURES["fig2_two_metrics"]["columns"],
         FIGURES["fig2_two_metrics"]["source"],
         [r for _a, f, d in FIGURES["fig2_two_metrics"]["arms"] for r in (f, d)])

    specs = [("raw retrieval", 4096, "artifacts/g2b-patch-localization-raw-b4096-n48.json"),
             ("dedup only", 4096, "artifacts/g2b-patch-localization-deduponly-b4096-n48.json"),
             ("dedup only", 8192, "artifacts/g2b-patch-localization-deduponly-b8192-n48.json"),
             ("compiler", 8192, "artifacts/g2b-patch-localization-compiled-b8192-n48.json"),
             ("compiler", 12288, "artifacts/g2b-patch-localization-compiled-b12288-n48.json"),
             ("window + compiler", 8192,
              "artifacts/g2b-patch-localization-windowcompiler-b8192-n48.json"),
             ("window + raw", 8192, "artifacts/g2b-patch-localization-windowraw-tf0.5-b8192-n48.json"),
             ("window 3K + raw", 8192,
              "artifacts/g2b-patch-localization-window3kraw-b8192-n48.json"),
             ("window + compiler (n=96)", 8192,
              "artifacts/g2b-patch-localization-windowcompiler-b8192-n96.json"),
             ("raw retrieval (n=96)", 4096,
              "artifacts/g2b-patch-localization-raw-b4096-n96.json")]
    rows = []
    for arm, budget, name in specs:
        path = Path(name)
        if path.exists():
            rows.append(end_task_row(path, arm, budget))
        else:
            missing.append(name)
    emit("fig3_compiler_ablation", rows, FIGURES["fig3_compiler_ablation"]["columns"],
         FIGURES["fig3_compiler_ablation"]["source"], FIGURES["fig3_compiler_ablation"]["receipts"])

    join = Path("artifacts/g4-quality-join-v2.json")
    if join.exists():
        payload = json.loads(join.read_text())
        emit("fig4_phase_diagram", payload["rows"], FIGURES["fig4_phase_diagram"]["columns"],
             FIGURES["fig4_phase_diagram"]["source"], FIGURES["fig4_phase_diagram"]["receipts"])
    else:
        missing.append(str(join))

    rows = []
    capacity = Path("artifacts/g5-capacity-planning-v1.json")
    if capacity.exists():
        for row in json.loads(capacity.read_text())["rows"]:
            rows.append({"row": f"{row['geometry']} / {row['resident']} / {row['tokens']}",
                         "value": row["sessions_per_worker"]})
    else:
        missing.append(str(capacity))
    failover = Path("artifacts/g5-failover-two-workers-samemodel-v1.json")
    if failover.exists():
        payload = json.loads(failover.read_text())
        rows.append({"row": "failover sessions recovered (of 6)",
                     "value": sum(1 for r in payload["rows"] if r["b_hit"])})
        rows.append({"row": "failover rebuild seconds p50",
                     "value": sorted(r["state_rebuild_s"] for r in payload["rows"])[
                         len(payload["rows"]) // 2]})
    else:
        missing.append(str(failover))
    emit("fig5_capacity_recovery", rows, FIGURES["fig5_capacity_recovery"]["columns"],
         FIGURES["fig5_capacity_recovery"]["source"], FIGURES["fig5_capacity_recovery"]["receipts"])

    rows = []
    for name in FIGURES["fig6_metric_lesson"]["receipts"]:
        path = Path(name)
        if path.exists():
            stats = bucket_stats(path, ARMS.get(name, path.stem))
            rows.append({"arm": stats["arm"], "rendering": RENDERING.get(name, ""),
                         "fidelity_pp": stats["fidelity_pp"], "n": stats["n"]})
        else:
            missing.append(name)
    emit("fig6_metric_lesson", rows, FIGURES["fig6_metric_lesson"]["columns"],
         FIGURES["fig6_metric_lesson"]["source"], FIGURES["fig6_metric_lesson"]["receipts"])

    for path in written:
        print("wrote", path)
    if missing:
        # a missing receipt used to be a warning while the figure was still written without that
        # arm, so a figure could silently lose a row and the generator still exit 0: in a fresh
        # clone that is how a reader gets a CSV that no longer matches the paper
        print(f"\nMISSING RECEIPTS ({len(missing)}):")
        for name in missing:
            print(" ", name)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
