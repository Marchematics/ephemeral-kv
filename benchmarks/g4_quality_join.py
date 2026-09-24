#!/usr/bin/env python
"""Express G4's advancing region in terms of *quality-admissible* active sets.

A routing result is only a system result if the active set it advances on is one the compiler
can hold quality at.  The widened grid advances in 39 of 264 cells, and the advance counts by
active set are 17/66 at 2,048, 16/66 at 4,096 and 4/66 at 8,192 - so the headline depends
entirely on which of those sizes is admissible.  This joins the phase summary with the measured
quality points and reports the region under three criteria:

* `fidelity`  - teacher-forced fidelity within 2 pp of full history;
* `decision_floor` - end-task F1 no worse than feeding the whole transcript (the floor a system
  must clear to be usable at all);
* `decision_parity` - end-task F1 no worse than plain retrieval at the same budget (the stricter
  bar: the compiler has to earn its place).

Every number comes from a receipt; nothing is interpolated between active sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", action="append",
                   default=[],
                   help="phase summary receipt; repeatable, and rows from every receipt are "
                        "merged (the grids differ in model geometry, history mix and capacity "
                        "pressure, and the union is the region the paper reports)")
    p.add_argument("--points", default="artifacts/quality-points-v1.json")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    phase_paths = args.phase or ["artifacts/g4b-phase-summary-v1.json",
                                 "artifacts/g4c-phase-summary-v1.json",
                                 "artifacts/g4d-phase-summary-v1.json",
                                 "artifacts/g4e-phase-summary-v1.json"]
    phase_rows = []
    for name in phase_paths:
        path = Path(name)
        if path.exists():
            phase_rows += json.loads(path.read_text())["rows"]
    phase = {"rows": phase_rows}
    points = json.loads(Path(args.points).read_text())
    by_size = {int(pt["active_tokens"]): pt for pt in points["points"]}
    floors = points["floors"]

    cells: dict[int, dict[str, int]] = {}
    for row in phase["rows"]:
        size = int(row["active_tokens"])
        entry = cells.setdefault(size, {"cells": 0, "advance": 0})
        entry["cells"] += 1
        if row["decision"] == "advance":
            entry["advance"] += 1

    out_rows = []
    for size in sorted(cells):
        pt = by_size.get(size)
        entry = {"active_tokens": size, **cells[size]}
        if pt:
            fidelity_ok = pt["fidelity_pp"] >= -points["fidelity_tolerance_pp"]
            decision_ok = pt["decision_f1"] > floors["full_history_f1"]
            parity_ok = pt["decision_f1"] >= floors["raw_retrieval_f1"] - points["decision_slack"]
            entry.update({
                "arm": pt["arm"],
                "fidelity_pp": pt["fidelity_pp"],
                "decision_f1": pt["decision_f1"],
                "fidelity_admissible": fidelity_ok,
                "decision_admissible": decision_ok,
                "decision_parity_admissible": parity_ok,
            })
        out_rows.append(entry)

    summary = {
        "advancing_cells_total": sum(r["advance"] for r in out_rows),
        "advancing_cells_fidelity_admissible": sum(
            r["advance"] for r in out_rows if r.get("fidelity_admissible")),
        "advancing_cells_decision_admissible": sum(
            r["advance"] for r in out_rows if r.get("decision_admissible")),
        "advancing_cells_decision_parity": sum(
            r["advance"] for r in out_rows if r.get("decision_parity_admissible")),
        "sizes_without_quality_points": [r["active_tokens"] for r in out_rows
                                         if "fidelity_pp" not in r],
    }
    payload = {"schema": "ephemeral-kv-g4-quality-join-v1", "kind": "derived_join",
               "criteria": {
                   "fidelity_tolerance_pp": points["fidelity_tolerance_pp"],
                   "decision_slack": points["decision_slack"],
                   "floors": floors},
               "rows": out_rows, "summary": summary}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'active':>7} {'cells':>6} {'advance':>8} {'fidelity':>10} {'decision':>9} "
          f"{'adm(fid)':>9} {'adm(dec)':>9} {'adm(>=raw)':>11}  arm")
    for r in out_rows:
        print(f"{r['active_tokens']:>7} {r['cells']:>6} {r['advance']:>8} "
              f"{r.get('fidelity_pp', float('nan')):>10.2f} {r.get('decision_f1', float('nan')):>9.3f} "
              f"{str(r.get('fidelity_admissible', '-')):>9} {str(r.get('decision_admissible', '-')):>9} "
              f"{str(r.get('decision_parity_admissible', '-')):>11}  {r.get('arm', '')}")
    print(json.dumps(summary, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
