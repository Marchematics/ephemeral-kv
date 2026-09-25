#!/bin/bash
# The join that makes G4 a quality claim rather than a cost claim.
#
# The replay reports which cells advance by active-set size; this join reads the measured quality
# points for those sizes and reports the advancing region under three criteria - fidelity within
# 2 pp of full history, a decision no worse than the full transcript, and a decision no worse than
# plain retrieval at the same budget.  Only the first two are met at 8,192; the third is the bar a
# compiler would have to earn (`VERDICT.md` C11).
#
# The quality points live in `artifacts/quality-points-v1.json`, one measured point per active size
# with the receipts it was read from, so a point can be re-checked rather than trusted.  The 4,096
# point is the windowed state (-0.96 pp at the 3,584-token window), not an arm that caps the newest
# span (-25.00 pp): scoring the column with the latter overstated the distance to admissibility
# tenfold.
#
# The grid is rebuilt from the receipts rather than trusted: each receipt contributes only the
# regimes it was asked to compute, so the two workload regimes add cells instead of repeating the
# balanced ones.  The 6,144-token column is included now that it carries its own measured quality
# point: it had been measured and reported as a separate capacity grid, and leaving it out made the
# join drop a column that passes the fidelity allowance.  `paper_numbers.py` asserts that the 624
# cells measured before the workload regimes existed reproduce exactly.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"

/root/qcc/venv/bin/python benchmarks/g4_phase_summary.py \
  --glob 'artifacts/g4[b-e]-*.json' \
  --out artifacts/g4-all-phase-summary-v3.json

/root/qcc/venv/bin/python benchmarks/g4_quality_join.py \
  --phase artifacts/g4-all-phase-summary-v3.json \
  --points artifacts/quality-points-v1.json \
  --out artifacts/g4-quality-join-v3.json
