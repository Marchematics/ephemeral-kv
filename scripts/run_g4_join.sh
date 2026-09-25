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
# point is the windowed state (-2.40 pp), not an arm that caps the newest span (-25.00 pp): scoring
# the column with the latter overstated the distance to admissibility tenfold.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"

/root/qcc/venv/bin/python benchmarks/g4_quality_join.py \
  --phase artifacts/g4-all-phase-summary-v2.json \
  --points artifacts/quality-points-v1.json \
  --out artifacts/g4-quality-join-v2.json
