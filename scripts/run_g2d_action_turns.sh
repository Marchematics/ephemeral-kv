#!/usr/bin/env bash
# The executable end task: does the bounded state let the agent take its next real step?
#
# Every quality arm in this repository generates at the *last* turn of a session, and on a coding
# corpus that turn is the agent submitting - measured, 48 of 48 rows, with no file target at all
# (`benchmarks/g2c_action_reproduction.py`).  So the end task has never scored the decision that
# matters: the next action in the middle of a session, the editor call that changes a file.
#
# This runs that measurement.  Both arms generate at the same mid-session turn - the full
# transcript, and the shipped bounded state (a 3,584-token verbatim window plus a consolidated far
# field at a 4,096-token total) - and each continuation is parsed into a tool invocation and
# compared with the action the agent actually took.
#
# Why the 64K-128K corpus: with `min_history_tokens = 32768` on a 32K-64K corpus almost every
# qualifying turn is the last one, which is why the earlier end task only ever scored submissions.
# In the 64K-128K sessions the middle of a session has enough history to be an example.
#
# Usage: bash scripts/run_g2d_action_turns.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

PY=${PY:-/root/qcc/venv/bin/python}
MODEL=${MODEL:-/root/qcc/models/Llama-3.2-1B-Instruct}
# the card is shared: fragmentation from a co-tenant's allocations is what killed two earlier runs
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# three turns per session, evenly spread, and the submission turn excluded: a receipt that only
# scored the end of the session would answer the question `g2c` already answered
"$PY" benchmarks/g2d_action_turns.py \
  --jsonl data/sessions-patch-64k.jsonl \
  --model "$MODEL" \
  --min-history-tokens 32768 \
  --max-history-tokens 49152 \
  --max-sessions 24 \
  --actions-per-session 2 \
  --token-budget 4096 --tail-tokens 3584 --compile-mode tail_state --tail-cap 0.5 \
  --min-free-mib 9000 \
  --out artifacts/g2d-action-turns-state4096-v1.json

# the harness checkpoints every scored turn next to its output, so a run that is interrupted resumes
# instead of regenerating.  If only the payload write failed, assemble it from the checkpoint rather
# than regenerating an hour of views:
#
#   python benchmarks/g2d_action_turns.py --from-checkpoint --jsonl data/sessions-patch-64k.jsonl \
#       --model "$MODEL" --min-history-tokens 32768 --max-history-tokens 49152 \
#       --max-sessions 24 --actions-per-session 2 --token-budget 4096 --tail-tokens 3584 \
#       --compile-mode tail_state --tail-cap 0.5 --out artifacts/g2d-action-turns-state4096-v1.json
#
# the checkpoint is not a receipt and is removed only once the payload exists
if [ -f artifacts/g2d-action-turns-state4096-v1.json ]; then
  rm -f artifacts/g2d-action-turns-state4096-v1.json.partial.jsonl
fi
