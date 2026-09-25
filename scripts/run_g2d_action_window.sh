#!/usr/bin/env bash
# The fix the action measurement pointed at: spend the window on actions, not on recency.
#
# The mid-session measurement (`g2d-action-turns-state4096-v1.json`) found the bounded state emits
# an executable action in 0.375 of turns against the full transcript's 0.625, and the mechanism is
# what the view *shows*: a median of 6 prior actions against 44, because a window chosen by recency
# of tokens spends its budget on whichever spans happen to be newest.  The action window keeps the
# newest span and then the most recent **action-bearing** spans, filling only what is left with
# ordinary recency: on the first session it holds 16 prior actions against 8 (median).
#
# Same 48 turns, same three arms, same budget - only the window's *selection rule* changes, so the
# comparison isolates which turns the window holds from how many tokens it holds.
#
# Usage: bash scripts/run_g2d_action_window.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

PY=${PY:-/root/qcc/venv/bin/python}
MODEL=${MODEL:-/root/qcc/models/Llama-3.2-1B-Instruct}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$PY" benchmarks/g2d_action_turns.py \
  --jsonl data/sessions-patch-64k.jsonl \
  --model "$MODEL" \
  --min-history-tokens 32768 \
  --max-history-tokens 49152 \
  --max-sessions 24 \
  --actions-per-session 2 \
  --token-budget 4096 --tail-tokens 3584 --compile-mode action_window --tail-cap 0.5 \
  --min-free-mib 9000 \
  --out artifacts/g2d-action-turns-actionwindow4096-v1.json

if [ -f artifacts/g2d-action-turns-actionwindow4096-v1.json ]; then
  rm -f artifacts/g2d-action-turns-actionwindow4096-v1.json.partial.jsonl
fi
