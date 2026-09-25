#!/usr/bin/env bash
# Is the emit gap a budget effect or a selection effect?
#
# The mid-session measurement found the bounded 4,096-token state emitting an executable action in
# 0.375 of turns against the full transcript's 0.625, and the selection fix - a window that holds
# more prior actions - did not help (0.229 against 0.375, interval including zero).  That leaves the
# other explanation: the state is simply too small, and what the model needs is more room rather
# than different contents.
#
# This runs the same 48 turns and the same three arms at 8,192 tokens, which is the size whose
# quality point the paper already carries (fidelity 0.00 pp, decision 0.161, and 59 of 310 advancing
# routing cells).  The window follows the measured point - half the budget, 4,096 tokens - so the
# comparison against the 4,096 run changes the size of the state and not the recipe that builds it.
# A dose-response that does not move says the gap is about *what* the view holds; one that closes it
# says the earlier result was a budget artefact and the paper's own 4-8K range contains the answer.
#
# Usage: bash scripts/run_g2d_action_turns_b8192.sh
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
  --token-budget 8192 --tail-fraction 0.5 --compile-mode tail_state --tail-cap 0.5 \
  --min-free-mib 9000 \
  --out artifacts/g2d-action-turns-state8192-v1.json

if [ -f artifacts/g2d-action-turns-state8192-v1.json ]; then
  rm -f artifacts/g2d-action-turns-state8192-v1.json.partial.jsonl
fi
