#!/usr/bin/env bash
# The age axis, actually exercised.
#
# The routing grid's history choices are *targets*, and a session's transcript grows 1.7x per turn
# from 2,048 tokens, so with 8 turns per session no session can exceed **113,300 tokens** whatever
# the largest choice is (measured, and now recorded in every receipt as `realised.history_max`).  The
# grid's "1M-mix" receipts therefore describe 113K sessions, which is exactly the scale the mobility
# argument is *not* about: the claim is that placement stops depending on age, and a workload that
# never ages cannot test it.
#
# This runs the same total number of turns (128, so utilisation and fleet sizing are unchanged)
# spread over 16 turns per session instead of 8, which lets a session reach its target: the 1M target
# is reached at turn 12.  Both the region and the placement decision are then read off a workload
# that contains genuinely old sessions.
#
# Usage: bash scripts/run_g4_long_sessions.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
g3=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
    --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v2.json
    --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json)
HIST=32768,131072,262144,1048576

for active in 4096 8192; do
  for warm in 4 16; do
    tier=8
    if [ "$warm" = "16" ]; then tier=32; fi
    name="g4b-longage-h32768_131072_262144_1048576-warm${warm}-tier${tier}-a${active}-v1"
    $PY benchmarks/g4_routing_replay.py "${g3[@]}" \
      --history-choices "$HIST" --active-tokens "$active" \
      --warm-capacity "$warm" --tier-capacity "$tier" \
      --sessions 8 --turns-per-session 16 \
      --regimes balanced,slow_worker,worker_failure,burst,size_skew \
      --out "artifacts/${name}.json" > /dev/null
    echo "wrote artifacts/${name}.json"
  done
done

echo "done: $(ls artifacts/g4b-longage-*.json | wc -l) receipts"
