#!/usr/bin/env bash
# Is the tail result a property of the flash crowd, or of the numbers chosen for it?
#
# The 30% p99 bar is cleared by 23 admissible cells and every one of them is in the burst regime.
# A reader is entitled to ask whether that is the regime or the two constants that define it - half
# the sessions inside a 4 s window - so this sweeps both: the share of sessions that arrive together
# (a quarter, half, three quarters) and the width of the window they arrive in (1 s, 4 s, 16 s).
#
# It is a *declared* arrival model either way - the public corpus has no timestamps - so the point
# is not to find the true value but to show whether the conclusion moves with it.  It runs the two
# active-set sizes the region is carried by, at the capacity setting the earlier cells used.
#
# Usage: bash scripts/run_g4_burst_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
g3=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
    --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v2.json
    --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json)
HIST=32768,131072,262144

for active in 4096 8192; do
  for share in 0.25 0.5 0.75; do
    for window in 1 4 16; do
      name="g4b-burstsweep-h32768_131072_262144-warm4-tier8-a${active}-s${share}-w${window}-v1"
      $PY benchmarks/g4_routing_replay.py "${g3[@]}" \
        --history-choices "$HIST" --active-tokens "$active" \
        --warm-capacity 4 --tier-capacity 8 --sessions 16 --turns-per-session 8 \
        --regimes burst --burst-share "$share" --burst-window "$window" \
        --out "artifacts/${name}.json" > /dev/null
      echo "wrote artifacts/${name}.json"
    done
  done
done

echo "done: $(ls artifacts/g4b-burstsweep-*.json | wc -l) receipts"
