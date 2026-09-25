#!/bin/bash
# Hotspot: one worker at a tenth of the service rate, which is the regime the plan's wording
# ("hotspot, skew, burst, failure") asks for and which the earlier grid's 0.35 multiplier only
# approximates.  Same measured primitives; only the regime parameter changes, and the receipt
# records it.  Run at the quality-admissible 8,192-token state.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"

common=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
        --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v1.json
        --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json
        --sessions 16 --workers 4 --active-tokens 8192 --slow-worker-speed 0.1)

for geom in 131072 327680; do
  for warm in 1 2; do
    for tier in 0 8; do
      for hist in "32768,131072,262144" "32768,131072,262144,1048576"; do
        tag=$(echo "$hist" | tr ',' '_')
        /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
          --kv-bytes-per-token "$geom" --warm-capacity "$warm" --tier-capacity "$tier" \
          --history-choices "$hist" \
          --out "artifacts/g4e-hotspot-g${geom}-warm${warm}-tier${tier}-h${tag}-v1.json"
      done
    done
  done
done
