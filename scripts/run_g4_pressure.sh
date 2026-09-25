#!/bin/bash
# Capacity pressure at a quality-admissible active set.
#
# With 16 sessions, 4 workers and a warm cache of 4 per worker, every session is resident, so the
# balanced and slow-worker regimes never take a cold route and no cold-route law can change their
# decision - which is why all 17 advancing cells at 8,192 tokens are forced mobility.  Production
# fleets do not look like that: a KV cache holds a *fraction* of the sessions in flight (that is
# the premise of tiered context caching), and at that point a cold route is the common case, not
# an incident.
#
# So: warm capacity 1-2 sessions per worker, tier 0 (no cluster KV store, so a cold route is a
# rematerialisation) and tier 8 (a store exists), at 8,192 active tokens and both model
# geometries.  Every cost is a G3 measurement; the workload and the geometry are declared in the
# receipt.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"

common=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
        --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v1.json
        --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json
        --sessions 16 --workers 4 --active-tokens 8192)

for geom in 131072 327680; do
  for warm in 1 2; do
    for tier in 0 8; do
      for hist in "32768,131072,262144" "32768,131072,262144,1048576"; do
        tag=$(echo "$hist" | tr ',' '_')
        /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
          --kv-bytes-per-token "$geom" --warm-capacity "$warm" --tier-capacity "$tier" \
          --history-choices "$hist" \
          --out "artifacts/g4d-pressure-g${geom}-warm${warm}-tier${tier}-h${tag}-v1.json"
      done
    done
  done
done
