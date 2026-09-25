#!/usr/bin/env bash
# The two workload-shaped G4 regimes: a flash crowd and a heavy-tailed size mix.
#
# The grid's three original regimes change a *worker* (balanced, slow worker, worker
# death).  These two change the *workload*, which is where the paper's claim has to hold
# if "history size does not imply mobility cost" is a statement about fleets rather than
# about failure handling:
#
#   burst      half the sessions arrive inside a 4 s window, so warm state is evicted on
#              every worker at once and cold routes happen under contention;
#   size_skew  session sizes are Zipf(1.2) over the same choices, so most sessions are 32K
#              and a few are 262K/1M - the mix a `cost ~ history` model is worst at.
#
# Each receipt carries only the two new regimes (`--regimes burst,size_skew`) so the cells
# already measured are neither repeated nor perturbed.  The admissible columns are 4,096
# and 8,192; 6,144 is the capacity column and 16,384 is where the region closes, so both
# are carried as boundary evidence.
#
# Usage: bash scripts/run_g4_regimes.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
g3=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
    --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v2.json
    --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json)

run() {   # run <history-choices> <name-infix> <active> <warm> <tier> [bandwidth]
  local hist=$1 infix=$2 active=$3 warm=$4 tier=$5 bw=${6:-}
  local scaling=${hist//,/_}
  local name="g4b-burst-h${scaling}-warm${warm}-tier${tier}-a${active}${infix}-v1"
  local extra=()
  if [ -n "$bw" ]; then extra=(--bandwidth-gbs "$bw"); fi
  $PY benchmarks/g4_routing_replay.py "${g3[@]}" \
    --history-choices "$hist" --active-tokens "$active" \
    --warm-capacity "$warm" --tier-capacity "$tier" --sessions 16 --turns-per-session 8 \
    --regimes burst,size_skew "${extra[@]}" \
    --out "artifacts/${name}.json" > /dev/null
  echo "wrote artifacts/${name}.json"
}
run "32768,131072,262144" "" 4096 4 8
run "32768,131072,262144" "" 4096 4 0
run "32768,131072,262144" "-bw2.0" 4096 4 8 2.0
run "32768,131072,262144" "" 4096 16 32
run "32768,131072,262144" "" 8192 4 8
run "32768,131072,262144" "" 8192 4 0
run "32768,131072,262144" "-bw2.0" 8192 4 8 2.0
run "32768,131072,262144" "" 8192 16 32
run "32768,131072,262144" "" 6144 4 8
run "32768,131072,262144" "" 6144 4 0
run "32768,131072,262144" "-bw2.0" 6144 4 8 2.0
run "32768,131072,262144" "" 6144 16 32
run "32768,131072,262144" "" 16384 4 8
run "32768,131072,262144" "" 16384 4 0
run "32768,131072,262144" "-bw2.0" 16384 4 8 2.0
run "32768,131072,262144" "" 16384 16 32
run "32768,131072,262144,1048576" "" 4096 4 8
run "32768,131072,262144,1048576" "" 4096 4 0
run "32768,131072,262144,1048576" "-bw2.0" 4096 4 8 2.0
run "32768,131072,262144,1048576" "" 4096 16 32
run "32768,131072,262144,1048576" "" 8192 4 8
run "32768,131072,262144,1048576" "" 8192 4 0
run "32768,131072,262144,1048576" "-bw2.0" 8192 4 8 2.0
run "32768,131072,262144,1048576" "" 8192 16 32
run "32768,131072,262144,1048576" "" 6144 4 8
run "32768,131072,262144,1048576" "" 6144 4 0
run "32768,131072,262144,1048576" "-bw2.0" 6144 4 8 2.0
run "32768,131072,262144,1048576" "" 6144 16 32
run "32768,131072,262144,1048576" "" 16384 4 8
run "32768,131072,262144,1048576" "" 16384 4 0
run "32768,131072,262144,1048576" "-bw2.0" 16384 4 8 2.0
run "32768,131072,262144,1048576" "" 16384 16 32

echo "done: $(ls artifacts/g4b-burst-*.json | wc -l) receipts"
