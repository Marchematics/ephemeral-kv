#!/bin/bash
# G4 widened: the 4/48 region came from a grid whose only active sets were {2,048, 16,384} and
# whose longest session was 262K tokens.  Both bounds were artefacts of the first grid rather
# than of the cost law:
#
#   * the active set is now a *quality-admissible* operating point.  G2 measured 16,384 as the
#     fidelity requirement before the compiler existed; tail_state holds fidelity parity at
#     8,192 and the end-task result is measured at 4,096 and 2,048, so the grid has to contain
#     the sizes the compiler can actually run at, not just the smallest one;
#   * the KV payload a full-move baseline pays grows with the *session*, while rematerialisation
#     pays the *active set*, so the crossover is a function of history length.  A 1M-token
#     session is where the durability argument lives; its lookup row is extrapolation and the
#     receipt labels it.
#
# Every cost is a G3 measurement; the arrival model is declared in the receipt.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
common=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
        --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v1.json
        --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json
        --sessions 16 --workers 4)

for hist in "32768,131072,262144" "32768,131072,262144,1048576"; do
  tag=$(echo "$hist" | tr ',' '_')
  for active in 2048 4096 8192 16384; do
    # capacity pressure: the cluster tier cannot hold every long-lived session
    for tier in 0 8 32; do
      for warm in 4 16; do
        /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
          --warm-capacity "$warm" --tier-capacity "$tier" --active-tokens "$active" \
          --history-choices "$hist" \
          --out "artifacts/g4b-capacity-h${tag}-warm${warm}-tier${tier}-a${active}-v1.json"
      done
    done
    # fabric speed, including the local measured link
    for bw in 23.3 10.0 2.0; do
      /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
        --warm-capacity 4 --tier-capacity 8 --active-tokens "$active" \
        --bandwidth-gbs "$bw" --history-choices "$hist" \
        --out "artifacts/g4b-fabric-h${tag}-bw${bw}-a${active}-v1.json"
    done
    # forced mobility: a worker disappears and its sessions must move
    for bw in 23.3 2.0; do
      /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
        --warm-capacity 4 --tier-capacity 8 --active-tokens "$active" \
        --bandwidth-gbs "$bw" --history-choices "$hist" \
        --out "artifacts/g4b-failure-h${tag}-bw${bw}-a${active}-v1.json"
    done
  done
done
