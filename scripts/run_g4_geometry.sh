#!/bin/bash
# G4's missing axis: the model geometry.
#
# The grid's narrow region came from a *measured* payload of 12,288 B/token - Qwen2.5-0.5B, the
# only model on this card that can hold a 1M-token context at all.  That number is what makes a
# full-KV move cheap: 32K of history is 0.4 GiB (17 ms) and 262K is 3 GiB (138 ms) on the
# measured 23.3 GB/s link, so rematerialising even 8,192 tokens (0.203 s) loses.  Real agent
# sessions do not run on a 0.5B model:
#
#   model class      KV bytes/token   32K session   128K session   512K session
#   0.5B (measured)  12,288           0.38 GiB      1.5 GiB        6.0 GiB
#   8B               131,072          4 GiB         16 GiB         64 GiB
#   70B              327,680          10 GiB        40 GiB         160 GiB
#
# At 16 GiB and above, a full-KV cold route is not a latency choice but an infeasible one (694 ms
# and 1.7 s of pure H2D at the measured rate, before queueing), while the ephemeral route is
# unchanged at 0.203 s for an 8,192-token state - a size the G2 measurements hold *both* quality
# metrics at.  This sweep therefore asks the question the earlier grid could not: at a
# quality-admissible active set, in which regimes does the changed cost law move the decision?
#
# Everything except the declared geometry is a measured primitive; each receipt records both.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
# no GPU is involved (the replay is pure simulation over measured costs), so this must not wait
# on the harness queue

common=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
        --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v1.json
        --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json
        --sessions 16 --workers 4)

for geom in 131072 327680; do
  for active in 8192 4096; do
    for hist in "32768,131072,262144" "32768,131072,262144,1048576"; do
      tag=$(echo "$hist" | tr ',' '_')
      for tier in 0 8 32; do
        for warm in 4 16; do
          /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
            --kv-bytes-per-token "$geom" --active-tokens "$active" \
            --warm-capacity "$warm" --tier-capacity "$tier" --history-choices "$hist" \
            --out "artifacts/g4c-capacity-g${geom}-h${tag}-warm${warm}-tier${tier}-a${active}-v1.json"
        done
      done
      for bw in 23.3 10.0 2.0; do
        /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
          --kv-bytes-per-token "$geom" --active-tokens "$active" --bandwidth-gbs "$bw" \
          --warm-capacity 4 --tier-capacity 8 --history-choices "$hist" \
          --out "artifacts/g4c-fabric-g${geom}-h${tag}-bw${bw}-a${active}-v1.json"
      done
      for bw in 23.3 2.0; do
        /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
          --kv-bytes-per-token "$geom" --active-tokens "$active" --bandwidth-gbs "$bw" \
          --warm-capacity 4 --tier-capacity 8 --history-choices "$hist" \
          --out "artifacts/g4c-failure-g${geom}-h${tag}-bw${bw}-a${active}-v1.json"
      done
    done
  done
done
