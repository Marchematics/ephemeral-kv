#!/bin/bash
# The column the region has been missing: 4,608 tokens.
#
# The paper's measured state floor is 4,608 tokens (4,096 scores -2.40 pp of fidelity, 4,608 scores
# 0.00 pp), and its quality point at that size is already measured on both metrics - fidelity 0.00 pp
# and a decision of 0.150 on 48 paired sessions.  The routing replay, however, only has columns at
# 2,048 / 4,096 / 8,192 / 16,384, so the *admissible* region has been reported at 8,192 when the
# system's own floor is a smaller state.
#
# Two things are missing to add the column, and this runner produces both:
#   1. the warm-prefill cost at 4,608 tokens, measured like the others (the replay's costs must stay
#      measurements, not interpolations);
#   2. the grid cells at that active size, replayed over the same regimes as the capacity grid.
#
# The quality point is *not* re-measured: `g2-compiler-window3k-b4608-v1.json` and
# `g2b-patch-localization-window3k-b4608-n48.json` already carry it.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

# 1. the prefill primitive at 4,608, measured with the same flags as the 2K/4K/8K/16K receipt
/root/qcc/venv/bin/python benchmarks/g3_hardware_primitives.py \
  --model /root/qcc/models/Qwen2.5-0.5B-Instruct \
  --histories 32768 --active 2048 4096 4608 8192 16384 \
  --kv-bytes-per-token 12288 --dtype bfloat16 --attn-implementation flash_attention_2 \
  --repeats 3 --skip-full-prefill \
  --out artifacts/g3-active-prefill-warm-qwen05b-v3.json

# 2. the capacity grid's cells at that active size, same regimes and capacities as `run_g4_grid2.sh`
common=(--g3 artifacts/g3-hardware-primitives-qwen05b-v1.json
        --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v3.json
        --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json
        --sessions 16 --workers 4 --active-tokens 4608)
for hist in "32768,131072,262144" "32768,131072,262144,1048576"; do
  tag=$(echo "$hist" | tr ',' '_')
  for tier in 0 8 32; do
    for warm in 4 16; do
      /root/qcc/venv/bin/python benchmarks/g4_routing_replay.py "${common[@]}" \
        --history-choices "$hist" --warm-capacity "$warm" --tier-capacity "$tier" \
        --out "artifacts/g4b-capacity-h${tag}-warm${warm}-tier${tier}-a4608-v1.json"
    done
  done
done

/root/qcc/venv/bin/python benchmarks/g4_phase_summary.py --glob 'artifacts/g4b-capacity-*-a4608-v1.json' \
  --out artifacts/g4b-capacity-a4608-phase-summary-v1.json
