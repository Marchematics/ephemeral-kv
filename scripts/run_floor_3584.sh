#!/bin/bash
# The window, not the budget: a 3,584-token window inside a 4,096-token state.
#
# The budget floor was 6,144 tokens for two rounds because the arms that measured it held the
# *window* at 3,072 while varying the total - and at a 4,096-token total that window scored -2.40 pp.
# Widening the window to 3,584 inside the same 4,096-token total puts the surface back inside the
# allowance (-0.96 pp), which says the binding quantity is how much newest evidence is kept whole,
# not how large the budget is.
#
# These arms are what move the paper's state from 6-8K to **4-8K**, and 4,096 is the routing replay's
# column with 63 advancing cells, so the admissible region grows with the state shrinking.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

common=(--model /root/qcc/models/Llama-3.2-1B-Instruct --max-examples 48 --min-history-tokens 32768
        --dedup-spans --consolidate --retrieve-multiplier 2 --provenance-terms 8 --max-spans 128
        --recency-fraction 0.6 --recency-spans 3 --max-span-fraction 0.25
        --compile-mode tail_state --far-compiler consolidate)

# 3,584-token window at a 4,096-token total: the surface holds (-0.96 pp)
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  "${common[@]}" --max-length 131072 --token-budget 4096 --tail-tokens 3584 \
  --out artifacts/g2-compiler-window3584-b4096-v1.json
# and the end task at the same configuration, which the routing join needs as its quality point
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl "${common[@]}" --max-length 65536 --max-new 96 \
  --token-budget 4096 --tail-tokens 3584 \
  --out artifacts/g2b-patch-localization-window3584-b4096-n48.json
# the compiler question at the floor size: the same 4,096-token total and the same 3,072-token
# window as the -2.40 pp arm, but with a *materialised* far field (-2.48 pp) - no gain, which is
# why the floor is bought by widening the window rather than by compiling harder
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  "${common[@]}" --max-length 131072 --token-budget 4096 --tail-tokens 3072 \
  --far-compiler materialize \
  --out artifacts/g2-compiler-window3k-farmaterialize-b4096-v1.json

# 3,584-token window at a 4,608-token total: the same window with more far field
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  "${common[@]}" --max-length 131072 --token-budget 4608 --tail-tokens 3584 \
  --out artifacts/g2-compiler-window3584-b4608-v1.json
