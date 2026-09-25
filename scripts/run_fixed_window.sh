#!/bin/bash
# The correct parameterisation of "window + evidence".
#
# The windowed arms scored the same ~0.16 decision at 8,192 and at 12,288 tokens because a
# *fraction* grows the window with the budget: at 50% the split was 4K+4K and then 6K+6K, so the
# ranked share never grew.  What the surface actually needs is a fixed amount of verbatim recent
# context (2.9K was enough for parity in the fraction sweep), which means every token beyond it
# should go to ranked evidence.  This sweep holds the window at 3,072 tokens and varies the total:
#
#   8,192  = 3,072 window + 5,120 ranked
#   12,288 = 3,072 window + 9,216 ranked   <- the ranked budget the compiler needed for 0.292
#   16,384 = 3,072 window + 13,312 ranked
#
# If the 12,288 row holds fidelity parity and the 8,192 row does too, the paper's claim is that a
# bounded execution state of 8-12K carries both quality metrics - which is the 16K -> 8/12K gate.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2" > /dev/null; do sleep 20; done

for b in 8192 12288; do
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
    --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 --token-budget "$b" \
    --compile-mode tail_state --tail-tokens 3072 \
    --out "artifacts/g2b-patch-localization-window3k-b${b}-n48.json"
  /root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
    --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
    --token-budget "$b" --min-history-tokens 32768 --dedup-spans --consolidate \
    --retrieve-multiplier 2 --provenance-terms 8 --max-spans 128 --recency-fraction 0.6 \
    --recency-spans 3 --max-span-fraction 0.25 --compile-mode tail_state --tail-tokens 3072 \
    --out "artifacts/g2-compiler-window3k-b${b}-v1.json"
done
