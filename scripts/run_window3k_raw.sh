#!/bin/bash
# Isolate the window size from the far-field treatment.
#
# window 3,072 + *compiled* far field at 8,192 scores 0.084 (paired -0.107 against window
# 4,096 + raw retrieval), but two things changed at once: the window shrank and the far field
# went back to consolidation, whose snippet and supersede stages are the ones measured to drop
# the file a patch touches.  This holds the treatment fixed (plain retrieval for the far field)
# and varies only the window, at 8,192 and at 12,288 - where the ranked share becomes 9,216
# tokens, the budget at which the compiled arm reached 0.213.
set -x
cd /root/ephemeral-kv || exit 1
export PYTHONPATH=/root/ephemeral-kv
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2" > /dev/null; do sleep 20; done

for b in 8192 12288; do
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
    --retrieve-multiplier 2 --token-budget "$b" --compile-mode tail_state --tail-tokens 3072 \
    --far-compiler raw --out "artifacts/g2b-patch-localization-window3kraw-b${b}-n48.json"
  /root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
    --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
    --token-budget "$b" --min-history-tokens 32768 --retrieve-multiplier 2 \
    --provenance-terms 8 --max-spans 128 --recency-fraction 0.6 --recency-spans 3 \
    --max-span-fraction 0.25 --compile-mode tail_state --tail-tokens 3072 --far-compiler raw \
    --out "artifacts/g2-compiler-window3kraw-b${b}-v1.json"
done
