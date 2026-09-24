#!/bin/bash
# The de facto baseline this space compares against: model-written compaction.
#
# KVMem reports its gain *over* compaction-only context management, so a paper about what state a
# turn needs has to measure compaction itself: keep the newest evidence verbatim, have the model
# summarise everything older, and pack both into the same 8,192-token budget.  Same instances,
# same metric, paired against the window+retrieval arm and against plain retrieval.
set -x
cd /root/ephemeral-kv || exit 1
export PYTHONPATH=/root/ephemeral-kv
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2" > /dev/null; do sleep 20; done

/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  --token-budget 8192 --min-history-tokens 32768 --compile-mode compact --tail-tokens 3072 \
  --out artifacts/g2-compiler-compact-b8192-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
  --token-budget 8192 --compile-mode compact --tail-tokens 3072 \
  --out artifacts/g2b-patch-localization-compact-b8192-n48.json
