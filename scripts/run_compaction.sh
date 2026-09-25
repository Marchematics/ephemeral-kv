#!/bin/bash
# The de facto baseline this space compares against: model-written compaction.
#
# KVMem reports its gain *over* compaction-only context management, so a paper about what state a
# turn needs has to measure compaction itself: keep the newest evidence verbatim, have the model
# summarise everything older, and pack both into the same 8,192-token budget.  Same instances,
# same metric, paired against the window+retrieval arm and against plain retrieval.
#
# The window is 6,656 tokens and the summary 512, which is what makes the comparison controlled:
# `run_window_plus_compiler.sh` measures the alternative use of the same remainder (retrieval) with
# the *same* 6,656-token window and the same budget.  Both flags are passed explicitly because this
# script previously said `--tail-tokens 3072` - a window the receipts it names were never produced
# with - and because the receipt records its own configuration only from now on, so the two cannot
# drift apart again without the audit seeing it.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  --token-budget 8192 --min-history-tokens 32768 --compile-mode compact \
  --tail-tokens 6656 --summary-tokens 512 \
  --out artifacts/g2-compiler-compact-b8192-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
  --token-budget 8192 --compile-mode compact --tail-tokens 6656 --summary-tokens 512 \
  --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
  --out artifacts/g2b-patch-localization-compact-b8192-n48.json
