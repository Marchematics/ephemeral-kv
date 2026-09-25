#!/bin/bash
# The paper's central quality claim at twice the instances.
#
# "One 8,192-token view holds both qualities" rests on the joint view being statistically tied with
# the best retrieval arm on the decision: at n=48 that tie is +0.081 against plain retrieval at
# 4,096 with a 95% CI of [-0.016, +0.177] and 13 wins / 3 losses - an interval whose lower end is
# close enough to zero to deserve a second sample.  96 paired sessions halves the interval.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 96 --min-history-tokens 32768 \
  --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 --token-budget 8192 \
  --compile-mode tail_state --tail-fraction 0.5 \
  --out artifacts/g2b-patch-localization-windowcompiler-b8192-n96.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 96 --min-history-tokens 32768 \
  --retrieve-multiplier 2 --token-budget 4096 \
  --out artifacts/g2b-patch-localization-raw-b4096-n96.json
