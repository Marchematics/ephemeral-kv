#!/bin/bash
# The killer table's long-history column, now against a pre-filtered corpus.
#
# The first attempt spent an hour scanning 2,736 sessions to find 24 examples whose history
# exceeds 64K - only 338 sessions in the corpus are that long, and an example reaches 64K only
# near the end of one.  `data/sessions-patch-64k.jsonl` is those 338 sessions, so the scan is
# replaced by a read.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-64k.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 131072 --max-new 96 --max-examples 24 --min-history-tokens 65536 \
  --token-budget 8192 --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
  --out artifacts/g2b-patch-localization-compiled-b8192-long-v2.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-64k.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 131072 --max-new 96 --max-examples 24 --min-history-tokens 65536 \
  --token-budget 8192 --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
  --recency-spans 3 --max-span-fraction 1.0 \
  --out artifacts/g2b-patch-localization-protectwhole-b8192-long-v1.json
