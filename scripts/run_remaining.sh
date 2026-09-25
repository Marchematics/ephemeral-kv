#!/bin/bash
# The measurements still outstanding after the window/isolation work, in priority order.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2" > /dev/null; do sleep 20; done

# 1. the equal-budget retrieval reference the compiler comparison needs at n=48
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
  --retrieve-multiplier 2 --token-budget 8192 \
  --out artifacts/g2b-patch-localization-raw-b8192-n48.json

# 2. the one compiler stage that might pay on its own
for b in 4096 8192; do
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
    --dedup-spans --retrieve-multiplier 2 --token-budget "$b" \
    --out "artifacts/g2b-patch-localization-deduponly-b${b}-n48.json"
done

# 3. the killer table's long-history decision column, against the filtered corpus
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-64k.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 131072 --max-new 96 --max-examples 24 --min-history-tokens 65536 \
  --token-budget 8192 --retrieve-multiplier 2 \
  --out artifacts/g2b-patch-localization-raw-b8192-long-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-64k.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 131072 --max-new 96 --max-examples 24 --min-history-tokens 65536 \
  --token-budget 8192 --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
  --out artifacts/g2b-patch-localization-compiled-b8192-long-v2.json
