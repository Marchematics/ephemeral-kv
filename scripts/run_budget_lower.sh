#!/bin/bash
# Can the whole thing fit in 4,096 tokens instead of 8,192?
#
# The measurements say the surface needs ~3K of verbatim newest evidence and that above ~4K the
# decision is limited by retrieval coverage rather than budget.  If a 3,072-token window plus a
# small consolidated far field holds fidelity at a 4,096-token total, the paper's headline state
# halves - and the G4 region doubles, because 4,096 is the column with 63 advancing cells.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

g2common=(--min-history-tokens 32768 --dedup-spans --consolidate --retrieve-multiplier 2)
for b in 4096 6144; do
  /root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
    --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
    --token-budget "$b" "${g2common[@]}" --provenance-terms 8 --max-spans 128 \
    --recency-fraction 0.6 --recency-spans 3 --max-span-fraction 0.25 \
    --compile-mode tail_state --tail-tokens 3072 \
    --out "artifacts/g2-compiler-window3k-b${b}-v1.json"
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
    --token-budget "$b" --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
    --compile-mode tail_state --tail-tokens 3072 \
    --out "artifacts/g2b-patch-localization-window3k-b${b}-n48.json"
done
