#!/bin/bash
# "Window + compiler", with both halves treated the way each half's winning arm treats them.
#
# The fidelity winner (recency: +0.29 pp) keeps the recent spans whole; the decision winner
# (evidence consolidation: F1 0.292) snippet-selects oversized spans and fills the whole budget
# with ranked evidence.  tail_state combined them but passed max_span_fraction=1.0 to its far
# field, so oversized far-field spans were *dropped* instead of snippet-selected - the arm was
# measuring a difference in span treatment, not a difference in what the budget is spent on.
# With that fixed, this is the arm that should hold both: the window whole, and every remaining
# token spent on exactly the evidence the decision winner uses.
set -x
cd /root/ephemeral-kv || exit 1
export PYTHONPATH=/root/ephemeral-kv
# match only the interpreter running the harness: a pattern like "g2_model_quality.py"
# also matches the shell that launched this script (its command line carries the heredoc
# text), which deadlocks the wait
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2" > /dev/null; do sleep 20; done

g2=(--token-budget 8192 --min-history-tokens 32768 --dedup-spans --consolidate
    --retrieve-multiplier 2 --provenance-terms 8 --max-spans 128 --recency-fraction 0.6
    --recency-spans 3 --max-span-fraction 0.25)
g2b=(--token-budget 8192 --min-history-tokens 32768 --dedup-spans --consolidate
     --retrieve-multiplier 2 --snippet-spans --recency-spans 3 --max-span-fraction 0.25)

for tf in 0.5 0.35; do
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --max-length 65536 --max-new 96 --max-examples 24 "${g2b[@]}" \
    --compile-mode tail_state --tail-fraction "$tf" \
    --out "artifacts/g2b-patch-localization-tailstate-compiledfar-tf${tf}-b8192-v1.json"
  /root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
    --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
    "${g2[@]}" --compile-mode tail_state --tail-fraction "$tf" \
    --out "artifacts/g2-compiler-tailstate-compiledfar-tf${tf}-b8192-v1.json"
done

# the budget sweep for the G4 link: raw retrieval at the sizes the grid advances on
for b in 4096 2048; do
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
    --retrieve-multiplier 2 --token-budget "$b" \
    --out "artifacts/g2b-patch-localization-raw-b${b}-n48.json"
done

# the killer table's long-history column
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 131072 --max-new 96 --max-examples 24 --token-budget 8192 \
  --min-history-tokens 65536 --dedup-spans --consolidate --snippet-spans \
  --retrieve-multiplier 2 --out artifacts/g2b-patch-localization-compiled-b8192-long-v1.json
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  --min-history-tokens 65536 --dedup-spans --consolidate --retrieve-multiplier 2 \
  --provenance-terms 8 --max-spans 128 --recency-fraction 0.6 --token-budget 8192 \
  --compile-mode tail_state --tail-fraction 0.5 \
  --out artifacts/g2-compiler-tailstate-tf0.5-b8192-long-v1.json
