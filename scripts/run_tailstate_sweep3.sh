#!/bin/bash
# tail_state arms, end-task first.
#
# The 8,192-token fidelity receipt is already in: tail_state tf=0.6 is at parity with full
# history (0.00 pp, NLL -0.016) where every arm that cut the tail lost 5-15 pp.  What decides
# the paper is whether the *decision* survives the same view, so the end-task arms come first,
# with the recency arm as the baseline that passes fidelity without a compiler.
#
# The 4,096 and 2,048 rows are the G4 link: routing wants a ~2K active set, and the question is
# whether a view that small can keep the decision once the tail is protected rather than cut.
set -x
cd /root/ephemeral-kv || exit 1
export PYTHONPATH=/root/ephemeral-kv
# match only the interpreter running the harness: a pattern like "g2_model_quality.py"
# also matches the shell that launched this script (its command line carries the heredoc
# text), which deadlocks the wait
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2" > /dev/null; do sleep 20; done

g2=(--token-budget 8192 --min-history-tokens 32768 --dedup-spans --consolidate
    --retrieve-multiplier 2 --provenance-terms 8 --max-spans 128 --recency-fraction 0.6)
g2b=(--token-budget 8192 --min-history-tokens 32768 --dedup-spans --consolidate
     --retrieve-multiplier 2 --snippet-spans)

# smoke the small-budget arms before spending GPU time on them
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 32 --max-examples 3 --token-budget 2048 \
  --min-history-tokens 32768 --dedup-spans --consolidate --snippet-spans \
  --compile-mode tail_state --tail-fraction 0.9 --out /tmp/smoke-tail-2k.json

# 1. does the arm that passes fidelity with no compiler also pass the decision?
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 "${g2b[@]}" --compile-mode recency \
  --out artifacts/g2b-patch-localization-recency-b8192-v1.json

# 2. the money number: both halves at 8K
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 "${g2b[@]}" \
  --compile-mode tail_state --tail-fraction 0.6 \
  --out artifacts/g2b-patch-localization-tailstate-tf0.6-b8192-v1.json

# 3. the trade-off: more far field, same budget
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  "${g2[@]}" --compile-mode tail_state --tail-fraction 0.4 \
  --out artifacts/g2-compiler-tailstate-tf0.4-b8192-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 "${g2b[@]}" \
  --compile-mode tail_state --tail-fraction 0.4 \
  --out artifacts/g2b-patch-localization-tailstate-tf0.4-b8192-v1.json

# 4. the G4 link: can a 4K and a 2K active set keep the decision?
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 --min-history-tokens 32768 --dedup-spans \
  --consolidate --retrieve-multiplier 2 --snippet-spans --token-budget 4096 \
  --compile-mode tail_state --tail-fraction 0.75 \
  --out artifacts/g2b-patch-localization-tailstate-tf0.75-b4096-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 --min-history-tokens 32768 --dedup-spans \
  --consolidate --retrieve-multiplier 2 --snippet-spans --token-budget 2048 \
  --compile-mode tail_state --tail-fraction 0.9 \
  --out artifacts/g2b-patch-localization-tailstate-tf0.9-b2048-v1.json

/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  --min-history-tokens 32768 --dedup-spans --consolidate --retrieve-multiplier 2 \
  --token-budget 4096 --compile-mode tail_state --tail-fraction 0.75 \
  --out artifacts/g2-compiler-tailstate-tf0.75-b4096-v1.json
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  --min-history-tokens 32768 --dedup-spans --consolidate --retrieve-multiplier 2 \
  --token-budget 2048 --compile-mode tail_state --tail-fraction 0.9 \
  --out artifacts/g2-compiler-tailstate-tf0.9-b2048-v1.json
