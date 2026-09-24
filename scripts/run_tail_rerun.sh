#!/bin/bash
# Re-run of every tail arm affected by the `text` rebinding bug.
#
# The oversized-span truncation added for the "the newest span is not optional" fix rebound the
# loop's `text` variable, which is both the example target and the next span appended to the
# durable index.  The unit test now pins it (test_tail_state.py::
# test_truncating_the_newest_span_does_not_corrupt_the_target_or_the_index), and the two
# artifacts produced with the bug (g2-compiler-tailstate-tf0.6-b8192-v2.json and
# g2b-patch-localization-tailstate-tf0.6-b8192-v2.json) are superseded, not reported.
#
# Order is by what decides the paper: both halves of the 8K claim first, then the arm that
# spends on both (tail_query), then the small-active-set rows that G4's 4,096 column needs.
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

# 1. the 8K claim, both halves, on the same view
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 "${g2b[@]}" \
  --compile-mode tail_state --tail-fraction 0.6 \
  --out artifacts/g2b-patch-localization-tailstate-tf0.6-b8192-v3.json
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  "${g2[@]}" --compile-mode tail_state --tail-fraction 0.6 \
  --out artifacts/g2-compiler-tailstate-tf0.6-b8192-v3.json

# 2. the arm that spends on both: newest span verbatim, everything else compiled
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 "${g2b[@]}" \
  --compile-mode tail_query --tail-cap 0.5 \
  --out artifacts/g2b-patch-localization-tailquery-cap0.5-b8192-v1.json
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  "${g2[@]}" --compile-mode tail_query --tail-cap 0.5 \
  --out artifacts/g2-compiler-tailquery-cap0.5-b8192-v1.json

# 3. the same arm with a smaller share for the newest span: more evidence, less surface
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 "${g2b[@]}" \
  --compile-mode tail_query --tail-cap 0.25 \
  --out artifacts/g2b-patch-localization-tailquery-cap0.25-b8192-v1.json
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  "${g2[@]}" --compile-mode tail_query --tail-cap 0.25 \
  --out artifacts/g2-compiler-tailquery-cap0.25-b8192-v1.json

# 4. the G4 link: 4,096 and 2,048 active tokens with the newest span protected
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 --min-history-tokens 32768 --dedup-spans \
  --consolidate --retrieve-multiplier 2 --snippet-spans --token-budget 4096 \
  --compile-mode tail_query --tail-cap 0.75 \
  --out artifacts/g2b-patch-localization-tailquery-cap0.75-b4096-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 --min-history-tokens 32768 --dedup-spans \
  --consolidate --retrieve-multiplier 2 --snippet-spans --token-budget 2048 \
  --compile-mode tail_query --tail-cap 0.9 \
  --out artifacts/g2b-patch-localization-tailquery-cap0.9-b2048-v1.json
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  --min-history-tokens 32768 --dedup-spans --consolidate --retrieve-multiplier 2 \
  --token-budget 4096 --compile-mode tail_query --tail-cap 0.75 \
  --out artifacts/g2-compiler-tailquery-cap0.75-b4096-v1.json
/root/qcc/venv/bin/python benchmarks/g2_model_quality.py --jsonl data/sessions-64k.jsonl \
  --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-examples 48 \
  --min-history-tokens 32768 --dedup-spans --consolidate --retrieve-multiplier 2 \
  --token-budget 2048 --compile-mode tail_query --tail-cap 0.9 \
  --out artifacts/g2-compiler-tailquery-cap0.9-b2048-v1.json
