#!/bin/bash
# The decision arms figure 3 is drawn from, and the equal-budget references around them.
#
# These are the receipts `benchmarks/make_figures.py` reads to build the compiler ablation ladder
# (figure 3) and the budget references: without them a clone cannot regenerate its own figures,
# which is how they were found - the receipt audit, extended to treat the figure generator as a
# citing document, reported them as cited but untracked.  Their invocations lived in the shell
# history rather than in a script, like the compaction baseline's window before it.
#
# Each command below is reconstructed from the `config` block the receipt itself records (the
# harness writes `vars(args)` into the payload), so the flags are the ones that produced the
# artifact and not a reconstruction from memory.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

# decision arms at 8,192 (state-first, protect-whole, window + compiler at a 4,096 window)
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 --min-history-tokens 32768 \
  --token-budget 8192 --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
  --compile-mode state_first \
  --out artifacts/g2b-patch-localization-statefirst-b8192-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 24 --min-history-tokens 32768 \
  --token-budget 8192 --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
  --compile-mode consolidate --recency-spans 3 --max-span-fraction 1.0 \
  --out artifacts/g2b-patch-localization-protectwhole-b8192-v1.json
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
  --token-budget 8192 --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
  --compile-mode tail_state --tail-fraction 0.5 \
  --out artifacts/g2b-patch-localization-windowcompiler-b8192-n48.json

# the equal-budget references: consolidated retrieval at 8,192 and 12,288, and raw retrieval at
# 4,096 (the last is the arm the shipped 4,096-window view is compared against)
for b in 8192 12288; do
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
    --token-budget "$b" --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
    --compile-mode consolidate \
    --out "artifacts/g2b-patch-localization-compiled-b${b}-n48.json"
done
/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
  --jsonl data/sessions-patch-long.jsonl --model /root/qcc/models/Llama-3.2-1B-Instruct \
  --max-length 65536 --max-new 96 --max-examples 48 --min-history-tokens 32768 \
  --token-budget 4096 --retrieve-multiplier 2 \
  --out artifacts/g2b-patch-localization-raw-b4096-n48.json

# The two fidelity-side inputs of figure 3 predate the `config` block, so their flags cannot be
# recovered from the receipt the way the commands above were; the arms they belong to are in
# `run_tail_rerun.sh` (protect-whole) and `run_window_plus_compiler.sh` (compiled far field at
# tf 0.5), which is where a re-run should be taken from:
#   artifacts/g2-compiler-protectwhole-b8192-v1.json
#   artifacts/g2-compiler-tailstate-compiledfar-tf0.5-b8192-v1.json
