#!/bin/bash
# The killer table, extended past the corpus: does the state that must move grow with the session?
#
# The longest real session in the corpus is ~156K tokens, so every point of the mobility law beyond
# that has been composed from measured primitives rather than measured.  This runner measures it:
# `build_composed_long_sessions.py` concatenates whole real sessions (same repository wherever the
# corpus allows) into 145K / 278K / 545K / 1.07M-token histories - 153 / 302 / 512 / 1321 turns -
# keeping the *final* turn's patch as the end task's ground truth, and the decision harness is then
# run on each bucket at the same 8,192-token state.
#
# What comes out is the two numbers the claim needs at each length: the size of the state that has
# to move (`active_tokens`) and whether the decision holds (`active.f1`).  The full-history arm in
# these receipts is *not* full history - the model's window is 131,072 tokens - so it is read as
# "the newest 131K" and is not used as a fidelity reference.
#
# The composition is labelled in the corpus itself (`composed_from`, `composed_same_repo`,
# `composed_target_files`) and in every receipt name, because a composed session is not a claim
# that real sessions are this long.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2" > /dev/null; do sleep 20; done

/root/qcc/venv/bin/python benchmarks/build_composed_long_sessions.py
/root/qcc/venv/bin/python benchmarks/build_composed_long_sessions.py --verify

for bucket in 128k 256k 512k 1m; do
  /root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py \
    --jsonl "data/sessions-composed-${bucket}.jsonl" \
    --model /root/qcc/models/Llama-3.2-1B-Instruct --max-length 131072 --max-new 96 \
    --max-examples 12 --min-history-tokens 32768 --token-budget 8192 \
    --dedup-spans --consolidate --snippet-spans --retrieve-multiplier 2 \
    --compile-mode tail_state --tail-tokens 4096 \
    --out "artifacts/g2b-patch-localization-composed-${bucket}-b8192-v1.json"
  /root/qcc/venv/bin/python benchmarks/g2_dead_state.py \
    --jsonl "data/sessions-composed-${bucket}.jsonl" \
    --out "artifacts/g2-dead-state-composed-${bucket}-v1.json"
done
