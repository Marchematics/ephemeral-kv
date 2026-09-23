# EphemeralKV

**A long-lived LLM session should have an identity, not a home.**

Agent-serving stacks increasingly make a session *sticky*: a follow-up turn is routed
back to the worker that already owns its KV prefix because moving or rebuilding a
history-sized KV cache is expensive. EphemeralKV tests a different systems abstraction:

> The transcript and its compact retrieval index are durable. Per-worker KV is an
> opportunistic execution cache, not session ownership. A turn may move when the
> queueing benefit exceeds the cost of rematerializing its active working set.

The core hypothesis is stronger than "KV can be evicted." It is that **session affinity
is a consequence of history-sized migration cost, not a fundamental property of agent
serving**. If a cold route costs `index_lookup + prefill(active_set)` instead of
`move_or_recompute(full_history)`, session age no longer determines where the next
turn may run.

## What this project must establish

1. **History-free mobility.** With the active working set held fixed, measured remote
   materialization cost stays roughly flat as history grows from 32K toward 1M tokens,
   while full-KV movement/re-prefill grows with history.
2. **A routing phase change.** Under realistic load skew, tool gaps, or worker failure,\n   replacing a history-sized cold-route penalty with an active-set-sized penalty changes\n   when a scheduler should leave the warm worker. The routing heuristic itself is not\n   claimed as novel.
3. **The inversion.** A 1M-token session with a small active set can be cheaper to move
   than a 32K-token session with a larger active set.
4. **No hidden QCC dependency.** BM25/embedding/provenance-style compilers are
   first-class backends. QCC may be evaluated as one optional backend but is never
   required for the claim.

## What is *not* the novelty

The first scaffold used the slogan "the conversation is durable; the KV cache is
disposable." The literature/implementation audit in
[`docs/NOVELTY.md`](docs/NOVELTY.md) found that this is not sufficient: regenerable-KV
designs already treat text as source and KV as a derived artifact, and KVMem /
sparse-attention systems already keep bounded active KV working sets.

Those observations remain useful enabling mechanisms. The paper claim is now about
**breaking hard session affinity by bounding the cost of a cold route**.

## Relationship to QCC

This is deliberately separate from
[qcc-transformer](https://github.com/Marchematics/qcc-transformer).

* QCC asks: **how much live state does one query need?**
* EphemeralKV asks: **when should a long-lived session be free to move between workers?**

Paper A's context compiler is not copied here. EphemeralKV owns a routing/lifecycle
problem, different workloads (multi-turn agents), and different primary metrics
(p99/TTFT/SLO goodput/failover rather than single-request quality-state curves).

## Current gate signal

G1's accounting model produced a useful negative result: with its original
history-scanning index assumption, `discard + recompile` did **not** beat DRAM/NVMe KV
restore by 512 turns. So tiered-storage latency is not the headline.

G3 now asks the sharper question: if the durable index supports bounded/sublinear
lookup, does the **remote-route tax** stop scaling with history? The checked-in
accounting artifacts define the phase boundary across active-set size, KV bytes/token,\nand fabric bandwidth; they are not hardware results.

## Repository layout

```text
benchmarks/   kill gates; each emits a machine-readable JSON artifact
artifacts/    checked-in gate receipts; accounting is labelled as accounting
docs/        PLAN.md, CLAIMS.md, NOVELTY.md
tests/       CPU tests; no GPU/downloads
```

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q

python benchmarks/kill_gate_crossover.py \
  --out artifacts/g1-accounting.json

python benchmarks/kill_gate_mobility.py \
  --out artifacts/g3-mobility-accounting.json
```
