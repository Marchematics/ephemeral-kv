# Novelty ledger

This file records claims we deliberately **will not** make after the September 2026
overlap audit. The goal is to keep EphemeralKV from drifting into an already occupied
KV-cache or routing story.

## Rejected headline 1 — "text is the source; KV is a build artifact"

A public `fak` design document already states essentially this abstraction and develops
text-backed KV regeneration across model/tokenizer/adapter/precision changes:

* `anthony-chaudhary/fak`, `docs/serving/regenerable-kv-plan.md`
* headline: *"Regenerable KV — the text is the source, the cache is a build artifact"*

Therefore model-agnostic transcript durability is an **enabling property**, not the
paper's novelty.

## Rejected headline 2 — "GPU memory tracks an active KV working set"

This space is also occupied.

* **KVMem** (arXiv:2609.04852) virtualizes million-token agent workspaces with paged KV
  across GPU/host/NVMe and materializes a query-dependent bounded GPU view.
* Sparse-attention serving systems such as **SPIN/SparseServe** already reason about
  active working sets and per-request HBM budgets.

Therefore "1M history, small GPU KV" and the footprint inversion are useful results, but
not sufficient as a paper identity.

## Rejected headline 3 — "make migration possible by storing all KV elsewhere"

That is still a history-sized state model. Tiered caches, distributed KV stores, and
cache-aware routers already do this. It preserves session affinity as a performance
constraint because a cold route must move a large prefix.

## Rejected headline 4 — "soft affinity routing"

Soft affinity is not new either. llm-d's 2026 "sticky until saturated" routing already
prefers the KV-warm endpoint and escapes to load-based placement after a calibrated
saturation point; cost-aware LLM schedulers likewise trade locality against queueing.

EphemeralKV may *use* a soft-affinity scheduler in evaluation, but the scheduler policy
is not the core contribution.

## Surviving hypothesis — affinity pressure should not grow with session age

Current agent serving has a structural reason to become increasingly sticky: the cold
route penalty grows with the history-sized state that must be moved or recomputed.

Let `L` be accumulated history and `W(q)` the active state needed by the next turn.

A conventional cold route has a cost shaped like

```text
M_full(L) = transfer_or_reprefill(history-sized state)
```

EphemeralKV targets

```text
M_eph(q) = indexed_lookup(q) + materialize(W(q))
```

with both terms bounded/sublinear in `L` on real agent traces.

For a warm worker `a` and cold worker `b`, migration is worthwhile when

```text
queue_delay(a) - queue_delay(b) > mobility_tax
```

The potential new systems result is therefore a **phase-boundary change**: in today's
systems the queue imbalance required to escape affinity increases with session age; in
EphemeralKV it should depend primarily on current working-set demand.

> **A session has an identity, not a home.** Local KV is an opportunistic accelerator,
> while the cost of choosing another worker is working-set-bounded rather than
> history-sized.

The paper succeeds only if this altered cost law is measured and changes cluster-level
p99/SLO-goodput decisions. Memory reduction alone is insufficient.

## Counterintuitive results worth pursuing

1. **More history, same mobility tax.** 32K -> 1M history at fixed active state should
   barely change cold-route cost.
2. **Longer can be cheaper to move.** A 1M-history / 2K-active session can have lower
   remote cost than a 32K-history / 16K-active session.
3. **Session age stops increasing stickiness.** A million-token session should remain
   movable under the same queue-gap threshold as a much younger session with the same
   active state.
4. **Changing the miss cost can reverse the router result.** Existing sticky/cache-aware
   policies are baselines; EphemeralKV must expand the operating region where routing
   away from locality improves p99 or SLO goodput.
5. **Failure does not own the session.** Killing a worker should lose an optimization,
   not durable session state; recovery should be active-set-sized.

## Independence from QCC

No main EphemeralKV result may require a QCC selector. The first prototype ships a
model-independent durable lexical/provenance index in `ephemeralkv/index.py`. An
embedding compiler is the next independent backend. If QCC is later plugged in, it is
an additional backend row only.
