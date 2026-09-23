# Novelty ledger

This file records claims we deliberately **will not** make after the September 2026
overlap audit. The goal is to keep EphemeralKV from drifting into an already occupied
KV-cache story.

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

## Surviving hypothesis — session affinity is not fundamental

Current agent serving explicitly rewards locality:

* vLLM AgentX reported session-aware sticky routing beating queue/KV-balanced routing
  because a moved turn pays KV transfer and consumes destination cache capacity.
* llm-d added session-affinity routing so follow-up turns return to the same backend.
* agent-serving simulators and schedulers model a program/session-to-instance affinity.

EphemeralKV targets a different operating regime:

> **A session has an identity, not a home.** KV locality is an opportunistic hint. A
> cold worker may rebuild only the active working set, so the migration tax is bounded
> by current demand instead of accumulated history.

The paper is successful only if this changes the *optimal scheduling decision* in real
measurements. Merely reducing memory is insufficient.

## New systems abstraction — soft affinity

A strict sticky system effectively assigns one worker per session. EphemeralKV treats
that location as a cache hint and chooses the worker minimizing:

```text
queue_delay(i) + (0 if warm(i) else predicted_mobility_tax(request, i))
```

The research claim is that the mobility tax can be made **working-set-bounded**. If so,
a longer history does not monotonically increase affinity, and queueing/failure can
dominate locality.

## Counterintuitive results worth pursuing

1. **More history, same mobility tax.** 32K -> 1M history at fixed active state should
   barely change the cost of moving the next turn.
2. **Longer can be cheaper to move.** A 1M-history / 2K-active session can have lower
   remote cost than a 32K-history / 16K-active session.
3. **Balance can beat locality for agent sessions.** This intentionally targets the
   opposite result from today's sticky-routing observation, after changing the cost
   model that made stickiness optimal.
4. **Failure does not own the session.** Killing a worker should lose an optimization,
   not durable session state; recovery should be active-set-sized.

## Independence from QCC

No main EphemeralKV result may require a QCC selector. The first real prototype must
ship with a non-QCC text/index compiler. If QCC is later plugged in, it is an additional
backend row only.
