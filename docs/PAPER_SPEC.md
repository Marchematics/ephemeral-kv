# Paper spec — Sessions Without Homes

Working title:

> **Sessions Without Homes: History-Independent Mobility for LLM Agents**

Venue target: OSDI-level systems paper. The project is killed or reframed if the
measured gates in `PLAN.md` do not support the central law.

## One-sentence claim

> Session affinity is not fundamental to long-lived LLM serving; it emerges because
> today's cold-route cost grows with accumulated KV history. If a cold worker
> rematerializes only the next turn's active working set from a durable index, affinity
> pressure can become approximately independent of session age.

## The quantity the paper owns

For request `q` from a session with accumulated history `L`, define the **mobility
tax**

```text
M(q, L) = completion_time(cold_remote(q, L))
        - completion_time(warm_local(q, L))
```

Current history-sized state gives a term shaped like

```text
M_full(L) ~= transfer(KV(L))  or  prefill(L)
```

The EphemeralKV hypothesis is

```text
M_eph(q, L) ~= lookup(q, index(L)) + prefill(W(q))
```

where indexed lookup is sublinear in practice and the active working set `W(q)`
does not scale with total history for the target agent workloads.

The strongest empirical statement is not "EphemeralKV is always faster." It is a
**phase-boundary change**: conventional affinity pressure rises with session age;
EphemeralKV's should rise mainly with current working-set demand.

## Four contribution slots

### C1 — Measure the affinity law

Establish how mobility tax scales with history length, KV bytes/token, fabric
bandwidth, and active working-set size. Show both regions of the phase diagram:
regions where moving full KV is best and regions where rematerialization is best.

The paper loses credibility if it only selects favorable hardware.

### C2 — History-independent mobility primitive

Implement a durable, model-independent transcript/span index and a cold-route path
that materializes only an active text view on the destination. There is no requirement
for a full history-sized KV object to exist in HBM, host DRAM, NVMe, or a remote KV
store for correctness.

The first implementation must work without QCC.

### C3 — Change a cluster scheduling decision

Use existing sticky/cache-aware routing as a strong baseline. Feed the measured
mobility tax into the same locality-vs-queue decision and show that changing the miss
cost expands the operating region where a session should move.

This contribution is the **changed cost model and resulting phase change**, not a novel
routing heuristic.

### C4 — Failover without state ownership

Kill/drain/replace the worker that owns a long-running session. Resume elsewhere with
recovery cost governed by the active working set. Demonstrate the same durable session
substrate across replica failure and, separately, across a model/adapter revision where
old KV is invalid.

Model-agnostic durability is an enabling property, not a novelty claim by itself.

## Main figures to earn

### Figure 1 — Session age versus mobility tax

Log x-axis: 32K -> 1M history.

Curves/dots:
* full KV one-way movement;
* full re-prefill;
* Ephemeral 2K / 4K / 8K / 16K active views.

The result we want to test is a family of nearly horizontal Ephemeral curves and
history-growing full-state curves, with honest crossovers.

### Figure 2 — Mobility phase diagram

Axes:
* KV bytes/token or model family;
* fabric bandwidth;
* active working-set size.

Color/region:
* keep sticky;
* migrate full KV;
* rematerialize active state.

This prevents the paper from degenerating into a single favorable configuration.

### Figure 3 — The inversion

Matched requests:
* older session, smaller active set;
* younger session, larger active set.

Show that session age alone can rank migration cost in the wrong order.

### Figure 4 — Cluster consequence

p99 / SLO goodput versus load skew and session age for:
* strict sticky;
* sticky-until-saturated/cache-aware;
* full-KV migration;
* mobility-cost routing with active rematerialization.

The paper's systems result is visible only if the best policy region changes.

### Figure 5 — Worker failure / rollout

Recovery time and data moved after worker loss, plus the model-revision case where old
KV cannot be reused.

## Baselines that cannot be omitted

* strict session-sticky routing;
* llm-d / sticky-until-saturated style cache-aware routing;
* queue/load-aware routing;
* full re-prefill;
* favorable one-way full-KV movement;
* tiered/distributed KV restoration where available;
* at least one active-working-set / paged-KV system if a reproducible implementation
  is available.

## Non-claims

The paper does **not** claim novelty for:

* "text is the source; KV is a rebuildable artifact";
* bounded/query-dependent active KV by itself;
* soft affinity or sticky-until-saturated routing;
* KV offload/tiering;
* the QCC selection algorithm.

See `NOVELTY.md`.

## Best-paper gates

Do not use "Best Paper candidate" internally unless all of the following are measured:

1. **G2 quality:** a non-QCC compiler keeps primary agent/task quality within 2 points
   of the strongest feasible full-history baseline or a measured fallback recovers the
   gap.
2. **G3 law:** at fixed active demand, 128K -> 1M measured mobility tax grows <=1.25x
   while a full-history cold route grows materially.
3. **G3 inversion:** measured 1M/2K is cheaper to move than 32K/16K.
4. **G4 consequence:** in at least two realistic skew/failure regimes, the changed
   mobility cost yields >=1.5x SLO goodput or >=30% p99 improvement over the strongest
   sticky/cache-aware baseline, with <=5% balanced-load median regression.
5. **G5 ownership:** worker failure does not require transfer/recovery of a durable
   history-sized model-state object.
6. Results include a region where the baseline wins; the phase boundary must be
   measured, not hidden.

## Immediate experiment order

1. G2 public real-trajectory structural run.
2. G2 model/task quality on a non-QCC compiler.
3. G3 matched hardware sweep on one GPU + host/remote transfer path.
4. Only if G2/G3 pass: implement multi-worker G4.
5. G5 after the mobility primitive is real.

No additional cache heuristic is allowed to preempt this order.
