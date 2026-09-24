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

## Main figures to earn, and the receipts that already populate them

### Figure 1 - The session grows, the state that must move does not

The paper's identity figure.  x: raw history (32K -> 1M, log).  Two families: history-sized state
(full KV, full re-prefill) rising with age, and the measured execution state flat at 7-8K.
Inset: the fidelity delta by history bucket (+0.00 / +0.00 / +1.43 pp at 64K / 83K / 156K).
Receipts: `g2-killer-table-v3.json`, `g3-*`.

### Figure 2 - Two metrics, two views (the frontier)

One scatter at 8,192 tokens: fidelity delta (x) against end-task F1 (y) for every arm measured -
recency, the window fractions, plain retrieval, the compiler, log replay, state-first, the
protected-whole variants.  It shows the trade-off is structural, and it is the figure that
explains why the compiler is not in the story:
`recency +0.29 pp / 0.154`, `window+compiler 0.00 pp / 0.161`, `compiled -5.44 pp / 0.292 (n=24)`
and `0.139 (n=48)`, `raw -6.94 pp / 0.243`, `protected-whole -20.33 pp / 0.211`,
`newest-only -25.36 pp / 0.083`.
Receipts: every `g2-compiler-*` and `g2b-patch-localization-*` artifact.

### Figure 3 - The compiler ablation, including what does not pay

Fidelity and decision per compiler stage, with the negative results kept: supersede-by-identity
pays 8.5 pp of fidelity over path collapse, snippet selection costs 20 pp, log replay buys nothing
because only 6% of retrieved spans are file events, and plain retrieval beats the whole pipeline
(0.243 against 0.139 paired on 48 sessions).
Receipts: `g2-compiler-consolidate-*`, `-materialize-*`, `-statefirst-*`, `-protectwhole-*`,
`g2b-patch-localization-raw-b4096-n48.json`, `-compiled-b8192-n48.json`.

### Figure 4 - The phase diagram with quality attached

Advancing cells by active set, with the quality criteria shown rather than assumed: 17/66 at
2,048, 63/198 at 4,096, **36/246 at the fidelity-admissible 8,192**, 2/66 at 16,384; and the same
cells annotated by which criterion they clear (fidelity within 2 pp, decision at least full
history, decision at least raw retrieval).
Receipts: `g4-quality-join-v1.json`.

### Figure 5 - Capacity and recovery

Sessions per worker under the two resource models (13.3 / 1.2 / 0.5 histories against 213 / 20 / 8
states at 128K on the 0.5B / 8B / 70B geometries), and recovery after a worker loss or a model
revision: index rebuild 0.55-1.04 s against 6.31 s of re-prefill, ~33 KiB of state text against
12-128 GiB of KV, end-task F1 0.137/0.145 against 0.017/0.042 on two models that never saw the
sessions.
Receipts: `g5-capacity-planning-v1.json`, `g5-failover-*.json`, `g3-*`.

### Figure 6 - The metric lesson

Teacher-forced next-token fidelity against the amount of the newest evidence kept verbatim: at
parity (+0.29 pp) with 8,192 tokens of recency, at parity with 2.9K of window inside an 8K view,
and 5-25 pp down when the newest evidence is truncated, reordered or replaced.  This is the figure
that says why the field's proxy metric cannot compare systems here, and it is a contribution in
its own right.
Receipts: `g2-compiler-recency-*`, `-tailstate-*`, `-tailquery-*`, `-windowcompiler-*`.

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

## The measurement that disqualified a metric

The gate was written as "keep primary agent/task quality within 2 points of full history". The
measurements say that which metric you pick decides the answer, so the paper has to name one and
show why. Every arm at an 8,192-token view, both metrics on the same examples:

| view (8,192 tokens) | teacher-forced fidelity | end-task F1 (patch files) |
|---|---:|---:|
| full history | reference | 0.045 |
| plain recency (newest spans whole) | **+0.29 pp** | 0.154 |
| newest span whole + compiled far field | 0.00 pp | 0.104 |
| raw lexical (dedup + snippet) | -6.94 pp | 0.237 |
| evidence consolidation, no state collapse | -5.44 pp | **0.292** |
| log replay into materialised state | -7.21 pp | 0.231 |
| 16,384-token reference view | -1.94 pp | 0.142 |

Teacher-forced next-turn fidelity is **saturated by recency**: 8,192 tokens of the newest spans -
9.7% of a 32K-128K history - sit at parity with the full transcript, so no compiler can separate
itself on it, and a paper whose headline is "we halve the budget at equal fidelity" would be
claiming what one line of truncation already does. The end task is not saturated: it is carried
by query-focused evidence (0.292 against 0.154 for recency), it *degrades* when budget is spent
on generic recency padding (0.104), and full history is the worst arm of all (0.045, because the
evidence is buried in 32K tokens of transcript).

So **the end task is the primary quality metric and fidelity is a constraint the view must also
satisfy.** Both are reported for every arm; the frontier between them is a result rather than an
embarrassment, because it is what makes the compiler's contribution visible at all.

## Best-paper gates

Do not use "Best Paper candidate" internally unless all of the following are measured:

1. **G2 quality:** the compiled view is at most 8,192 tokens and (a) is within 2 points of full
   history on teacher-forced next-turn fidelity *and* (b) does not regress the end task against
   full history, with the end task named as the primary metric (next section). Measured so far:
   (a) holds for tail-dominated views (recency +0.29 pp, newest-span-whole 0.00 pp) and (b) holds
   for query-focused compiled views (F1 0.292 against 0.045); one view holding both is what
   `--compile-mode tail_query` tests.
2. **G3 law:** at fixed active demand, 128K -> 1M measured mobility tax grows <=1.25x
   while a full-history cold route grows materially.
3. **G3 inversion:** measured 1M/2K is cheaper to move than 32K/16K.
4. **G4 consequence:** in at least two realistic skew/failure regimes, the changed
   mobility cost yields >=1.5x SLO goodput or >=30% p99 improvement over the strongest
   sticky/cache-aware baseline, with <=5% balanced-load median regression. Measured: **118 of 576
   cells across three grids, of which 36 sit at the 8,192-token active set where both quality
   metrics hold, in all three regimes** (balanced 5, slow worker 6, worker loss 25). The three
   axes the first grid lacked: active sets between 2,048 and 16,384, sessions beyond 262K, and the
   model geometry (a 128K session is a 16 GiB transfer on an 8B-class model and 40 GiB on a
   70B-class one, against 0.203 s to rematerialise 8,192 tokens). Scope stated with the number:
   the balanced and slow-worker wins are the no-cluster-KV-store regime (baseline re-prefills
   1.6-14.3 s against 0.203 s); against a KV-moving tier the wins remain worker loss plus the 1M
   mixes, and the 4,096 column is not fidelity-admissible (-25 pp).
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
