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

## Best-paper gates, with the measured state

Do not use "Best Paper candidate" internally unless all of the following are measured.  Each row
now carries what was measured, what its scope is, and whether the gate is met.

| gate | threshold | measured | status |
|---|---|---|---|
| 1a view size | <= 8,192 tokens | the shipped view is 8,192 and the *floor* is measured: 4,096 scores -2.40 pp, 6,144 scores 0.00 pp, above that neither metric improves | **met** |
| 1b fidelity | within 2 pp of full history | window 3-7K kept whole: **0.00 pp** (NLL -0.020); plain retrieval at the same budget: -6.94 pp; model-written compaction at the same budget and window: **0.00 pp** (the window carries the surface) | **met** |
| 1c end task | no regression vs full history | 96 paired sessions: the shipped view 0.165 against plain retrieval's 0.169, paired **+0.013, 95% CI [-0.068, +0.092]**; a stricter action-level rescoring is also a bound (every interval includes zero, no arm measurably better) | **met as no-regression**; "better than retrieval" is not claimed |
| 2 G3 law | 128K -> 1M grows <= 1.25x at fixed active demand | active prefill depends only on the active count; lookup grows 2.8x while the corpus grows ~16x; the 1M lookup row is extrapolated and labelled | **met within measured buckets** |
| 3 G3 inversion | 1M/2K cheaper than 32K/16K | 1M with a 4,096-token state: 0.0954 s; with 8,192: 0.2031 s; 32K with the pre-compiler 16,384-token view: 0.4855 s; a session 32x older is 2.4x (5.1x) cheaper, and the footprint estimate ranks the two backwards | **met** |
| 4 G4 consequence | >= 1.5x goodput or >= 30% p99 in several regimes | **134 of 624 cells advance; 52 at the fidelity-admissible 8,192** across balanced, hotspot, slow-worker and worker-loss; balanced/hotspot/slow-worker wins are the no-cluster-KV-store regime, and a strict retrieval-parity bar clears **0** cells | **met with the scope stated** |
| 5 G5 ownership | worker loss transfers no history-sized object | process-level kill: a fresh worker rebuilds 8,192 tokens in 0.91-1.42 s with identical token accuracy and 32,455 bytes read; model rollout keeps the decision on two foreign models (0.137/0.145 against 0.017/0.042) | **met** |
| 6 honest boundary | a region where the baseline wins, measured | balanced load with every session resident against a KV-moving tier does not advance; the 4,096 and 16,384 columns barely advance; the decision advantage vanishes into a tie at 64K-96K histories | **met** |
| 7 deeper: `dM/dL ~ 0` | state flat in age with quality flat | state 7.1-8.2K while history grows 64K -> 156K and on the turn axis 40 -> 96 turns, fidelity delta +0.00/+0.00/+1.43 pp; **and** the lexical evidence mass does grow with age, so the bound is a design choice validated by quality, not an intrinsic ceiling | **met, with the caveat** |

Two comparisons the space will ask for, and their honest state: **compaction** (the de facto
baseline) is equivalent to the shipped view at the same budget and window on both metrics (paired
intervals include zero), so this paper does not claim to beat it; and the *compiler* variant the
project started from is not just unproven but measured worse or tied, which is why it was withdrawn
rather than softened.

The one gate that is *not* met is the compiler gate the project started from: a semantic compiler
does not beat plain retrieval on the decision (+0.055 at n=24, -0.104 on one 48-instance set where
it is worse, and tied at equal budget), and two of its stages measurably hurt.  That claim is
withdrawn rather than softened, and the paper reports the ablation that says why.

## Before submission

Open items, in the order they matter:

1. **Citations for the deployments named in Section 5.**  The two systems compared against (Strata,
   KVMem) are cited from their primary sources; SGLang, vLLM-LMCache, SPIN/SparseServe and llm-d are
   named as deployments, without citations, and need them.
2. **Task success on a benchmark like DeepSWE.**  The end task here is file-level localisation of the
   next turn plus an offline action-level rescoring, which is weaker evidence than task success, and
   Section 6 says so.
3. **The 8B/70B capacity geometries are declared**, from published KV geometries, rather than
   measured on those models; the 0.5B geometry is measured.
4. **The routing consequence is a replay, not a cluster**: the primitives (H2D bandwidth, lookup,
   active-set prefill) are measured; the arrival models and the fleet geometries are declared.
5. **The corpus stops at 96 turns and 156K tokens.**  The 1M rows are composed with one extrapolated
   lookup input and are labelled wherever they appear.
6. **Float numbering.**  Figures and tables are numbered in this draft by the order they were
   generated; in a LaTeX build they are renumbered by order of appearance.  Figure 5 is a
   table-shaped CSV of the capacity numbers and has no drawing.
7. **Configuration coverage.**  Receipts now record the flags that produced them;
   `python benchmarks/check_scripts.py --config-audit` lists the ones that predate that block, and
   re-running those arms is what closes the gap.
