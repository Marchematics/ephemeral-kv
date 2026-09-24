# Sessions Without Homes

**What must move in a long-lived LLM session**

Draft manuscript.  Every number here is a receipt in `artifacts/`; `CLAIMS.md` carries the
per-claim provenance, `VERDICT.md` the claim/limit ledger, `PLAN.md` the gate structure.  Numbers
that are derived rather than measured are labelled; numbers whose scope is narrower than the
sentence around them carry the scope.

---

## Abstract

A long-lived LLM session is served today as if its KV cache *were* the session: the resident object
grows with the transcript, so placement, recovery and capacity planning all inherit the session's
age.  We measure what a turn actually needs and find that the object those decisions depend on is
bounded and independent of age.  On real coding-agent traces, an 8,192-token execution state holds
the next turn at a teacher-forced fidelity delta of **0.00 pp** against the full transcript, and its
end-task score is statistically indistinguishable from the best retrieval baseline we can build,
while the state stays at **7-8K tokens** as the raw history grows from 64K to 156K (and, on the turn
axis, from 40 to 96 turns).  The consequence is a phase change rather than a speedup: a session
**32x older costs 2.4x less to move** (5.1x at a 4K state) because the transfer term leaves the cold
path; one worker holds **16x more sessions**; recovery and a model revision rebuild the state from a
durable index instead of moving 12-128 GiB of KV; and a replay against measured hardware primitives
advances routing in **52 cells at a quality-admissible state across balanced, hotspot and worker-loss
regimes**, where the original grid advanced in four cells at a 2K corner that no quality measurement
supported.

We also report what does not work, because it is the hypothesis this space reaches for first.  A
semantic compiler - content dedup, collapse-each-file-to-its-latest-state, snippet re-selection, log
replay into materialised file state - does **not** beat plain retrieval on the real task, and two of
its stages measurably hurt: collapsing a file to its latest state costs 8.5 pp of fidelity against
keeping the superseded views, and re-selecting the newest output's lines costs 20 pp.  What carries
quality is weaker and more useful: render the newest evidence **whole** (a ~3K window suffices) and
spend the rest of the budget on retrieval with a consolidated far field.  Finally, the metric the
field uses to compare long-context systems - teacher-forced next-token fidelity - cannot see any of
this: on these traces it is saturated by keeping the newest spans whole, which is why the end task is
the primary metric and fidelity is a constraint.

---

## 1. Introduction

Serving stacks conflate three things that a long-lived agent session keeps separate:

```text
durable history     the transcript, append-only, grows without bound
execution state     what the model needs for the next action
local KV            a device-local artifact of having computed the state once
```

Because they are one object today, the resource model of a session is its footprint:
`session cost ~ history / KV footprint`.  Tiered context caches move that object between memory
levels, retention policies decide how long to keep it, and workspace virtualisation maps it to host
or NVMe and materialises a query-dependent view.  All of that work is coherent *within* the
conflation.  This paper measures the conflation itself.

Three measurements break it.  First, a coding-agent transcript is not a long sequence of comparable
turns: **five spans hold 73-96% of a long session's tokens**, and the turn-by-turn content a view
actually walks is 4-27% (median 8%) - so a system whose cost is the footprint is retaining and moving
state that was needed once.  Second, what a turn needs is bounded: across raw histories of 64K, 83K
and 156K tokens the compiled state sits at 7.1-8.2K and the fidelity delta does not move
(+0.00 / +0.00 / +1.43 pp).  Third, because of that bound the cold-route cost stops tracking age: a
1M-token session with a 4,096-token state costs 0.0954 s to place while a 32K session carrying the
16,384-token view that fidelity needed before this bound costs 0.4855 s - the *older* session is
cheaper, and a footprint-based estimate ranks the two exactly backwards (5.899 s against 0.184 s).

The abstraction that follows is:

```text
durable session   = history + index          (append-only, model-independent, on the tier)
execution state   = compile(history, q)      (bounded, rebuilt on demand, never transferred)
local KV          = a disposable artifact    (dropped on eviction, rebuilt in 0.05-1.4 s)
```

and the paper's contributions are its measured consequences:

1. **A bounded execution state holds the turn.**  8,192 tokens, fidelity at parity with the full
   transcript, end-task score statistically tied with the best retrieval baseline measured.
2. **The state does not grow with the session.**  `dM/dL ~ 0` on both the fidelity and the state-size
   axes, measured across a 2.4x growth in history and on the turn axis.
3. **Session age stops predicting placement cost.**  The inversion, and the ranking reversal that
   makes a footprint-based scheduler prefer exactly the wrong session.
4. **The consequences are systemic**: 16x sessions per worker, lossless recovery on a fresh process
   (0.91-1.42 s, no KV transfer), and a routing phase change at the quality-admissible state size in
   every regime we model, including a 10x-slow hotspot.
5. **A negative result with an exact attribution.**  The obvious way to shrink the state - a semantic
   compiler - does not pay here: it is tied or worse on the decision, and the two stages that
   actually "compile state" are the ones that hurt.  The window and retrieval are what carry quality.
6. **A metric lesson.**  Teacher-forced next-token fidelity is saturated by the newest evidence on
   these traces and therefore cannot compare systems; the end task can, and we report both for every
   arm.

---

## 2. What a turn needs

### 2.1 Most of a session is state that is never needed again

Measuring the token share held by the largest spans of each long session (`g2_dead_state.py`):

| session tokens | spans | largest span | top-2 share | top-5 share | the rest |
|---:|---:|---:|---:|---:|---:|
| 156,207 | 45 | 45,051 | 58% | 93% | 7% |
| 155,997 | 61 | 45,416 | 58% | 94% | 6% |
| 123,210 | 29 | 55,320 | 90% | 96% | 4% |
| 141,168 | 65 | 56,514 | 80% | 92% | 8% |
| 127,126 | 83 | 28,261 | 44% | 73% | 27% |

This is the arithmetic behind the abstraction: the footprint is dominated by blocks that were
needed once (a task prompt carrying the workspace, a whole-file dump, a long plan), and the
turn-by-turn content is a minority.

### 2.2 Two laws, and the view that satisfies both

Every arm measured at an 8,192-token view on the same examples (fidelity n=44 long-history
examples, decision n=48 paired sessions, 26 scoreable):

| view | teacher-forced fidelity | end-task F1 |
|---|---:|---:|
| full history | reference | 0.089 |
| plain recency (newest spans whole, nothing else) | **+0.29 pp** | 0.154 |
| **window 3-4K + consolidated far field** | **0.00 pp** | **0.161** |
| window 3K + raw retrieval far field | -6.94 pp | 0.191 |
| raw retrieval, no window | -6.94 pp | 0.191 (4,096: 0.243) |
| evidence consolidation, no window | -5.44 pp | 0.139 |
| log replay into materialised state | -7.21 pp | 0.231 |
| protected spans kept whole + ranked rest | -20.33 pp | 0.211 |
| newest span only + compiled far field | -25.36 pp | 0.083 |

1. **Fidelity is set by how the newest evidence is rendered.**  Kept whole it is at parity with as
   little as 3K of window; prefix-truncated it costs 5-7 pp; reordered into a query-selected excerpt
   20 pp; replaced by a single span 25 pp.  The *far field* must also be consolidated rather than
   prefix-truncated raw dumps, or the surface drops ~6-7 pp even with the window whole - at both
   8,192 and 12,288 tokens, so it is a treatment requirement, not a budget one.
2. **The decision is set by which evidence is retrieved.**  In the end-task configuration the
   window-plus-retrieval view and plain retrieval at the same budget produce byte-identical contexts
   on all 48 instances, and every bounded retrieval arm is statistically tied: window+raw 0.191,
   plain retrieval 0.191 at 8,192 and 0.243 at 4,096 (paired +0.081, 95% CI [-0.016, +0.177], 13 wins
   / 3 losses), dedup-only 0.207-0.239, window+compiled 0.161 (+0.029 against window+raw, CI
   [-0.089, +0.146]).  More budget does not help: the same arm scores 0.191 at 8,192 and 0.173 at
   12,288, so above ~4K the constraint is retrieval coverage, not budget.

So the two requirements do not compete for content.  They compete for *rendering*: render the newest
evidence whole, consolidate the far field, and retrieve the rest.  That is the view the system
ships, and it is the configuration in bold above.

### 2.3 The metric cannot see the difference

Keep the newest spans whole and teacher-forced fidelity is at parity with **8,192 tokens of
recency** - 9.7% of a 32K-128K history.  A metric that is satisfied by the most naive possible
policy cannot compare systems, and it explains why the compiler literature's proxy would report
"compression works" for a policy that merely truncates.  We therefore gate on the end task (file
level localisation of the next turn against the recorded patch) and report fidelity as a constraint.

---

## 3. The system

### 3.1 The durable object

The transcript and a model-independent lexical/identifier index over it (`DurableSpanIndex`):
append-only, ~0.09-0.25 ms p50 per lookup for histories up to 128K, with kind postings for
structural selection and content fingerprints for identity.  Nothing in it is model-specific, which
is what makes a model revision a recovery operation rather than a migration.

### 3.2 The compiler, and what its stages are worth

| stage | semantics | measured effect |
|---|---|---|
| recency window | the current turn is not optional; newest evidence admitted whole | fidelity -6.94 -> **0.00 pp** |
| content-identity dedup | one copy of repeated tool output and repeated reasoning | neutral: 0.239 against 0.243 (paired -0.004, CI [-0.024, +0.016]) |
| supersede by state identity, no path collapse | a file's or command's newest state stands for its earlier views, replaced turns kept as provenance | fidelity -6.94 -> -5.44 pp; decision 0.237 -> 0.292 (n=24) |
| collapse each path to its latest state | (the ablation) | **-13.90 pp**, i.e. 8.5 pp worse |
| snippet re-selection of an oversized newest span | keep the query-relevant lines | **-20 pp** on fidelity |
| log replay into materialised state | dumps set content, diffs and SEARCH/REPLACE apply, repeated commands collapse | 0.231 against 0.237 on the decision; neutral, because only **6%** of retrieved spans are file events |
| identifier provenance | pull spans sharing paths/ids with the query | part of the 4K -> 2K improvement |

Two of these are negative results the paper keeps, because they bound what "executable state" can
mean on this workload: the stages that *compile state* are the ones that hurt, and the reason is a
property of coding-agent traces (most retrieved evidence is not file events) rather than of the
compiler.

### 3.3 What the runtime does

A cold route is: look up the query in the durable index (sub-millisecond), compile a bounded view
(window + consolidated far field), prefill it on the destination, serve.  Nothing history-sized is
transferred and no worker owns the session; the KV a worker holds is a cache of a computation it can
repeat.

---

## 4. Evaluation

Setup: a single shared 24 GiB A10G; real agent traces (SWE-smith / SWE-rebench style trajectories,
2,736 long sessions, histories to 156K tokens); Llama-3.2-1B-Instruct for quality, Qwen2.5-0.5B/1.5B
for primitives and cross-model checks.  All comparisons are paired on the same instances; intervals
are paired bootstrap (20k resamples).

### 4.1 The state is bounded, and quality does not decay with age

| raw history p50 | execution state p50 | fidelity delta |
|---:|---:|---:|
| 64,401 | 7,133-7,829 | +0.00 pp |
| 82,972 | 7,133-7,899 | +0.00 pp |
| 155,574 | 8,203 | +1.43 pp |

| turns p50 | raw history p50 | execution state p50 | fidelity delta |
|---:|---:|---:|---:|
| 40 | 79,769 | 8,121 | -1.43 pp |
| 58 | 84,617 | 7,887 | **+0.00 pp** |
| 96 | 155,222 | 8,203 | +0.00 pp |

A 22-turn example with 67K of history fails badly (-46.9 pp); it is a singleton, reported as one, and
it points at a limitation: a short session whose individual turns are enormous.

### 4.2 The mobility law and the inversion

| session | state | mobility (lookup + active prefill) | full-KV move |
|---|---:|---:|---:|
| 32K, raw 16K view | 16,384 | 0.4855 s | 0.0173 s |
| **1M, compiled** | **8,192** | **0.2031 s** | 0.5530 s |
| **1M, compiled** | **4,096** | **0.0954 s** | 0.5530 s |
| 262K, compiled | 8,192 | 0.2031 s | 0.1383 s |

and the ranking consequence: with a 4,096 state the 1M session ranks *cheapest* (0.0954 s) while the
32K session carrying a 16K state ranks *most expensive* (0.4855 s); the footprint estimate ranks
them 5.899 s against 0.184 s - exactly backwards.

### 4.3 Capacity

Sessions per worker with 20 GiB usable HBM (state sizes measured; KV geometry measured for the 0.5B
model, declared for 8B/70B classes):

| geometry | resident history, 128K | resident history, 1M | resident compiled state, 8,192 |
|---|---:|---:|---:|
| Qwen2.5-0.5B (12,288 B/token) | 13.3 | 1.7 | **213** |
| 8B-class (131,072 B/token) | 1.2 | 0.2 | **20** |
| 70B-class (327,680 B/token) | 0.5 | 0.1 | **8** |

### 4.4 Routing

Replay over measured G3 primitives (H2D 23.2-23.4 GB/s, active-set prefill 0.053/0.095/0.203/0.485 s
at 2K/4K/8K/16K, lookup by history bucket), with declared arrival models and geometries:

| active set | advancing cells |
|---|---:|
| 2,048 | 17/66 |
| 4,096 | 63/198 |
| **8,192 (quality-admissible)** | **52/306** |
| 16,384 | 2/66 |

134 of 624 cells advance; the 8,192 column covers balanced, hotspot (a worker at a tenth of the
service rate), slow-worker and worker-loss regimes.  The balanced/hotspot/slow-worker wins are the
capacity-pressure regime: a warm cache holding a fraction of the fleet and no cluster KV store, so
the baseline's alternative is a full re-prefill (1.6-14.3 s) against 0.203 s of rematerialisation.
Against a tier that can move KV, the wins are worker loss plus the 1M-history mixes.

### 4.5 Recovery and rollout

| operation | measured |
|---|---|
| two-worker failover, owner SIGKILLed | fresh process rebuilds 8,192 tokens in 0.91-1.42 s (p50 1.387 s); recorded files named in 4/6 sessions; token accuracy identical to the killed owner's (0.593 both) |
| model rollout | resumes on a model that never saw the session: end-task 0.137/0.145 against full history's 0.017/0.042 (Qwen2.5-0.5B/1.5B) |
| data moved | ~32,455 bytes of state text against 0.38 GiB of KV the dissolved owner held at these 33K histories (12-128 GiB at 1M on the declared geometries) |
| cost | 0.55-1.04 s of active-set prefill against 6.31 s of re-prefill |

---

## 5. Related work

**Context caching and tiering.**  Production systems optimise how a history-sized context cache
moves between memory levels and report throughput gains of that optimisation; retention policies for
multi-turn agents decide how long a span of KV should stay resident across tool gaps.  Those
decisions are correct *given* that the resident object is the history.  Our measurement is that the
resident object need not be: the execution state is bounded and independent of age, so the object
these systems move is mostly state that will never be needed again (73-96% of tokens sit in five
blocks).  The tier still holds something - the transcript and the index - but what a cold route
rebuilds is 32 KB of text, not 12-128 GiB of KV.

**Query-dependent workspace virtualisation.**  Virtualising a million-token workspace to host or NVMe
and materialising a query-dependent view is the closest prior position, and it shares the view
mechanism.  What it does not change is the *cost law*: the workspace is still the session's
footprint, moved and materialised as such.  Our claim is about the law - `dM/dL ~ 0`, the inversion,
the ranking reversal - and about the scheduler and capacity consequences that follow, and we measure
that the view itself is *not* where the win comes from: a compiler that "improves" the view is tied
or worse than plain retrieval here, while the bound on the state is what moves the phase boundary.

**Affinity and load balancing.**  Cache-affinity routing and its interaction with load balancing is
well studied, and sticky-until-saturated is the production answer.  We keep those policies as
baselines and change only the miss cost; the phase change we report is therefore attributable to the
cost law rather than to a new heuristic, and we state the regime where the baselines still win
(balanced load with every session resident against a tier that can move KV).

**Compaction-based context management.**  Summarising or compacting history before it re-enters the
model is a crowded space and predates this work.  Our contribution there is negative and specific:
on these traces compaction is neutral at best (dedup alone: tie), the state-compiling stages are
harmful (path collapse -8.5 pp of fidelity, snippet re-selection -20 pp), and the metric usually
quoted for it is saturated.  Reporting that is what keeps the positive claim honest - the win is the
bound, not the compression.

---

## 6. Limitations and non-claims

* **The decision advantage is a moderate-length phenomenon.**  At 64K-96K histories the bounded view,
  the compiler and the full transcript are one indistinguishable cluster (0.123 / 0.111 / 0.113);
  the metric separates arms at ~33K and does not beyond that.
* **The regime scope of the routing result is stated with it.**  Balanced, hotspot and slow-worker
  wins assume a fleet whose warm cache holds a fraction of its sessions and no cluster KV store;
  against a KV-moving tier they reduce to worker loss and the 1M mixes.
* **The 1M lookup row is extrapolation.**  No public trace is that long; the receipt says so.
* **The 8B and 70B geometries are declared, not measured.**  The fabric, prefill and 0.5B KV numbers
  are measured on this card.
* **We do not claim a semantic compiler beats retrieval.**  It does not, on 48 paired sessions
  (-0.104, CI [-0.208, -0.007]) or at equal budget (raw 0.191 against compiled 0.139 at 8,192).
* **We do not claim fidelity and the decision prefer the same *rendering*.**  They prefer the same
  *content*; the rendering that satisfies both is a whole newest window over a consolidated far field.
* **A fleet-level scheduler is modelled, not deployed.**  The G4 replay uses measured primitives and
  declared workloads; the process-level failover is real, the cluster is not.
* **A singleton counterexample is reported as one** (the 22-turn/67K example above) rather than
  smoothed into the table.

---

## 7. Conclusion

For a long-lived agent session, the object a placement, recovery or capacity decision depends on is
not the transcript and not its KV footprint: it is a bounded execution state compiled from the
durable record, and its size is set by the current action and the query rather than by the session's
age.  Measured on real traces, that bound makes a 32x-older session cheaper to move, gives a worker
16x the sessions, makes recovery and model rollouts the same operation, and moves the routing phase
boundary in every regime we model - while the obvious way to shrink the state further, a semantic
compiler, is measured not to pay.  The session keeps growing; the state that must move does not.

---

## Appendix A. Receipts

The claim-to-receipt map is `VERDICT.md` §7 (C1-C10) and `CLAIMS.md` (Q1-Q10);
`benchmarks/check_receipts.py` verifies that every receipt a document cites exists in the repository
(35 cited, 0 missing at the time of writing).  The runner scripts that produce each family are in
`/root/qcc/run_*.sh` and are named in the ledger entries.
