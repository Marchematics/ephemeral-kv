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
bounded and independent of age.  On real coding-agent traces, a 6-8K execution state holds
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
(+0.00 / +0.00 / +1.43 pp).  Third - and this is a design choice rather than an intrinsic property of the traces, as we show by
measurement - because of that bound the cold-route cost stops tracking age: a
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
2. **The state does not grow with the session, at a budget whose validity is measured.**  `dM/dL ~ 0`
   on the fidelity and state-size axes across a 2.4x growth in history and on the turn axis - and,
   separately, the *evidence mass* that shares rare query terms does grow with age (22K tokens at
   45K histories, 46K at 95K), so what makes the bound safe is the measured quality flatness at the
   fixed budget, not an intrinsic ceiling on what a query needs.
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

*Figure 2 (`figures/fig2_two_metrics.svg`) plots these two columns against each other for every
arm, and Figure 6 (`figures/fig6_metric_lesson.svg`) plots the fidelity column against how the
newest evidence was rendered.*

1. **Fidelity is set by how the newest evidence is rendered.**  Kept whole it is at parity with as
   little as 3K of window; prefix-truncated it costs 5-7 pp; reordered into a query-selected excerpt
   20 pp; replaced by a single span 25 pp.  The *far field* must also be consolidated rather than
   prefix-truncated raw dumps, or the surface drops ~6-7 pp even with the window whole - at both
   8,192 and 12,288 tokens, so it is a treatment requirement, not a budget one.
2. **The decision is set by which evidence is retrieved.**  In the end-task configuration the
   window-plus-retrieval view and plain retrieval at the same budget produce byte-identical contexts
   on all 48 instances, and every bounded retrieval arm is statistically tied.  On **96 paired
   sessions** the view the system ships - a verbatim window over a consolidated far field at 8,192
   tokens - scores 0.165 against plain retrieval's 0.169 at 4,096: **paired +0.013, 95% CI
   [-0.068, +0.092], 19 wins / 12 losses** - a tie, at fidelity parity, against an arm that is 25 pp
   out of tolerance.  More budget does not help either: the same family scores 0.191 at 8,192 and
   0.173 at 12,288 on the smaller set, so above ~4K the constraint is retrieval coverage, not budget.

So the two requirements do not compete for content.  They compete for *rendering*: render the newest
evidence whole, consolidate the far field, and retrieve the rest.  That is the view the system
ships, and it is the configuration in bold above.

**A stricter end task, scored offline.**  File *mention* is a weak bar, so the same receipts are
rescored one notch higher: `action_hit` requires the generated turn to name an edit-type tool or
command (`str_replace_editor`, `apply_patch`, `sed -i`, `cat >`, `patch`, ...) *and* to target a file
the recorded patch touched (`g2b_action_metric.py`, `artifacts/g2b-action-metric-v1.json`):

| arm | instances | full-history action rate | active action rate | paired delta |
|---|---:|---:|---:|---:|
| raw retrieval, 4,096 | 96 | 0.042 | 0.031 | -0.010 [-0.062, +0.042] |
| window + compiled far field, 8,192 | 96 | 0.042 | 0.052 | +0.010 [-0.042, +0.062] |
| evidence consolidation, 8,192 | 48 | 0.042 | 0.021 | -0.021 [-0.104, +0.042] |
| raw retrieval, 8,192 | 48 | 0.042 | **0.125** | **+0.083 [0.000, +0.188]** |
| plain recency, 8,192 | 24 | 0.042 | 0.042 | 0.000 [-0.125, +0.125] |

On this bar the bounded state is **not worse** than the full transcript - every interval includes
zero, and one arm is nominally ahead at the edge of significance - but **no arm is measurably
better**, and the absolute rates are low because the metric demands both an edit intent and the
right file.  We report it as a bound our claim must clear rather than as a result: the stronger
metric neither confirms nor refutes the weaker one, and task success remains unmeasured.

**Why 6,144 and not less.**  The budget is measured, not chosen for convenience.  Holding the window
at 3,072 tokens and varying the total:

| total budget | fidelity (window + consolidated far field) | end-task F1 |
|---:|---:|---:|
| 4,096 | **-2.40 pp** (outside the 2 pp allowance) | 0.117 |
| **6,144** | **0.00 pp** | **0.150** (tied with retrieval at 4,096: +0.093, CI [-0.030, +0.217], 13W/9L) |
| 8,192 | 0.00 pp | 0.161 (window ~4.1K) |
| 12,288 | 0.00 pp | 0.155 |

At 4,096 the window consumes three quarters of the budget: the surface is 2.4 pp out of tolerance and
the far field has too little room, so the decision falls to 0.117 against 0.243 for plain retrieval at
the same total.  The joint requirement therefore has a **measured floor at 6,144 tokens** - the
smallest budget at which fidelity is within tolerance *and* the decision is tied with the best
retrieval arm - and above it neither metric improves, which is why the paper's state is 6-8K rather
than as large as the machine will hold.

### 2.3 The metric cannot see the difference

Keep the newest spans whole and teacher-forced fidelity is at parity with **8,192 tokens of
recency** - 9.7% of a 32K-128K history.  A metric that is satisfied by the most naive possible
policy cannot compare systems, and it explains why the compiler literature's proxy would report
"compression works" for a policy that merely truncates.  We therefore gate on the end task (file
level localisation of the next turn against the recorded patch) and report fidelity as a constraint.

---

## 3. The system

### 3.1 The durable object

The transcript and a model-independent index over it.  Each completed turn becomes a *span*
(text, role, turn index, token estimate, kind), the index keeps a posting list per term, a posting
list per kind, and a whitespace-insensitive content fingerprint per span; lookup is a scored
intersection of the query's terms with those postings (~0.09-0.25 ms p50 for histories up to 128K,
sublinear in history because the postings are).  Three properties are load-bearing and each has a
receipt:

* **model independence.**  Nothing in the span, the postings or a compiled view depends on the
  model that built them, so the same durable object resumes on a model that never saw the session
  (end-task 0.137/0.145 against full history's 0.017/0.042 on two such models).
* **append-only.**  A worker never rewrites the durable object; compaction, if any, is a property
  of a *view*, not of the record.
* **provenance.**  Every compiled unit carries the turns it came from and the views it replaced, so
  a view can always be traced back - and the paper's negative result (collapsing a file to its
  latest view) is measurable precisely because the replaced turns are still addressable.

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

*Figure 3 (`figures/fig3_compiler_ablation.svg`) draws the end-task ladder these stages sit on.*

Two of these are negative results the paper keeps, because they bound what "executable state" can
mean on this workload: the stages that *compile state* are the ones that hurt, and the reason is a
property of coding-agent traces (most retrieved evidence is not file events) rather than of the
compiler.

### 3.3 The compile contract

`compile(history, q) -> view` is the only interface between the durable record and a worker, and it
is specified by what it must guarantee rather than by how it selects:

```text
in:   spans (append-only, model-independent), the current turn, the query, a budget B
out:  a view whose token estimate is <= B, with provenance on every unit
must: render the newest evidence whole and in order                     (the surface)
      spend every remaining token on retrieved evidence, consolidated   (the decision)
may:  drop superseded views, duplicate content and stale tool output, so long as
      what it drops stays addressable through provenance
```

The budget B is chosen by the fidelity floor rather than by preference: at 4,096 the same window
scores -2.40 pp, at 6,144 it is at 0.00 pp with the decision tied, and above that neither metric
improves.  Two of the three "may" clauses are measured *not* to pay on this workload - collapsing a
file to its latest view costs 8.5 pp of fidelity, re-selecting the newest output's lines 20 pp - so
the deployed configuration keeps the window whole and consolidates only the far field.

### 3.4 What the runtime does

A cold route is: look up the query in the durable index (sub-millisecond), compile a bounded view
(window + consolidated far field), prefill it on the destination, serve.  Nothing history-sized is
transferred and no worker owns the session; the KV a worker holds is a cache of a computation it can
repeat.  Recovery is the same code path on a different process or a different model: measured at
0.91-1.42 s to rebuild an 8,192-token state after the owner was killed, with teacher-forced token
accuracy identical to the killed owner's, and 0.55-1.04 s for a model rollout against 6.31 s of
re-prefill.

---

## 4. Evaluation

Setup: a single shared 24 GiB A10G; real agent traces (SWE-smith / SWE-rebench style trajectories;
2,736 sessions with histories to 156K tokens for the end task, 338 of them above 64K; a 1,010-session
subset for the fidelity sweeps); Llama-3.2-1B-Instruct for quality, Qwen2.5-0.5B/1.5B for hardware
primitives and cross-model recovery.  All comparisons are paired on the same instances and intervals
are paired bootstrap (20k resamples); receipts whose instance counts differ state both counts rather
than averaging them away.  The fidelity metric is teacher-forced next-token accuracy on the recorded
next turn, scored as a suffix (target capped at 512 tokens) so a 32K context does not materialise
logits for the whole sequence; the end task is file-level localisation of a generated turn against
the recorded patch, with an offline action-level rescoring as a stricter variant.

### 4.1 The state is bounded, and quality does not decay with age

*Figure 1 (`figures/fig1_state_vs_age.svg`).*

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

### 4.2b The baseline this space compares against: model-written compaction

Other systems in this space report their gains *over* compaction-only context management, so the
same baseline is measured here end to end: keep the newest 3,072 tokens verbatim, have a model
write a summary of everything older under the remaining budget, and pack both into 8,192
(`--compile-mode compact`, `scripts/run_compaction.sh`), with the summariser varied separately from
the scorer (`--summarizer-model`).

The corrected measurement is at **fidelity parity**: 0.00 pp (NLL -0.020) with 223 summariser calls
and 395 summary reuses across 48 examples, and a view of 6,650 tokens p50 - because the window is
6,656 tokens of untouched newest evidence and *that* is what carries the surface.  A **stronger
summariser does not change it**: rewriting the same summaries with Qwen2.5-1.5B while the scorer
stays Llama-3.2-1B also gives 0.00 pp (NLL -0.020), so the answer to "your summariser was weak" is
that the summariser is not what the surface depends on.

Which makes the comparison a controlled one: the two designs spend the same budget and keep the
same window, and differ only in what the remaining ~1.5K buys.  Measured on the decision, that
difference is decisive:

| view (8,192 tokens) | window | remainder | fidelity | end-task F1 (n=48) |
|---|---:|---|---:|---:|
| window + consolidated retrieval | 6,656 | 1.5K retrieved | 0.00 pp | **0.089** |
| compaction | 6,656 | 0.5K summary | 0.00 pp | (measuring) |
| **window + consolidated retrieval (shipped)** | **4,096** | **4.1K retrieved** | **0.00 pp** | **0.161** |
| plain retrieval, no window | 0 | 4,096 retrieved | -25.00 pp | 0.243 |
| full history | - | - | reference | 0.089 |

At a 6,656-token window the decision falls to the full-history level (0.089), because the window has
consumed the budget the decision needs; at a 4,096-token window it is 0.161 at the same fidelity.
That is the trade-off the two laws describe, measured along the window axis at a fixed budget, and
it is why the shipped view keeps the *smallest* window that still holds the surface.

For the record, an earlier version of this baseline reported -23.86 pp and was **withdrawn**: the
arm had never called its summariser, because a branch-placement bug left the "compaction" view a
malformed recency variant.  The bug is fixed - the branch is hoisted out of the consolidation gate,
the window respects the same budget as every other arm, the summary is counted in the reported
state, and every row records how many times the summariser was called and how often a summary was
reused - so a receipt that claims to be a compaction can be checked rather than trusted.

### 4.3 Capacity

Sessions per worker with 20 GiB usable HBM (state sizes measured; KV geometry measured for the 0.5B
model, declared for 8B/70B classes):

| geometry | resident history, 128K | resident history, 1M | resident compiled state, 8,192 |
|---|---:|---:|---:|
| Qwen2.5-0.5B (12,288 B/token) | 13.3 | 1.7 | **213** |
| 8B-class (131,072 B/token) | 1.2 | 0.2 | **20** |
| 70B-class (327,680 B/token) | 0.5 | 0.1 | **8** |

### 4.4 Routing

*Figure 4 (`figures/fig4_phase_diagram.svg`).*

Replay over measured G3 primitives (H2D 23.2-23.4 GB/s, active-set prefill 0.053/0.095/0.203/0.485 s
at 2K/4K/8K/16K, lookup by history bucket), with declared arrival models and geometries:

| active set | advancing cells |
|---|---:|
| 2,048 | 17/66 |
| 4,096 | 63/198 |
| **8,192 (quality-admissible)** | **52/294** |
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

Two lines of work are close enough that the difference has to be stated in objects and in cost
laws rather than in adjectives.

**Hierarchical context caching.**  [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
caches KV across GPU HBM, host memory and SSDs, and its contributions are a GPU-assisted I/O
mechanism that decouples layouts so large transfers are possible, and a cache-aware scheduler that
mitigates delay hits and hides cache-loading latency; it is implemented in SGLang, deployed, and
reports up to 5x throughput over vLLM-LMCache.  Its stated problem is that naive designs become
I/O-bound: fragmented layouts cause small transfers, cache loading stalls prefill.  That is the
right optimisation *given* that the object being moved is the session's history.  Our measurement is
that the object need not be: the execution state is 6-8K tokens compiled from a model-independent
index, so what a cold route moves is ~32 KB of text and the transfer term leaves the cold path
instead of being made efficient.  Strata's cache-aware scheduling remains the right design for the
durable tier, where the transcript and index do live.

**KV virtualisation for agent workspaces.**  [KVMem](https://www.alphaxiv.org/abs/2609.04852)
preserves overflowed workspace history as paged KV state across GPU, host and NVMe, indexes it with
model-native attention-space summaries (Mean-K over blocks), and materialises a *query-dependent
execution view* bounded by the model's native context window - 1M tokens of workspace on a 24 GB
consumer GPU for a 27B model, with DeepSWE task success improving from 43.8% under compaction-only
context management to 48.4%.  Two things are shared and we do not claim them: the idea of a
query-dependent view, and the observation that compaction is lossy.  Two things differ, and they are
the paper's subject:

| | KVMem | this paper |
|---|---|---|
| what the view is assembled from | paged **KV blocks**, with raw keys re-rotated at new positions | **text** compiled from a model-independent index |
| what a cold route moves | KV blocks from host or NVMe | ~32 KB of state text; no model-specific state |
| what bounds the view | the model's native context window (256K for the model it evaluates) | the query and the current turn: **6-8K, measured** (4,096 scores -2.40 pp) |
| model change | RoPE re-application keeps the blocks usable by the same model | the same durable object resumes on a **different** model (0.137/0.145 against 0.017/0.042) |
| reported end task | DeepSWE task success 43.8% -> 48.4% over compaction | next-turn file localisation and an offline action-level rescoring; task success unmeasured |
| reported cost | KV restoration under one second for a 1M workspace | 0.0954-0.2031 s to rebuild a 4-8K state, 0.55-1.04 s of active-set prefill for rollout |


* **The object.**  KVMem's view is assembled from *KV blocks*, and its cold path is a transfer with
  RoPE re-application at the new positions.  Our state is *text* compiled from a model-independent
  index; nothing model-specific is transferred, and the same durable object resumes on a different
  model (measured: end-task 0.137/0.145 against full history's 0.017/0.042 on two models that never
  saw the sessions).
* **The size.**  KVMem's view is bounded by the model's native window - 256K tokens for the model it
  evaluates - while ours is bounded by the query and the current turn at 6-8K, a measured floor
  (below it, 4,096 scores -2.40 pp of fidelity).  That difference is what produces the mobility,
  capacity and routing consequences: a 32x-older session costing 2.4x less to move, 16x the
  sessions per worker, and a routing phase change at a state size the quality measurements certify.

KVMem also reports an end-to-end agent-success metric that we do not: our end task is file-level
localisation of the next turn against the recorded patch.  Task success on a benchmark like DeepSWE
is the stronger evidence for a deployed agent, and adopting it is the natural next step for this
work rather than something we have measured.

**Affinity, retention and load balancing.**  Cache-affinity routing and its interaction with load
balancing is well studied, and sticky-until-saturated is the production answer; retention policies
for multi-turn agents decide how long a span of KV stays resident across tool gaps, and workspace
virtualisation decides what to keep where.  We keep those policies as baselines and change only the
miss cost, so the phase change we report is attributable to the cost law rather than to a new
heuristic - and we state the regime where the baselines still win: balanced load with every session
resident, against a tier that can move KV.

**Compaction-based context management.**  Summarising or compacting history before it re-enters the
model is a crowded space and predates this work; KVMem itself uses compaction as its baseline, and
Section 4.2b measures that baseline under the same budget and instances: keeping the newest 3,072
tokens verbatim and having the model summarise the rest scores **-23.86 pp** of fidelity where
keeping the evidence and retrieving the rest scores 0.00 pp, with the decision tied in both.  Our
contribution there is negative and specific: on these traces compaction is neutral at best (dedup
alone is a statistical tie with plain retrieval), the state-compiling stages are harmful
(collapse-each-file-to-its-latest costs 8.5 pp of fidelity, snippet re-selection of the newest output
20 pp), and the metric usually quoted for it - teacher-forced next-token fidelity - is saturated by
keeping the newest evidence whole.  Reporting that is what keeps the positive claim honest: what
carries quality is the window and retrieval, and what carries mobility is the bound.

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
* **We do not claim `|E_q|` is intrinsically bounded.**  Lexical evidence mass grows with session
  length on these traces; the bound is chosen, and its validity is the measured quality flatness at
  that budget.
* **We do not claim a semantic compiler beats retrieval.**  It does not, on 48 paired sessions
  (-0.104, CI [-0.208, -0.007]) or at equal budget (raw 0.191 against compiled 0.139 at 8,192).
* **We do not claim fidelity and the decision prefer the same *rendering*.**  They prefer the same
  *content*; the rendering that satisfies both is a whole newest window over a consolidated far field.
* **We do not measure an end-to-end agent-success metric.**  Our end task is file-level
  localisation of the next turn; the closest published comparison - KVMem's DeepSWE success
  improving from 43.8% under compaction to 48.4% with KV virtualisation - is not reproduced here,
  and a task-success benchmark is the metric a deployed claim should carry.
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

## Appendix A. Claims, receipts and reproduction

Every claim in this paper maps to a receipt in `artifacts/` and to the script in `scripts/` that
produced it.  `benchmarks/paper_numbers.py` re-derives the headline numbers from the receipts and
asserts them (27 checks); `benchmarks/check_receipts.py` verifies that every receipt a document
cites exists and is not a partial write; `benchmarks/make_figures.py` regenerates the figures from
the receipts.

| claim | receipt | script |
|---|---|---|
| most of a session is state that is never needed again | `g2-dead-state-v1.json` | `benchmarks/g2_dead_state.py` |
| the state is bounded and the fidelity delta does not move with age | `g2-killer-table-v4.json` | `benchmarks/g2_killer_table.py` |
| the budget floor is 6,144 tokens | `g2-compiler-window3k-b{4096,6144,12288}-v1.json` | `scripts/run_budget_lower.sh`, `-window3k_raw.sh` |
| the joint view ties the best retrieval arm (n=96) | `g2b-patch-localization-{windowcompiler-b8192-n96,raw-b4096-n96}.json` | `scripts/run_joint_n96.sh` |
| the stricter action-level end task is a bound, not a win | `g2b-action-metric-v1.json` | `benchmarks/g2b_action_metric.py` |
| compaction costs ~24 pp of fidelity at the same budget | `g2-compiler-compact-b8192-v1.json` | `scripts/run_compaction.sh` |
| the compiler's stages do not pay, with attribution | `g2-compiler-{consolidate,dedup,materialize,statefirst,protectwhole}-*.json` | `scripts/run_tail_rerun.sh` |
| the mobility law and the inversion | `g2-killer-table-v4.json`, G3 receipts | `benchmarks/g6_placement_inversion.py` |
| the routing phase change at an admissible state | `g4-quality-join-v2.json` | `scripts/run_g4_{grid2,geometry,pressure,hotspot}.sh` |
| capacity and recovery | `g5-capacity-planning-v1.json`, `g5-failover-two-workers-samemodel-v1.json` | `benchmarks/g5_*.py` |

### A.1 How these numbers are kept honest

Four checks run against the repository rather than against the prose:

* **value audit** - `benchmarks/paper_numbers.py` re-derives every headline number from the
  receipts and asserts it (27 checks, including that the routing cells which clear a strict
  retrieval-parity bar are exactly zero, and that the stricter end task's interval contains zero);
* **receipt audit** - `benchmarks/check_receipts.py` reports any cited receipt that is missing, and
  any whose payload is a *partial* write (the harnesses write incrementally, so a killed run leaves
  a partial artifact at its final path);
* **semantic audit** - receipts record facts about what the arm actually did (for example
  `summariser_calls` per row), and the audit asserts them, because a value check cannot catch a
  receipt that does not do what its name claims.  This is not hypothetical: an earlier version of
  the compaction baseline in this paper reported a plausible number while never calling its
  summariser at all, and was withdrawn when the semantic check exposed it;
* **figure regeneration** - `benchmarks/make_figures.py` and `make_svg_figures.py` rebuild every
  figure from the receipts, and re-running them leaves the tree unchanged.

Two failure modes were found while building this and are worth naming, because both produced
*plausible-looking* evidence rather than errors.  A killed duplicate run left a **partial artifact
at its final path** (16 of 48 rows), and a value audit cannot see that - the numbers it checks are
real, just few.  And two arms were reported as queued for two rounds while **never having started**,
because one harness did not accept a flag they passed and the runner scripts had no `set -e`: an
experiment that never starts looks exactly like one that is still running.  The defences are the
partial-receipt check, the semantic checks (`summariser_calls`), and queues that stop on failure.

The runner scripts that produced each family of receipts are in `scripts/`, so the path from a
claim to its evidence is a claim -> receipt -> script triple, and the paper states which of the
three things that triple cannot cover: extrapolated rows, declared geometries, simulated clusters,
and unmeasured task success.

What is *not* here is as deliberate: the 1M lookup row is extrapolated (no public trace is that
long), the 8B/70B geometries are declared rather than measured, the fleet-level scheduler is a replay
over measured primitives rather than a deployment, and task success on a benchmark like DeepSWE is
not measured at all.  Those four sentences are the paper's honest perimeter, and none of them is
load-bearing for a claim above.
