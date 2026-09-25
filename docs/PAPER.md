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
bounded and independent of age.  On real coding-agent traces, a 4-8K execution state holds
the next turn at a teacher-forced fidelity delta of **0.00 pp** against the full transcript, and its
end-task score is statistically indistinguishable from the best retrieval baseline we can build,
while the state stays at **7-8K tokens** as the raw history grows from 64K to 156K (and, on the turn
axis, from 40 to 96 turns).  The consequence is a phase change rather than a speedup: a session
**32x older costs 2.4x less to move** (5.1x at a 4K state) because the transfer term leaves the cold
path; one worker holds **32x more sessions**; recovery and a model revision rebuild the state from a
durable index instead of moving 12-128 GiB of KV; and a replay against measured hardware primitives
advances routing in **117 cells at the states where fidelity holds** - 63 of them at 4,096 tokens,
52 at 8,192 and 2 at 16,384 (the decision at those sizes is a tie with the best retrieval baseline,
not a win) - across the three
regimes the replay models - balanced, slow-worker (a worker at a tenth of the service rate) and
worker-loss - where the original grid advanced in four cells at a 2K corner that no quality
measurement supported.

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

1. **A bounded execution state holds the turn.**  8,192 tokens shipped, with a *measured* floor of
   4,096 (3,584-token window; the same total with a 3,072-token window scores -2.40 pp): fidelity
inside the 2 pp allowance against the full transcript, end-task score
   statistically tied with the best retrieval baseline measured.
2. **The state does not grow with the session, at a budget whose validity is measured.**  `dM/dL ~ 0`
   on the fidelity and state-size axes across a 2.4x growth in history and on the turn axis - and,
   separately, the *evidence mass* that shares rare query terms does grow with age (22K tokens at
   45K histories, 46K at 95K), so what makes the bound safe is the measured quality flatness at the
   fixed budget, not an intrinsic ceiling on what a query needs.
3. **Session age stops predicting placement cost.**  The inversion, and the ranking reversal that
   makes a footprint-based scheduler prefer exactly the wrong session.
4. **The consequences are systemic**: 32x sessions per worker at the measured floor, lossless recovery on a fresh process
   (0.91-1.42 s, no KV transfer), and a routing phase change at the 8,192-token state where fidelity
   holds, in every regime we model, including a 10x-slow hotspot.
5. **A negative result with an exact attribution.**  The obvious way to shrink the state - a semantic
   compiler - does not pay here: it is tied or worse on the decision, and the two stages that
   actually "compile state" are the ones that hurt.  The window and retrieval are what carry quality.
   Compaction, the baseline this space uses, is measured separately and is **equivalent** rather than
   worse (Section 4.3), so the paper claims no win over it either.
6. **A metric lesson.**  Teacher-forced next-token fidelity is saturated by the newest evidence on
   these traces and therefore cannot compare systems; the end task can, and we report both for every
   arm.

---

## 2. What a turn needs

### 2.1 Most of a session is state that is never needed again

**Table 1:** Where a long session's tokens sit. The spans that dominate a session are a small
fraction of the turns that produced them - which is the whole reason a bounded state is possible.

Table 1 measures the token share held by the largest spans of each long session
(`g2_dead_state.py`):

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

**How far that concentration extends is measured, and the answer is that it does not extend to
composed lengths.**  Concatenating whole real sessions into 145K, 278K, 545K and 1.07M-token
histories (`benchmarks/build_composed_long_sessions.py`, 12 examples per bucket) leaves the largest
span at ~28K and the top-5 share at **3-14%** (p50 3%, ~2,600 spans per session), against 73-96% on
real sessions of 123K-156K.  Concentration is a property of *a real session*, not of a long token
stream: at a million tokens the same tokens are spread across twenty-five sessions' worth of tool
output.  What justifies a bounded state at those lengths is therefore not that a few spans dominate
but that the turn does not need them, which is what the two laws in Section 2.2 and the decision
measurements in Section 4.3 establish - and it is why the bound is reported as a design choice
validated by quality rather than as a property the corpus hands us.

### 2.2 Two laws, and the view that satisfies both

**Table 2:** Every arm at an 8,192-token view, on both metrics.

Table 2 lists every arm measured at an 8,192-token view on the same examples (fidelity n=44
long-history examples, decision n=48 paired sessions, 26 scoreable):

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

**Figure 2** plots the two metrics against each other and **Figure 6** plots the fidelity column
against how the newest evidence was rendered.

**Figure 2:** The two metrics against each other for every arm. No arm moves both: the arms that
hold fidelity spend the budget on the window, the arms that win the decision spend it on retrieval.

**Figure 6:** Fidelity against how the newest evidence was rendered. Rendering it whole is what
holds the surface; the way the far field is compiled is what the decision reads.

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
the recorded patch touched (`g2b_action_metric.py`, `artifacts/g2b-action-metric-v1.json`);
Table 3 reports it for every arm:

Read the other way, the same arms are a frontier, and it is the honest answer to "does the bounded
state beat retrieval": of the twelve arms measured at this budget that hold fidelity, the bounded
state's score highest - **0.179** (n=24) and 0.165 (the shipped view, n=96) against full history's
0.089 and compaction's 0.122 - and every arm that scores *higher* pays for it in fidelity:
evidence consolidation 0.292 at -5.44 pp, state-first 0.242 at -15.38 pp, the collapsing compiler
0.237 at -13.90 pp, log replay 0.231 at -7.21 pp, protect-whole 0.211 at -20.33 pp, and raw
retrieval 0.191 at -6.94 pp.  So the claim is not that
a bounded view decides better than retrieval, which it does not; it is that among the views a
fidelity-gated system may actually use, this is the one that decides best.

**Table 3:** The stricter action-level end task: a continuation has to name the right kind of
command *and* target a file the recorded patch touched.

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

**The compiler does not rescue the floor; the window does.**  The one remaining place a state
compiler could have paid is the size the system actually runs at, where the far field has least room
and materialised state is most attractive.  Measured there, log replay into materialised state
scores **-2.48 pp** at a 4,096-token total against consolidation's **-2.40 pp** - the same total,
the same window, no gain - while widening the window to 3,584 at that total reaches **-0.96 pp**.
So the surface is bought by keeping more of the newest evidence whole, not by compiling the older
evidence harder, which is the negative result Section 2.2's two laws predict and Section 3.2's
ablation attributes.

**Why 4,096 and not less.**  The budget is measured, not chosen for convenience, and Table 4 varies
the total *and* the window, because which of the two binds depends on the other:

**Table 4:** The budget floor. The smallest measured state that holds the surface is **4,096 tokens
with a 3,584-token window**; the identical total with a 3,072-token window does not hold it, which
says the window rather than the total is what the surface needs.

| total budget | window | fidelity (window + consolidated far field) | end-task F1 |
|---:|---:|---:|---:|
| 4,096 | 3,072 | -2.40 pp (outside the 2 pp allowance) | 0.117 |
| **4,096** | **3,584** | **-0.96 pp** (inside the allowance) | measured in the join, below |
| 4,608 | 3,584 | 0.00 pp | - |
| 6,144 | 3,072 | 0.00 pp | 0.150 (tied with retrieval at 4,096: +0.093, CI [-0.030, +0.217], 13W/9L) |
| 8,192 | ~4.1K | 0.00 pp | 0.161 |
| 12,288 | ~5.3K | 0.00 pp | 0.155 |

Three things follow, and all three are measured.  At a 4,096-token total with a 3,072-token window
the surface is 2.4 pp out of tolerance and the decision falls to 0.117 against 0.243 for plain
retrieval.  Holding the total at 4,096 and widening the *window* to 3,584 brings the surface inside
the allowance (-0.96 pp, NLL +0.049 against the narrower window's +0.129) - better on the surface
than the 3,072-token window at a 6,144-token total (+0.061), which is the whole point: **the surface
tracks how much newest evidence is kept whole, not how large the budget is.**  And above 4,096 more
budget does not improve either metric, which is why the paper's state is **4-8K** rather than as
large as the machine will hold.  The margin is worth stating: the 4,096-token floor spends half the
2 pp allowance (-0.96 pp), where the 6,144-token arm spends none of it, so a deployment that wants
the whole allowance in reserve should run the 6,144-token configuration - the paper reports both and
the routing consequence is computed at the size each implies.

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

A compiler contract is only worth stating if its stages pay for themselves.  Table 5 lists the
stages, what each means, and what each is worth when measured.

**Table 5:** The compiler's stages, their semantics, and what each is worth when measured.

| stage | semantics | measured effect |
|---|---|---|
| recency window | the current turn is not optional; newest evidence admitted whole | fidelity -6.94 -> **0.00 pp** |
| content-identity dedup | one copy of repeated tool output and repeated reasoning | neutral: 0.239 against 0.243 (paired -0.004, CI [-0.024, +0.016]) |
| supersede by state identity, no path collapse | a file's or command's newest state stands for its earlier views, replaced turns kept as provenance | fidelity -6.94 -> -5.44 pp; decision 0.237 -> 0.292 (n=24) |
| collapse each path to its latest state | (the ablation) | **-13.90 pp**, i.e. 8.5 pp worse |
| snippet re-selection of an oversized newest span | keep the query-relevant lines | **-20 pp** on fidelity |
| log replay into materialised state | dumps set content, diffs and SEARCH/REPLACE apply, repeated commands collapse | 0.231 against 0.237 on the decision; neutral, because only **6%** of retrieved spans are file events.  At the 4,096-token floor it is also *worse on the surface* than mere consolidation: -2.48 pp against -2.40 pp at the same total, where widening the window instead reaches -0.96 pp |
| identifier provenance | pull spans sharing paths/ids with the query | part of the 4K -> 2K improvement |

**Figure 3** draws the end-task ladder these stages sit on.

**Figure 3:** The end-task ladder the compiler's stages sit on. Dedup is neutral, and the two
stages that re-select or collapse the newest evidence are what cost the decision.

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
scores -2.40 pp, at the same 4,096-token total with a 3,584-token window it is at -0.96 pp, at 6,144
it is at 0.00 pp with the decision tied, and above that neither metric
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


Table 6 gives raw history against the compiled state's size and the fidelity delta at three
session lengths, Table 7 the same along the turn axis, and Figure 1 plots both against age.

**Table 6:** The state does not grow with age: raw history against the compiled state's size and
fidelity on the same sessions.

| raw history p50 | execution state p50 | fidelity delta |
|---:|---:|---:|
| 64,401 | 7,133-7,829 | +0.00 pp |
| 82,972 | 7,133-7,899 | +0.00 pp |
| 155,574 | 8,203 | +1.43 pp |

**Table 7:** The same measurement along the turn axis.

| turns p50 | raw history p50 | execution state p50 | fidelity delta |
|---:|---:|---:|---:|
| 40 | 79,769 | 8,121 | -1.43 pp |
| 58 | 84,617 | 7,887 | **+0.00 pp** |
| 96 | 155,222 | 8,203 | +0.00 pp |

**Figure 1:** The compiled state against session age. Raw history grows by 2.4x across these
sessions while the state stays inside its bound and the fidelity delta stays inside the allowance.

A 22-turn example with 67K of history fails badly (-46.9 pp); it is a singleton, reported as one, and
it points at a limitation: a short session whose individual turns are enormous.

**Table 7b:** The same law past the corpus, on histories composed out of whole real sessions
(`benchmarks/build_composed_long_sessions.py`). The state is the shipped 8,192-token view; the
tokens and bytes are CPU measurements of what a cold route transfers, and the rows count every
qualifying turn in the bucket.

| corpus | raw history p50 | turns p50 | state tokens p50 | state KB p50 |
|---|---:|---:|---:|---:|
| real sessions (64K+) | 111,084 | 42 | 8,203 | 18.1 |
| composed | 140,666 | 291 | 6,598 | 40.1 |
| composed | 284,518 | 614 | 7,174 | 25.9 |
| composed | 514,211 | 963 | 7,010 | 25.4 |
| composed | 983,692 | 1,879 | 7,039 | 24.8 |

Across an **8.9x** range of raw history - and a 45x range of turns - the state that must move stays
between 4,588 and 8,439 tokens with no trend (p50 6,598 -> 7,039), and between 18 and 40 KB of text.
**The session keeps growing; the state that must move does not.**  The quality side of these same
buckets - whether the decision and the surface hold at these lengths - is measured with the arms of
Section 2.2 and reported in Section 4.2; the composition itself is the honest limit: the transcripts
are real, the million-token sessions are not.

### 4.2 The mobility law and the inversion

Table 8 gives the two costs side by side.  The lookup term in it is now measured rather than
extrapolated: on real transcripts concatenated into 143K, 301K, 530K and 1.07M-token histories, the
durable index answers the runtime's query in **0.27 / 0.21 / 0.31 / 0.68 ms p50** (p95 1.35 ms) - and
it is the *query* that sets the cost, not the history: the same million-token index answers an
11-token query in 0.40 ms and a 388-token query in 3.8 ms, while a real 141K-token session whose
last tool message carries 45K tokens pays 15 ms.  Against 53-203 ms of active-set prefill the term
is negligible either way, which is the point: the cold path's cost tracks what the query asks for,
not how old the session is.

**Table 8:** Mobility: rebuilding and prefilling the compiled state, against moving the session's
full KV.

| session | state | mobility (lookup + active prefill) | full-KV move |
|---|---:|---:|---:|
| 32K, raw 16K view | 16,384 | 0.4855 s | 0.0173 s |
| **1M, compiled** | **8,192** | **0.2031 s** | 0.5530 s |
| **1M, compiled** | **4,096** | **0.0954 s** | 0.5530 s |
| 262K, compiled | 8,192 | 0.2031 s | 0.1383 s |

and the ranking consequence: with a 4,096 state the 1M session ranks *cheapest* (0.0954 s) while the
32K session carrying a 16K state ranks *most expensive* (0.4855 s); the footprint estimate ranks
them 5.899 s against 0.184 s - exactly backwards.

### 4.3 The baseline this space compares against: model-written compaction

Other systems in this space report their gains *over* compaction-only context management, so the
same baseline is measured here end to end: keep the newest 6,656 tokens verbatim, have a model
write a 512-token summary of everything older (incrementally, as a serving stack would), and pack
both into 8,192
(`--compile-mode compact`, `scripts/run_compaction.sh`), with the summariser varied separately from
the scorer (`--summarizer-model`).

The corrected measurement is at **fidelity parity**: **+0.13 pp** (NLL -0.032 in the 32K-128K bucket,
p50 over those examples) with 346 summariser calls and 915 summary reuses across 48 examples, a view
of 7,159 tokens p50, and the summary present in **every** example's scored context (106-512 summary
tokens, recorded per row as `summary_tokens_in_view`) - because the window is 6,656 tokens of
untouched newest evidence and *that* is what carries the surface, with the summary adding a slight
improvement on the loss.  The summariser is fixed (the scorer writes the summaries, and at this size
it mostly copies older text rather than abstracting it); one summariser family is a scope limit of
this measurement, not a claim that the surface is summariser-independent.

An earlier version of this arm was measured with a **window-only view**: the builder skipped the
oversized span it was meant to summarise, so no summary was ever built on the long-history turns the
arm keeps, while the summariser was still called on other turns and the view size still matched.  The
arm above is the corrected run - the same runner, with the summary in the view - and the difference
is visible exactly where it should be: the NLL delta moves from -0.020 to -0.032 and fidelity from
0.00 to +0.13 pp.  Appendix A.2 records it as the third silent-failure mode.

Which makes the comparison a controlled one: the two designs spend the same budget and keep the
same window, and differ only in what the remaining ~1.5K buys.  On the decision, that comparison
is the last cell of Table 9:

**Table 9:** What the remainder of an 8,192-token budget buys, at three window sizes.

| view (8,192 tokens) | window | remainder | fidelity | end-task F1 |
|---|---:|---|---:|---:|
| window + consolidated retrieval | 6,656 | 1.5K retrieved | 0.00 pp | 0.089 (n=48) |
| compaction | 6,656 | 0.5K summary | +0.13 pp | 0.122 (n=48) |
| **window + consolidated retrieval (shipped)** | **4,096** | **4.1K retrieved** | **0.00 pp** | **0.161 (n=48)** |
| window (2,867) + compiled far field | 2,867 | 5.3K retrieved | 0.00 pp | 0.179 (n=24) |
| plain retrieval, no window | 0 | 4,096 retrieved | -22.84 pp | 0.243 (n=48) |
| full history | - | - | reference | 0.089 (n=48) |

**Compaction is equivalent here, not worse.**  Against the same-window retrieval arm the paired
difference is **+0.033, 95% CI [-0.038, +0.105]** (7 wins / 3 losses); against the shipped view it is
**-0.039, 95% CI [-0.148, +0.064]** (7 wins / 7 losses).  Both intervals include zero.  So the
controlled answer to "what should the remainder of the budget buy" is that at this window and this
sample size it does not measurably matter - and, importantly, the withdrawn claim that summarising
*damages* the view is not supported by the corrected measurement either.  Compaction is a viable
baseline on this corpus, and the paper's contribution is not that it beats compaction.

What the table does show is where the decision comes from: at a 6,656-token window it drops to the
full-history level (0.089-0.122), because the window has consumed the budget the decision needs,
while a 4,096-token window spends 4.1K on retrieval and reaches 0.161, and a 2,867-token window
spends 5.3K and reaches 0.179 - **at 0.00 pp of fidelity in all three cases**.  Paired on the
instances the arms share, each step in that direction is positive and none is resolvable on its own:
6,656 -> 4,096 is **+0.072, 95% CI [-0.041, +0.188]** (8W/5L, n=48), 6,656 -> 2,867 is **+0.069,
CI [-0.121, +0.271]** (5W/5L, n=24) and 4,096 -> 2,867 is **+0.040, CI [-0.132, +0.220]** (4W/4L,
n=24).  **The lever is the window/budget split**, which is the two laws read along one axis rather
than a claim about summaries - and it is why the window can be as small as ~2.9K while the state
stays 4-8K: the window's job is the surface, and every token beyond it that the surface does not
need is a token the decision does not get.

The honest limit on all three of those comparisons is the same: 48 paired sessions, 26 scoreable,
and differences of 0.03-0.07 are not resolvable at that size.  The extremes are: plain retrieval
with no window scores 0.243 while losing 22.8 pp of fidelity, and full history scores 0.089.

For the record, an earlier version of this baseline reported -23.86 pp and was **withdrawn**: the
arm had never called its summariser, because a branch-placement bug left the "compaction" view a
malformed recency variant.  The bug is fixed - the branch is hoisted out of the consolidation gate,
the window respects the same budget as every other arm, the summary is counted in the reported
state, and every row records how many times the summariser was called and how often a summary was
reused - so a receipt that claims to be a compaction can be checked rather than trusted.

### 4.4 Capacity

Sessions per worker with 20 GiB usable HBM (state sizes measured; KV geometry measured for the 0.5B
model, declared for 8B/70B classes):

**Table 10:** Sessions per worker at 20 GiB of usable HBM: resident history against resident
compiled state.

| geometry | resident history, 128K | resident history, 1M | compiled state, 8,192 | compiled state, 4,096 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B (12,288 B/token) | 13.3 | 1.7 | 213 | **427** |
| 8B-class (131,072 B/token) | 1.2 | 0.2 | 20 | **40** |
| 70B-class (327,680 B/token) | 0.5 | 0.1 | 8 | **16** |

The compiled-state columns are one number per geometry rather than one per history length, because
the state does not grow with age (Section 4.1): a worker holds **427** 0.5B sessions whether their
histories are 64K tokens or a million, which is **32x** the resident-history column and stays 32x at
every age.  Capacity planning stops depending on session age the same way placement does.

### 4.5 Routing

**Figure 4** shows the phase diagram that replay produces.

**Figure 4:** The routing phase diagram: cells where the bounded state advances against the
history-resident baseline, by active-set size and by regime.

Replay over measured G3 primitives (H2D 23.2-23.4 GB/s, active-set prefill 0.053/0.095/0.203/0.485 s
at 2K/4K/8K/16K, lookup by history bucket), with declared arrival models and geometries.  Table 11
gives the advancing cells:

**Table 11:** Advancing cells by active-set size in the routing replay, over the four regimes.

| active set | advancing cells |
|---|---:|
| 2,048 | 17/66 (fidelity -27.92 pp: not admissible) |
| **4,096 (fidelity-admissible)** | **63/198** |
| **8,192 (fidelity-admissible)** | **52/294** |
| **16,384 (fidelity-admissible)** | **2/66** |

134 of 624 cells advance, and **117** of them sit in a column whose measured quality point passes:
the **4,096** column, where a 3,584-token window holds the surface at -0.96 pp (the 3,072-token
window that scores -2.40 pp at the same total is what kept this column out before), the 8,192 column,
and the 16,384 column whose retrieval-only point is -1.94 pp with a 0.142 decision.  The 4,096 and
8,192 columns cover all three regimes the replay models (balanced, slow-worker - the hotspot, a
worker at a tenth of the service rate - and worker-loss).  It is admissible because its fidelity holds
(0.00 pp); its decision is a tie with plain retrieval, so the cells that advance there do so on cost,
and a strict decision-parity bar clears none of them.  The balanced and slow-worker wins are the
capacity-pressure regime: a warm cache holding a fraction of the fleet and no cluster KV store, so
the baseline's alternative is a full re-prefill (1.6-14.3 s) against 0.203 s of rematerialisation.
Against a tier that can move KV, the wins are worker loss plus the 1M-history mixes.

**How much, not just where.**  Every one of the 52 advancing cells at the admissible state clears
the 1.5x SLO-goodput bar that defines an advance, and none needs the alternative route through the
tail (the best p99 improvement in that column is 26%, short of the 30% bar).  Half of them clear it
with a finite ratio - median **1.72x**, max **5.06x** - and in the other half the strongest baseline
completes *no* work at all, so the ratio is unbounded rather than large: 26 of 52.  By regime,
balanced 10 cells (all finite, median **2.15x**), slow-worker 13 (5 finite, median 3.15x), worker
loss 29 (11 finite, median 1.56x).  The honest limit is the tail: this is a throughput win, and in
the regimes where the baseline stalls the ephemeral policy's own p99 is worse in absolute terms -
the p99 reduction is positive only in the balanced cells (median +6%, max +26%), because a baseline
that completes nothing still has a p99.

**And the same at the smaller state.**  The 4,096 column's 63 cells behave the way the 8,192
column's do: every one clears the 1.5x bar, 53 of them with a finite ratio (median **1.57x**, max
2.19x) and 10 against a baseline that completes no work - balanced 8 cells (median 1.61x),
slow-worker 11 (1.63x), worker loss 44 (1.50x).  Across the **117** admissible cells: 79 clear the
bar with a finite ratio (median **1.61x**) and 38 face a baseline that completes nothing.  No cell in
either column advances through the p99 route - the best reduction in the admissible region is +26%,
under the 30% bar - so this is a throughput result, and the paper says so rather than implying a tail
improvement it did not measure.

**What the region's size was waiting on.**  The columns differ by cell count: 4,096 advances in 63
cells of 198 against the 8,192 column's 52 of 294, so moving the system to a 4,096-token state
widens the region from 52 to 115 cells at *half* the state size - and capacity and mobility improve
with it.  That needed one measurement, and it was the window rather than the budget: the same
4,096-token total with a 3,584-token window scores **-0.96 pp** and puts the column inside the
allowance, so the region is **117 cells** (with the 16,384 column's 2) and the state is 4-8K.  The
same 3,072-token window holds 0.00 pp once the total is 6,144, which is why the earlier reading put
the floor at 6,144.  The stricter bar is a second, larger target: decision parity
with plain retrieval at the same budget means reaching its floor of 0.237-0.243 F1, where the best
fidelity-admissible arms sit at 0.161-0.179, a gap of 0.06-0.08.  We report both targets rather than
treating the region's current width as a property of the workload.

### 4.6 Recovery and rollout

Table 12 records what the runtime does under process loss and what reuse across models costs.

**Table 12:** What the runtime does when it loses a worker, and what resuming on another model
costs.

| operation | measured |
|---|---|
| two-worker failover, owner SIGKILLed | fresh process rebuilds 8,192 tokens in 0.91-1.42 s (p50 1.387 s); recorded files named in 4/6 sessions; token accuracy identical to the killed owner's (0.593 both) |
| model rollout | resumes on a model that never saw the session: end-task 0.137/0.145 against full history's 0.017/0.042 (Qwen2.5-0.5B/1.5B) |
| data moved | ~32,455 bytes of state text against 0.38 GiB of KV the dissolved owner held at these 33K histories (12-128 GiB at 1M on the declared geometries) |
| data moved, at any age | the state is 18-40 KB of text at every history length measured, from 111K to 984K tokens (Table 7b), so what a recovering worker reads does not grow with the session either |
| cost | 0.55-1.04 s of active-set prefill against 6.31 s of re-prefill |

---

## 5. Related work

Two lines of work are close enough that the difference has to be stated in objects and in cost
laws rather than in adjectives.

**Hierarchical context caching.**  Strata [1] caches KV across GPU HBM, host memory and SSDs, and its contributions are a GPU-assisted I/O
mechanism that decouples layouts so large transfers are possible, and a cache-aware scheduler that
mitigates delay hits and hides cache-loading latency; it is implemented in SGLang, deployed, and
reports up to 5x throughput over vLLM-LMCache.  Its stated problem is that naive designs become
I/O-bound: fragmented layouts cause small transfers, cache loading stalls prefill.  That is the
right optimisation *given* that the object being moved is the session's history.  Our measurement is
that the object need not be: the execution state is 4-8K tokens compiled from a model-independent
index, so what a cold route moves is ~32 KB of text and the transfer term leaves the cold path
instead of being made efficient.  Strata's cache-aware scheduling remains the right design for the
durable tier, where the transcript and index do live.

**KV virtualisation for agent workspaces.**  KVMem [2] preserves overflowed workspace history as paged KV state across GPU, host and NVMe, indexes it with
model-native attention-space summaries (Mean-K over blocks), and materialises a *query-dependent
execution view* bounded by the model's native context window - 1M tokens of workspace on a 24 GB
consumer GPU for a 27B model, with DeepSWE task success improving from 43.8% under compaction-only
context management to 48.4%.  Two things are shared and we do not claim them: the idea of a
query-dependent view, and the observation that compaction is lossy.  Two things differ, and they
are the paper's subject; Table 13 states them in objects and in costs.

**Table 13:** The two closest systems, compared in objects and in costs. The KVMem column is as
reported in its paper.

| | KVMem [2] | this paper |
|---|---|---|
| what the view is assembled from | paged **KV blocks**, with raw keys re-rotated at new positions | **text** compiled from a model-independent index |
| what a cold route moves | KV blocks from host or NVMe | ~32 KB of state text; no model-specific state |
| what bounds the view | the model's native context window (256K for the model it evaluates) | the query and the current turn: **4-8K, measured** (a 3,584-token window inside a 4,096-token total holds the surface at -0.96 pp; the same total with a 3,072-token window does not) |
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
  capacity and routing consequences: a 32x-older session costing 2.4x less to move, 32x the
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
Section 4.3 measures that baseline under the same budget and the same window, with the summariser
written by the served model.  The honest result is that **compaction is equivalent here, not
worse**: fidelity 0.00 pp both ways, and decision 0.122 against 0.089 with a paired interval that
includes zero.  We therefore do not claim to beat
compaction, and the paper's contribution is not in that comparison - it is the bound and its
consequences.

What we do report against the *compiler* variant of the same idea is negative and specific: content
dedup alone is a statistical tie with plain retrieval, the two stages that actually compile state are
harmful (collapsing each file to its latest costs 8.5 pp of fidelity, re-selecting the newest
output's lines 20 pp), and the metric usually quoted for compaction - teacher-forced next-token
fidelity - is saturated by keeping the newest evidence whole, which is why a system can appear to
improve under it while nothing about the turn's evidence has changed.  Reporting that is what keeps
the positive claim honest: what
carries quality is the window and retrieval, and what carries mobility is the bound.

## 6. Limitations and non-claims

* **The decision advantage is a moderate-length phenomenon.**  At 64K-96K histories the bounded view,
  the compiler and the full transcript are one indistinguishable cluster (0.123 / 0.111 / 0.113);
  the metric separates arms at ~33K and does not beyond that.
* **The regime scope of the routing result is stated with it.**  Balanced and slow-worker wins
  assume a fleet whose warm cache holds a fraction of its sessions and no cluster KV store;
  against a KV-moving tier they reduce to worker loss and the 1M mixes.
* **The 1M lookup is measured, but on composed history.**  No public trace is that long, so the
  lookup is measured on real transcripts concatenated into million-token histories
  (`g2-index-lookup-composed-1m-v1.json`): p50 **0.4-4.4 ms** depending on query size, against
  53-203 ms of active-set prefill.  What is still composed is the *session*, not the measurement.
* **Histories past 156K tokens are composed.**  The transcripts are real and the final turn's patch
  is real, but a million-token session is twenty-five sessions concatenated (Table 7b); the corpus
  has no session that long, and the composition is labelled in the rows, the receipts and the table.
* **The 8B and 70B geometries are declared, not measured.**  The fabric, prefill and 0.5B KV numbers
  are measured on this card.
* **We do not claim `|E_q|` is intrinsically bounded.**  Lexical evidence mass grows with session
  length on these traces - 11K tokens at 51K-token histories, 46K at 103K, 195K at 321K, and **395K**
  of 764K at the long end (`g2-evidence-mass-composed-1m-v1.json`) - so at a million tokens more
  than half the history still shares terms with the query.  The bound is chosen, and its validity is
  the measured quality flatness at that budget, not a ceiling the corpus imposes.
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
32x the sessions, makes recovery and model rollouts the same operation, and moves the routing phase
boundary in every regime we model - while the obvious way to shrink the state further, a semantic
compiler, is measured not to pay.  The session keeps growing; the state that must move does not.

---

## References

The two systems Section 5 compares against are cited from their primary sources; the remaining
systems named in that section (SGLang, vLLM-LMCache, SPIN/SparseServe, llm-d) are named as
deployments rather than cited, and need citations before submission.

[1] **Strata** - KV caching across GPU HBM, host memory and SSDs, with a GPU-assisted I/O path and a
cache-aware scheduler, implemented in SGLang.  OSDI '26 presentation:
<https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang>.  Full bibliographic details to
be taken from the published paper.

[2] **KVMem** - virtualises million-token agent workspaces as paged KV across GPU, host and NVMe,
with a query-dependent bounded view.  arXiv:2609.04852, <https://www.alphaxiv.org/abs/2609.04852>.

## Appendix A. Claims, receipts and reproduction

### A.1 The claim map

Every claim in this paper maps to a receipt in `artifacts/` and to the script in `scripts/` that
produced it.  `benchmarks/paper_numbers.py` re-derives the headline numbers from the receipts and
asserts them (including the figures' own CSVs, so a figure cannot drift from its receipt);
`benchmarks/check_receipts.py` verifies that every receipt a document cites exists and is not a
partial write; `benchmarks/check_scripts.py` checks that every script's promised output exists;
`benchmarks/make_figures.py` regenerates the figures from the receipts.  The figures are stored as
`figures/figN_*.csv` (the numbers, each row traceable to a receipt) and `figures/figN_*.svg` (the
drawing), so a caption carries no path: figure 5 is a table-shaped CSV of the capacity numbers in
Table 10 and has no drawing.

Table 14 maps each claim to the receipt that backs it and to the script that produced it.

**Table 14:** Each claim, the receipt that backs it, and the script that produced it.

| claim | receipt | script |
|---|---|---|
| most of a session is state that is never needed again | `g2-dead-state-v1.json` | `benchmarks/g2_dead_state.py` |
| the state is bounded and the fidelity delta does not move with age | `g2-killer-table-v4.json` | `benchmarks/g2_killer_table.py` |
| the budget floor is 4,096 tokens (window 3,584) | `g2-compiler-window3584-b4096-v1.json`, `g2-compiler-window3k-b{4096,6144,12288}-v1.json` | `scripts/run_budget_lower.sh`, `scripts/run_floor_3584.sh` |
| the joint view ties the best retrieval arm (n=96) | `g2b-patch-localization-{windowcompiler-b8192-n96,raw-b4096-n96}.json` | `scripts/run_joint_n96.sh` |
| the stricter action-level end task is a bound, not a win | `g2b-action-metric-v1.json` | `benchmarks/g2b_action_metric.py` |
| compaction is equivalent at the same budget and window, not worse | `g2-compiler-compact-b8192-v1.json`, `g2b-patch-localization-compact-b8192-n48.json` | `scripts/run_compaction.sh` |
| the compiler's stages do not pay, with attribution | `g2-compiler-{consolidate,dedup,materialize,statefirst,protectwhole}-*.json` | `scripts/run_tail_rerun.sh` |
| the mobility law and the inversion | `g2-killer-table-v4.json`, G3 receipts | `benchmarks/g6_placement_inversion.py` |
| the routing phase change at an admissible state | `g4-quality-join-v2.json` | `scripts/run_g4_{grid2,geometry,pressure,hotspot}.sh` |
| capacity and recovery | `g5-capacity-planning-v1.json`, `g5-failover-two-workers-samemodel-v1.json` | `benchmarks/g5_*.py` |

### A.2 How these numbers are kept honest

Seven checks run against the repository rather than against the prose - five of them automated, and
the cold-start drill runs all of them in a fresh clone:

* **value audit** - `benchmarks/paper_numbers.py` re-derives every headline number from the
  receipts and asserts it - including that the routing cells which clear a strict
  retrieval-parity bar are exactly zero, and that the stricter end task's interval contains zero;
* **receipt audit** - `benchmarks/check_receipts.py` reports any cited receipt that is missing, any
  whose payload is a *partial* write (the harnesses write incrementally, so a killed or still-running
  run leaves a partial artifact at its final path), and any that exists only in the working
  directory.  It reads both citation styles - the manuscript and the ledger name receipts by bare
  filename, earlier documents by full path - and treats `make_figures.py` as a citing document too,
  because a figure whose input is untracked cannot be regenerated even though its CSV is committed;
* **cold-start drill** - all of the above run in a fresh clone, which is how the untracked class was
  found: eleven receipts that this paper or the figure generator cites existed only in the working
  directory, so a reader could neither check those claims nor regenerate those figures.  A clone now
  reports no missing, partial or untracked cited receipt, regenerates every figure byte-identically
  from the receipts, and passes the same value checks as the working tree;
* **semantic audit** - receipts record facts about what the arm actually did (for example
  `summariser_calls` per row), and the audit asserts them, because a value check cannot catch a
  receipt that does not do what its name claims.  This is not hypothetical: an earlier version of
  the compaction baseline in this paper reported a plausible number while never calling its
  summariser at all, and was withdrawn when the semantic check exposed it;
* **script audit** - `benchmarks/check_scripts.py` reads every reproduction script, extracts the
  `--out` paths it promises and reports which exist, which are partial, and which are missing;
  arms superseded by a later variant are listed with the reason, so anything else missing is an
  arm that never produced its receipt.  This is the check that answers "did this experiment
  actually run", which a queue cannot answer while it is still waiting;
* **figure regeneration** - `benchmarks/make_figures.py` and `make_svg_figures.py` rebuild every
  figure from the receipts, and re-running them leaves the tree unchanged.  A missing input is an
  error (non-zero exit) rather than a figure written with an arm silently dropped;
* **configuration audit** - a receipt records the flags that produced it, and
  `benchmarks/check_scripts.py --config-audit` lists the receipts that predate that block.  This
  closes the last gap: two different runs can leave receipts whose *numbers* look alike, and then an
  arm cannot be re-run, only believed.  It found a live instance in this repository - the compaction
  baseline was produced with a 6,656-token window while its runner script passed 3,072, and the two
  cannot be told apart from the fidelity column (both hold the surface) or from the view size alone
  (a 3,072 window with a 512-token summary and a 5,120 window with a 1,536-token one both cap the
  view at 6,656 tokens).  The receipt was matched to its configuration by re-running candidate
  configurations and comparing per-example view sizes, summariser call counts and scores; the script
  was corrected, and the re-run through it reproduced the previous receipt exactly - all 48 rows, the
  summary block and 223 summariser calls - while adding the configuration that had been missing.

Three failure modes were found while building this and are worth naming, because all three
produced *plausible-looking* evidence rather than errors.  A killed duplicate run left a **partial artifact
at its final path** (16 of 48 rows), and a value audit cannot see that - the numbers it checks are
real, just few.  And two arms were reported as queued for two rounds while **never having started**,
because one harness did not accept a flag they passed and the runner scripts had no `set -e`: an
experiment that never starts looks exactly like one that is still running.  The defences are the
partial-receipt check, the semantic checks (`summariser_calls`), and queues that stop on failure.
The third is the one the semantic audit itself missed for two rounds, and it is the sharpest of the
three: the compaction baseline *called* its summariser, counted the calls, and scored a view that
contained no summary at all - the builder walked the older spans newest-first, hit a single
20K-45K-token tool dump that did not fit its 8,192-token summary input, and broke out of the loop
before collecting any text, so the arm silently became a recency window.  Recording a mechanism's
*activity* is not the same as recording its *effect*: the check now asserts that the summary's tokens
are in the scored context (`summary_tokens_in_view`), not merely that the summariser ran.

The runner scripts that produced each family of receipts are in `scripts/`, so the path from a
claim to its evidence is a claim -> receipt -> script triple, and the paper states which of the
four things that triple cannot cover: histories composed out of real sessions, declared hardware
geometries, a simulated cluster replay, and unmeasured task success.  `docs/REPRODUCING.md` gives the CPU-only path that re-derives every
number and figure here from the receipts, the rules that derive the corpora from the downloaded
corpus, and the cold-start drill that checks the repository is self-contained.

What is *not* here is as deliberate: the 1M lookup is measured on composed histories rather than on
a real million-token session (no public trace is that long), the 8B/70B geometries are declared rather than measured, the fleet-level scheduler is a replay
over measured primitives rather than a deployment, and task success on a benchmark like DeepSWE is
not measured at all.  Those four sentences are the paper's honest perimeter, and none of them is
load-bearing for a claim above.
