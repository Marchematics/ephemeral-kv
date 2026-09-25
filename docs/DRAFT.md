# Draft — what the system state of a long-lived session is

> Historical.  This is the earlier draft of the framing; the manuscript is
> [`PAPER.md`](PAPER.md), which supersedes it wherever the two differ.

## Abstract (draft)

A long-lived LLM session is served today as if its KV cache *were* the session: the resident
object grows with the transcript, so placement, recovery and capacity planning all inherit the
session's age.  We measure what a turn actually needs and find that the object these decisions
depend on is bounded and independent of age.  On real coding-agent traces, an 8,192-token
execution state holds the next turn with a teacher-forced fidelity delta of 0.00 pp against the
full transcript and an end-task score statistically indistinguishable from the best measured
retrieval baseline, while the compiled state stays at 7-8K as the raw history grows from 64K to
156K tokens.  The consequence is a phase change rather than a speedup: a session 32x older costs
2.4x less to move (5.1x at a 4K state) because the transfer term leaves the cold path, one worker
holds 16x more sessions, recovery and a model revision rebuild the state from a durable index
instead of moving 12-128 GiB of KV, and a replay against measured hardware primitives advances
routing in every workload regime at the quality-admissible state size, where the original grid
advanced only in a 2K corner.

We also report what does *not* work, because it is the hypothesis this space would reach for
first.  A semantic compiler - content dedup, supersede-by-identity, snippet re-selection, log
replay into materialised file state - does not beat plain retrieval on the real task, and two of
its stages measurably hurt: collapsing a file to its latest state costs 8.5 pp of fidelity
against keeping the superseded views, and re-selecting the newest output's lines costs 20 pp.
What carries quality is weaker and more useful: keep the newest evidence verbatim (a ~3K window
is enough) and spend the rest of the budget on retrieval.  The metric the field uses to compare
long-context systems, teacher-forced next-token fidelity, cannot see any of this: on these traces
it is saturated by keeping the newest spans whole, which is why we report the end task alongside
it and say which one a system should be judged by.

## Introduction skeleton (draft)

1. **The problem is an identity, not a cost.**  Serving stacks treat history, execution state and
   local KV as one object; the resource model is `session cost ~ history footprint`.  Tiered
   context caches, retention policies and workspace virtualisation all optimise within that
   identity.
2. **The measurement that breaks it.**  A turn needs a bounded amount of state, and most of the
   durable footprint is state that was needed once: five spans hold 73-96% of a long session's
   tokens, while the turn-by-turn content a view walks is 4-27% (median 8%).  Compiling to a fixed
   budget therefore does not lose quality as the session grows, which is a different claim from
   "compression works".
3. **The consequence for systems.**  `dM/dL ~ 0`; the inversion (old/small cheaper than
   young/large); 16x capacity; recovery and rollout on foreign models; a routing phase change at
   the admissible state size.  Each is a receipt.
4. **The negative result.**  The obvious way to shrink the state is a semantic compiler, and it
   does not pay here - it is measured against plain retrieval on the same instances, with the
   per-stage ablation that says why.  The window plus retrieval is what pays.
5. **The metric lesson.**  Fidelity cannot compare these systems; the end task can.  Report both,
   and say which one a deployment should gate on.

This is the paper's central section in draft form.  Every number is a receipt in `artifacts/`;
the ledger (`CLAIMS.md`) carries the per-claim provenance and `PLAN.md` the gate structure.
Nothing here is a projection except where it is labelled one.

## 1. The abstraction

A serving stack that keeps a long-lived agent session on a worker conflates three things:

```text
durable history     the transcript, append-only, grows without bound
execution state     what the model needs for the *next* action
local KV            a device-local artifact of having computed the state once
```

Today they are one object: the KV cache *is* the session, so its size is the session's age, and
placement, recovery and capacity planning all inherit that.  The systems in this space optimise
within that conflation - tiered context caches move the object between memory levels, retention
policies decide how long to keep it, virtualisation maps it to host or NVMe and materialises a
view for a query.  The measurements below say the conflation is not necessary:

```text
durable session   = history + index          (append-only, model-independent, on the tier)
execution state   = compile(history, q)      (bounded, rebuilt on demand, never transferred)
local KV          = a disposable artifact    (dropped on eviction, rebuilt in 0.05-0.20 s)
```

The claim is not that the transcript can be shortened.  It is that the object a placement
decision depends on - the execution state - is a *function of the current action and the query*,
not of the session's age, and therefore that session age should stop predicting mobility cost.

**Why the conflation is so expensive here.**  A long coding-agent session is not a long
transcript.  Measuring the token share held by the largest spans of each long session
(`g2_dead_state.py`, receipt `artifacts/g2-dead-state-v1.json`):

| session tokens | spans | largest span | top-2 share | top-5 share | the rest |
|---:|---:|---:|---:|---:|---:|
| 156,207 | 45 | 45,051 | 58% | 93% | 7% |
| 155,997 | 61 | 45,416 | 58% | 94% | 6% |
| 123,210 | 29 | 55,320 | 90% | 96% | 4% |
| 141,168 | 65 | 56,514 | 80% | 92% | 8% |
| 127,126 | 83 | 28,261 | 44% | 73% | 27% |

Five blocks hold 73-96% of the tokens, and the turn-by-turn content a view actually walks is
4-27% (median 8%) of the footprint.  A system whose cost is the footprint is therefore paying to
retain and to move state that was needed once - which is the cheapest thing for a compiler to
drop and the most expensive thing for a cache to keep.

## 2. The compiler, and what its stages actually buy

`ephemeralkv/` implements the split: `DurableSpanIndex` is the append-only model-independent
index over the transcript (`spans_of_kind`, postings, provenance), `statecompile.py` replays the
event log into current state, and `consolidate.py` holds the supersede and dedup semantics.

An executable-state compiler has to have system semantics, not a summarisation habit.  Ours are:

| stage | semantics |
|---|---|
| recency window | the current turn is not optional: the newest evidence is admitted whole |
| content-identity dedup | one copy of repeated tool output and repeated reasoning |
| supersede by state identity | a file's or a command's newest state stands for its earlier views; replaced turns are recorded as provenance |
| identifier provenance | pull spans sharing paths/tool ids with the query, on top of the lexical ranking |
| log replay | a dump *sets* a file's content, a diff or SEARCH/REPLACE block *applies* to it, a repeated command collapses to its latest result |

Measured at an 8,192-token view on the same examples, one change per row:

| configuration | fidelity (32K-128K, n=44) | end-task F1 (n=24) |
|---|---:|---:|
| raw query ranking (dedup + snippet), 3 truncated recent spans | -6.94 pp | 0.237 |
| + supersede by state identity, no path collapse | -5.44 pp | **0.292** |
| same, but collapsing each path to its latest state | **-13.90 pp** | - |
| log replay into the materialised state | -7.21 pp | 0.231 |
| state-first selection (states before chatter) | -15.38 pp | 0.242 |
| **the window kept whole, ranked evidence filling the rest** | **0.00 pp** | 0.139 |
| plain recency, no compiler at all | +0.29 pp | 0.154 |

The attribution is exact, which matters more than the ranking.  Content-identity dedup on its own
is **neutral**: 0.239 against plain retrieval's 0.243 on the same 48 instances (paired -0.004,
95% CI [-0.024, +0.016], 1 win / 2 losses).  What hurts are the two stages that actually *compile
state* - collapsing each path to its newest view, and re-selecting the newest output's lines.

Two of these are negative results the paper should keep, because they bound what "executable
state" can mean here.  Collapsing a file to its latest state *costs* 8.5 pp against keeping the
superseded views: a later view does not contain the region an earlier one showed, and the next
turn often needs exactly that.  And log replay buys nothing on the decision (0.231 against 0.237)
because only **6%** of the retrieved spans are file events - 74% are loose text and 14% command
results, so there is almost no state for a log to materialise.  The semantics that pay on this
workload are *supersede by identity* and *provenance*, and the reason is a property of coding-agent
traces rather than of the compiler.

## 3. Two laws, and why one view cannot hold both at 8K

| view (8,192 tokens) | fidelity | decision F1 | state |
|---|---:|---:|---:|
| full history | reference | 0.045 (n=24) / 0.089 (n=48) | unbounded |
| plain recency (newest spans whole) | **+0.29 pp** | 0.154 | 8,286 |
| window 35% + compiled far field | 0.00 pp | 0.056 | 6,832 |
| window 50% + compiled far field | **0.00 pp** | 0.139 | 7,141 |
| window 60% + compiled far field | 0.00 pp | 0.104 | 6,894 |
| evidence consolidation (fully ranked) | -5.44 pp | **0.292** | 8,372 |
| protected spans kept whole + ranked rest | -20.33 pp | 0.211 | - |
| newest span only + compiled far field | -25.36 pp | 0.083 | - |
| 16,384-token reference view | -1.94 pp | 0.142 | 16,384 |

1. **Fidelity is set by how the newest evidence is treated, not by how much is kept.**  Kept
   whole it is at parity with as little as 2.9K of window; prefix-truncated it costs 5-7 pp;
   reordered into a query-selected excerpt it costs 20 pp; replaced by a single span 25 pp.
2. **The decision is set by how much *ranked* evidence the budget holds.**  A fully ranked 8K view
   scores 0.292 and 0.237; a ranked view squeezed to the remainder after a window scores
   0.056-0.158; full history - all the evidence, unranked - scores 0.045.

**The two are orthogonal, which the receipts now say explicitly.**  In the end-task
configuration, the window-plus-retrieval view and plain retrieval at the same 8,192-token budget
produce **byte-identical contexts on every one of the 48 instances** (`active_tokens` equal on
48/48, and a paired decision difference of exactly 0.000 with no wins and no losses) - because the
retrieval arm already protects its three newest spans, so the window adds no new evidence, it only
changes how much of each span is rendered.  Where the newest span is larger than its protection
room the rendering differs and the fidelity differs with it (0.00 pp against -6.94 pp) while the
decision does not move at all (0.191 either way).  So:

* the **decision** is set by *which evidence is retrieved*, not by how it is rendered;
* the **fidelity** is set by *whether the newest evidence is rendered whole*, not by what else is
  in the view.

One caveat keeps this from being a free lunch: the far field must be *consolidated* rather than
prefix-truncated raw dumps.  The same 3,072-token window with raw retrieval behind it scores
**-6.94 pp** on fidelity - the truncated dumps in the far field hurt the surface even though the
newest evidence is whole - while a consolidated far field is at 0.00 pp.  On the decision the two
are tied (+0.029, 95% CI [-0.089, +0.146]), so consolidation is the better far field *because it is
fidelity-neutral at no measurable decision cost*, not because it wins anything.

A system that renders the newest evidence whole and consolidates the rest therefore gets both at
one budget, and that is the configuration measured here: **0.161 decision with 0.00 pp fidelity**,
statistically tied on the decision with every other bounded arm measured - and more budget does not help: the same arm scores 0.191 at 8,192 and 0.173 at 12,288, so above ~4K the constraint is retrieval coverage rather than budget, which is why the paper's view is 8,192 tokens - plain retrieval at
4,096 (0.243, paired +0.081, CI [-0.016, +0.177], 13 wins / 3 losses), window plus raw retrieval
(0.191), dedup only (0.207-0.239) - while those arms carry -25 pp to -6.94 pp of fidelity.  The honest
requirement is therefore the smallest budget at which both hold, and that is what the
12,288-token arm measures (16,384 is the pre-compiler reference, already measured at -1.94 pp).

Paired on one set of 48 sessions, the comparisons that matter are blunt: full history scores
0.089, plain retrieval 0.243 at 4,096 tokens and 0.149 at 2,048, and **the compiler 0.139 at 8,192
- i.e. plain retrieval at half the budget beats the compiler at the full budget** (paired -0.104,
95% CI [-0.208, -0.007], 6 wins / 12 losses).  The earlier, favourable comparison came from a
different instance set (0.292 against 0.237 on the first 24 sessions) and does not replicate.
Per-instance inspection gives the mechanism: on several instances the compiler returns 0.00 where
plain retrieval returns 1.00 or 0.62, because supersede-by-identity and snippet selection drop the
file the patch touches - the same effect the path-collapse ablation measured from the fidelity
side (-13.90 pp against -5.44 pp).

**The damage is systematic, not a configuration.**  Holding the design fixed and varying only the
far field, on the same 48 instances:

| window | far field | 8,192 | 12,288 |
|---|---|---:|---:|
| 2.9-4.1K | plain retrieval | **0.191** | (measuring) |
| 3.1K | compiled | 0.084 | 0.133 |
| 4.1K | compiled | 0.161 | 0.155 |
| none | compiled | 0.139 | 0.213 |
| none | plain retrieval | 0.243 at 4,096; **0.191 at 8,192** | - |
| 3.1K | plain retrieval | 0.191 | 0.173 |
| 3.1K | plain retrieval | 0.191 | 0.173 |
| none | plain retrieval, 64K-96K histories | +0.011 paired (tie: 0.123 vs 0.113 full history) | - |

Every arm whose far field is the compiler sits below the same-budget plain-retrieval arm, across
two window sizes and two budgets, and the two stages responsible are identified separately
(path collapse 8.5 pp of fidelity, snippet re-selection 20 pp).  At this point "our compiler" is
not a claim the paper can make; "the obvious compiler does not pay, and here is the measurement"
is one.

**What does hold both, at one budget and with no compiler.**  A view that keeps the newest
evidence verbatim (a window) and spends every remaining token on plain retrieval scores end-task
F1 **0.191** against **0.243** for plain retrieval at 4,096 and **0.161** for the same window with
the compiler - and those three arms are statistically indistinguishable on the decision (paired
-0.052, 95% CI [-0.154, +0.047], 5 wins / 7 losses on 48 sessions), while their fidelity separates
them sharply: 0.00 pp for the windowed arms, -25 pp for plain retrieval at 4,096, -5.44 pp for the
compiler at 8,192.  The honest statement is therefore not that the windowed view wins - it is that
it **ties the best decision measured while removing the fidelity loss entirely**, at the same
8,192-token budget, with no semantic compiler in it.

**So the compiler as implemented does not earn its place on either metric**, and the paper says
so.  What the measurements support is narrower and still worth a paper: the execution state is
*bounded* (a fixed budget holds quality flat as the session grows), the surface needs the newest
evidence untouched, the decision needs ranked evidence, and which of the two a system prioritises
decides how small its state can be.  The negative compiler result is part of the contribution -
it is what rules out "compression" as the explanation for the mobility law.

## 4. The mobility law, measured

Bucketing the G2 receipts by session length (`g2_killer_table.py`), at a fixed 8,192-token budget:

| raw history p50 | execution state p50 | fidelity delta |
|---:|---:|---:|
| 64,401 | 7,133-7,829 | +0.00 pp |
| 82,972 | 7,133-7,899 | +0.00 pp |
| 155,574 | 8,203 | +1.43 pp |

Across a 2.4x growth in history the state grows 5% and the quality delta does not move.  On the
axis the claim is usually stated in - session age in turns - the same table reads:

| turns p50 | raw history p50 | execution state p50 | fidelity delta |
|---:|---:|---:|---:|
| 40 | 79,769 | 8,121 | -1.43 pp |
| 58 | 84,617 | 7,887 | **+0.00 pp** |
| 96 | 155,222 | 8,203 | +0.00 pp |

Measured to 96 turns because that is where this corpus ends.  The corpus also holds one 22-turn
example with 67K of history in which the arm fails badly (-46.9 pp); it is a singleton, reported
as one rather than smoothed into the table, and the failure mode it hints at - a short session
whose individual turns are enormous - is a limitation worth stating.

Composed with the measured hardware primitives:

| session | state | mobility (lookup + active prefill) | full-KV move |
|---|---:|---:|---:|
| 32K, raw 16K view (the pre-compiler requirement) | 16,384 | 0.4855 s | 0.0173 s |
| **1M, compiled** | **8,192** | **0.2031 s** | 0.5530 s |
| **1M, compiled** | **4,096** | **0.0954 s** | 0.5530 s |
| 262K, compiled | 8,192 | 0.2031 s | 0.1383 s |

A session 32x older is 2.4x cheaper to move (5.1x at a 4K state).  Session age stops predicting
placement cost; the compiled state does.  The sentence the paper wants is the plain one: **the
session keeps growing; the state that must move does not.**

## 5. The routing consequence, with the quality attached

`g4_quality_join.py` refuses to report the advancing region without the quality criteria.
Across three grids (measured primitives; the model geometry and the arrival model are declared in
each receipt):

| active set | balanced | slow worker | worker loss | total |
|---|---:|---:|---:|---:|
| 2,048 | 2/22 | 2/22 | 13/22 | 17/66 |
| 4,096 | 8/66 | 11/66 | 44/66 | 63/198 |
| **8,192 (fidelity-admissible)** | **5/82** | **6/82** | **25/82** | **36/246** |
| 16,384 | 0/22 | 0/22 | 2/22 | 2/66 |

118 of 576 cells advance, 36 of them at a size where fidelity is within tolerance and the
decision clears raw retrieval.  The scope is part of the claim: the balanced and slow-worker wins
are the capacity-pressure regime where the fleet's warm cache holds a fraction of its sessions
and no cluster KV store exists, so the baseline's alternative is a full re-prefill (1.6-14.3 s)
against 0.203 s of rematerialisation; against a tier that can move KV the wins are worker loss and
the 1M-history mixes.  The 4,096 column advances widely but is not fidelity-admissible (-25 pp).

## 6. What this paper must not claim

* not that the compiler beats plain retrieval on the decision metric - it ties, at a resolution
  this corpus cannot improve;
* not that one 8,192-token view holds both metrics - it does not, and the paper reports the
  smallest budget that does;
* not that summarisation is the mechanism - the mechanisms that pay are the verbatim window and
  supersede-by-identity, both of which are measurements, not intuitions;
* not that ephemeral mobility beats moving KV in general - it beats it where the payload is large
  or the session must move, and the phase boundary is in the paper rather than in a footnote;
* not that the 1M lookup row is measured - no public trace is that long, and the receipt says so.
