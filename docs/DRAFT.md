# Draft — what the system state of a long-lived session is

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

At 8,192 tokens the two cannot both hold, and the reason is arithmetic rather than algorithmic:
the surface needs the newest output untouched and in order, the decision needs the remaining
budget ranked, and the newest output can be large enough that there is no remainder.  The honest
requirement is therefore the smallest budget at which both hold, and that is what the
12,288-token arm measures (16,384 is the pre-compiler reference, already measured at -1.94 pp).

Paired over the same sessions (`paired_f1.py`), the decision claims that survive are: the compiler
against windowed views (`+0.188`, 95% CI `[+0.062, +0.326]`, 9 wins / 1 loss) and against full
history (`+0.248` at 8,192 on 24 instances); the compiler against plain retrieval is a tie
(`+0.055`, `[-0.107, +0.220]`) at a difference SD near 0.40, i.e. ~400 paired sessions would be
needed to resolve it.  The paper states the established comparisons and no more.

## 4. The mobility law, measured

Bucketing the G2 receipts by session length (`g2_killer_table.py`), at a fixed 8,192-token budget:

| raw history p50 | execution state p50 | fidelity delta |
|---:|---:|---:|
| 64,401 | 7,133-7,829 | +0.00 pp |
| 82,972 | 7,133-7,899 | +0.00 pp |
| 155,574 | 8,203 | +1.43 pp |

Across a 2.4x growth in history the state grows 5% and the quality delta does not move.  Composed
with the measured hardware primitives:

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
