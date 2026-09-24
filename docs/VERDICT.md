# Verdict — is Paper B on a Best path, and what exactly is claimed

Internal decision document.  Every number is a receipt in `artifacts/`; the ledger is
`CLAIMS.md`, the gate structure `PLAN.md`, the draft section `DRAFT.md`.  Written so that the
go/no-go call does not depend on anyone's memory of the last few rounds.

## 1. The claim set the measurements support

```text
durable session   = transcript + model-independent index       (grows without bound)
execution state   = compile(history, q)                        (bounded, rebuilt on demand)
local KV          = a disposable artifact of having computed it (never transferred)
```

What is measured for that abstraction:

| statement | evidence |
|---|---|
| the state that must move is bounded and does not grow with session age | raw history 64,401 / 82,972 / 155,574 -> state 7.1-8.2K, fidelity delta +0.00 / +0.00 / +1.43 pp (`g2-killer-table-v3.json`) |
| therefore session age stops predicting placement cost | 1M history with an 8,192-token state: 0.2031 s; with 4,096: 0.0954 s; against 0.4855 s for a 32K history carrying the 16,384-token raw view that fidelity needed before the bound - a session 32x older is 2.4x (5.1x) cheaper to move (`g2-killer-table-v3.json`, G3 primitives) |
| most of the durable footprint is state that is never needed again | five spans hold 73-96% of a long session's tokens; the turn-by-turn content is 4-27% (median 8%) (`g2-dead-state-v1.json`) |
| the changed cost law moves the routing decision at a quality-admissible state | 118/576 replay cells advance; **36 at the 8,192-token state**, in all three regimes (balanced 5, slow worker 6, worker loss 25) (`g4-quality-join-v1.json`) |

## 2. What is *not* claimed, because it was measured false

| hypothesis | measurement |
|---|---|
| a semantic compiler beats plain retrieval on the real task | **false**: on 48 paired sessions plain retrieval at 4,096 scores 0.243 and the compiler at 8,192 scores 0.139 (paired -0.104, 95% CI [-0.208, -0.007], 6W/12L).  Per instance the compiler returns 0.00 where retrieval returns 1.00 or 0.62, because supersede-by-identity and snippet selection drop the file the patch touches |
| a compiler is needed for the fidelity bound | **false**: the bound comes from keeping the newest evidence untouched.  Plain recency is at parity (+0.29 pp) and a 2.9K verbatim window inside an 8,192-token view is at 0.00 pp |
| teacher-forced next-token fidelity can compare systems here | **false on this corpus**: it is saturated by recency (8,192 tokens of newest spans = parity), so it cannot see a compiler at all |
| collapsing a file's history to its latest state is good state management | **false**: it costs 8.5 pp of fidelity against keeping the superseded views (-13.90 pp against -5.44 pp) |
| log replay into materialised state pays | **no**: 0.231 against 0.237 on the decision, because only 6% of retrieved spans are file events (74% loose text, 14% command results) |
| one 8,192-token view holds both metrics | **it does, if the window is verbatim and the rest is retrieval**: window + plain retrieval scores 0.191 against 0.243 for plain retrieval at 4,096 - a statistical tie (paired -0.052, CI [-0.154, +0.047]) - while the windowed view is at 0.00 pp fidelity and plain retrieval at 4,096 is at -25 pp.  The compiler is not what buys this |

## 3. The five hard results, scored

| hard result | threshold | measured | verdict |
|---|---|---|---|
| semantic compiler 16K -> <=8K | 4-8K | fidelity: the 8,192 view is at parity, but through the window, not the compiler (the compiler alone is -5.44 pp) | **met by the view, not by the compiler** |
| fidelity vs full history | <=2 pp | +0.29 pp (recency), 0.00 pp (2.9-4.9K window) | **met** |
| real agent task not regressing, ideally > raw retrieval | no regression | all arms beat full history (0.089 -> 0.139-0.243); against raw retrieval the compiler *loses* | **no regression met; "better than raw" failed** |
| G4 winning region wider than 4/48, not only the 2K extreme | material widening | 4/48 -> 118/576, with 36 cells at the quality-admissible 8,192 across all three regimes | **met** (the widening came from the grid's missing axes - active-set granularity, session scale, model geometry, capacity pressure - not from the compiler) |
| cluster consequence >=1.5x goodput or >=30% p99 in several regimes | two or more regimes | capacity-pressure regime: goodput x2-5, p99 +6-26% in balanced and slow-worker; worker-loss cells advance against a KV-moving tier; balanced with all sessions resident and a KV-moving tier does not advance | **met with a stated scope** |

## 4. The deeper criterion

```text
M(q, L) ~= f(|E_q|)          with  dM/dL ~ 0
old session, small state  <  young session, large state
```

* `dM/dL ~ 0` **is measured** on the state side: the compiled state is flat (7.1-8.2K) while the
  raw history grows 2.4x, and the fidelity delta does not move with it.
* the inversion **is measured** in cost: 1M/8K = 0.2031 s against 32K/16K = 0.4855 s.
* the ranking consequence is now measured as arithmetic over the same primitives
  (`g6_placement_inversion.py`, `artifacts/g6-placement-inversion-v1.json`).  With a 4,096-token
  state the oldest session in the set (1M history) is the **cheapest** to place cold at 0.0954 s,
  while the youngest carrying the 16,384-token state that fidelity needed before the bound costs
  **0.4855 s** - and the footprint estimate ranks them exactly backwards (5.899 s against
  0.184 s).  Across bounded-state sessions the true cost is flat at 0.2028-0.2031 s for histories
  from 32K to 1M, so a scheduler ranking by history is ranking by a quantity uncorrelated with
  what it pays.

## 4a. The capacity consequence (derived from measured state sizes and KV geometry)

If the resident object is the history, sessions per worker = budget / (history x B/token) and the
arithmetic is brutal.  If it is the execution state, the same budget holds states whose size does
not depend on age.  Usable HBM 20 GiB per worker (`g5_capacity_planning.py`,
`artifacts/g5-capacity-planning-v1.json`):

| model geometry | resident history, 128K session | resident history, 1M session | resident compiled state, 8,192 |
|---|---:|---:|---:|
| Qwen2.5-0.5B (12,288 B/token, measured) | 13.3 sessions | 1.7 | **213** |
| 8B-class (131,072 B/token) | 1.2 | 0.2 | **20** |
| 70B-class (327,680 B/token) | 0.5 | 0.1 | **8** |

Sixteen times the sessions at 128K, two orders of magnitude at 1M, and on real model geometries
the full-history object does not fit at all (0.1-0.5 sessions per worker).  This is also what
makes the G4 pressure sweep's "warm cache holds a fraction of the fleet" regime concrete rather
than arbitrary: one or two resident histories per worker is what an 8B-class fleet with 128K
sessions actually looks like, while sixteen resident states is what the bound allows.

## 4b. What recovery actually moves (derived, labelled as derived)

The state sizes are measured and the KV geometry is measured, so the data-movement comparison is
arithmetic rather than a new experiment:

| recovery of a 1M-token session | bytes moved | time |
|---|---:|---:|
| move the KV cache (Qwen2.5-0.5B geometry, 12,288 B/token) | 12.00 GiB | 0.5530 s at the measured 23.3 GB/s |
| move the KV cache (8B-class geometry, 131,072 B/token) | 128 GiB | 5.9 s |
| rebuild from the durable index: 8,192 tokens of state text (~4 B/token) + one lookup | ~33 KiB | 0.2031 s prefill + 0.4 ms lookup |

Five orders of magnitude less data, and the direction is the point: the transfer term disappears
from the recovery path entirely, so a model or adapter revision that invalidates all old KV costs
nothing extra - which is the property that makes rollout and failover the same operation.

## 5. Decision

* **Do not ship "we built a semantic compiler".**  The compiler's stages are measured not to pay,
  and two of them to hurt.  Shipping that claim would be exactly the incremental version the
  competitive landscape already covers.
* **Ship the state object, the law and the phase change**, with the compiler ablations as a
  negative result that bounds what compilation can mean on coding-agent traces.
* Paper B remains **Best-targeted** on that basis - the abstraction is the identity, and the
  measurements support it - but the Best case now rests on four legs, all measured: a bounded
  state, `dM/dL ~ 0`, the cost inversion, and a routing phase change at a quality-admissible
  state.  The "compiler" leg is removed.
* **If** the fixed-window sweep (3,072-token window plus ranked evidence at 8,192 and 12,288)
  shows a single bounded view holding both metrics, the paper gains the "one view" result without
  a compiler and the identity gets stronger.  If it does not, the paper states the frontier - the
  surface needs a fixed window, the decision needs the ranked budget, and a system chooses which
  to prioritise - which is a finding, not a gap.

## 6. What remains, in order

1. the fixed-window sweep (in flight) - the last quality cell;
2. the age-aware routing trace (the inversion as a scheduler decision);
3. G5 at the measured operating point: **done for the model-rollout half** - on two models that
   never saw the sessions, the index-built 4,096-token state scores end-task F1 0.137 against
   0.017 (Qwen2.5-0.5B) and 0.145 against 0.042 (Qwen2.5-1.5B), so recovery on a foreign model
   keeps the decision; what remains is a two-worker failover with a killed owner and a scheduler
   that re-places the session;
4. the paper: abstraction, law, phase change, negative compiler result, and the metric lesson.

## 7. The one-page ledger

Paper-ready claims, each with its receipt and its limit.  A claim without a receipt does not go in
the paper; a claim whose limit is not stated here does not go in the abstract.

| # | claim | receipt | limit |
|---|---|---|---|
| C1 | the execution state is bounded and flat in session age | killer table: state 7.1-8.2K at raw histories 64K / 83K / 156K, fidelity delta +0.00 / +0.00 / +1.43 pp | 44 long-history examples; the corpus reaches 156K, the 1M row is composed from primitives |
| C2 | fidelity parity needs only ~3K of verbatim newest evidence | window sweep 2.9K-6.1K -> 0.00 pp at both 8,192 and 12,288 budgets | teacher-forced metric, whose recency saturation is itself C6 |
| C3 | one 8,192-token view holds both qualities, and the two requirements are orthogonal | window + plain retrieval: 0.191 decision against 0.243 for plain retrieval at 4,096 (paired -0.052, CI [-0.154, +0.047]) with 0.00 pp fidelity against -25 pp; in the end-task configuration the window view and plain retrieval at 8,192 are **byte-identical on 48/48 instances** (paired decision difference exactly 0.000, 0 wins 0 losses), so the decision is set by *which* evidence is retrieved and the fidelity by *whether the newest evidence is rendered whole* | a tie on the decision, not a win; 48 paired sessions, 26 scoreable; the byte-identity holds in the end-task configuration (snippet on, 25% protection), while the fidelity split was measured in the fidelity configuration |
| C4 | the obvious compiler does not pay, and two of its stages hurt | same-instance arms across two window sizes and two budgets; path collapse -8.5 pp of fidelity; snippet re-selection -20 pp; log replay neutral because only 6% of retrieved spans are file events | corpus-specific: coding-agent traces with few file events |
| C5 | mobility law and inversion | 1M with a 4,096 state: 0.0954 s; with 8,192: 0.2031 s; against 0.4855 s for 32K carrying the 16,384-token pre-compiler view; flat 0.2028-0.2031 s from 32K to 1M | composed from G3 measured primitives; the 1M lookup row is extrapolated |
| C6 | session age ranks placement cost backwards | 1M/4K ranks cheapest (0.0954 s) and 32K/16K most expensive (0.4855 s); the footprint estimate ranks them 5.899 s against 0.184 s | arithmetic over the same primitives |
| C7 | the cost law moves routing at a quality-admissible state | 118/576 replay cells advance; 36 at the 8,192 state where both metrics hold, in balanced, slow-worker and worker-loss regimes | balanced and slow-worker wins are the no-cluster-KV-store regime |
| C8 | capacity: the same worker holds far more sessions | 213 / 20 / 8 states against 13.3 / 1.2 / 0.5 histories per 20 GiB, at 128K, on 0.5B / 8B / 70B geometries | 8B and 70B geometries are declared, not measured |
| C9 | recovery and rollout without moving KV | on two models that never saw the sessions: end-task 0.137 / 0.145 against 0.017 / 0.042; ~33 KiB of state text against 12-128 GiB of KV; 0.55-1.04 s against 6.31 s of re-prefill | the model-rollout half; a two-worker failover with a killed owner is not measured |
| C10 | the field's proxy metric cannot compare these systems | teacher-forced fidelity is at parity with 8,192 tokens of recency and with a 2.9K window, and falls 5-25 pp only when the newest evidence is truncated, reordered or replaced | the end task is the metric we gate on; both are reported for every arm |

Measured false, and therefore not claimed anywhere: a semantic compiler beating retrieval; a
compiler being required for the bound; teacher-forced fidelity separating systems on this corpus;
collapsing a file to its latest state; log replay paying; one 8,192-token view holding both metrics
*through compilation*.

## 8. Status of the decision

**Paper B is Best-targeted on C1-C10 and has dropped the compiler claim, which the measurements
falsified rather than left unproven.**  What the Best case rests on is the identity (three
different objects, not one), the law (flat state, flat quality), the inversion (age ranks cost
backwards), the capacity consequence, the routing phase change at the admissible size, and
recovery that never moves KV - with the compiler ablation as the negative result that rules out
the explanation everyone reaches for first.  Remaining before writing: the window-size isolation
runs (3,072-token window with plain retrieval at 8,192 and 12,288), the equal-budget retrieval
reference at 8,192, the dedup-only compiler arm, and the 64K-128K decision column of the killer
table.
