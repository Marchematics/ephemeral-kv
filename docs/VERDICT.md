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
| one 8,192-token view holds both metrics | **not with the arms measured**: fidelity parity costs the ranked budget the decision needs (best fidelity-admissible arm 0.161 against plain retrieval's 0.243 at half the budget) |

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
* what is *not* measured is the scheduler consequence of the inversion as a separate experiment:
  the G4 replay uses the cost law, so the inversion is inside it, but no artifact yet shows a
  policy choosing to move an old session *because* it is old.  That is a small, honest experiment
  (a routing trace with per-session age and state size, comparing a size-blind policy against a
  size-aware one) and it is the cleanest remaining Best-shaped result.

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
3. G5 at the measured operating point: worker loss and a model rollout resumed from the durable
   index, reporting recovery cost and data moved (the primitive is measured; the end-to-end
   recovery receipt is not);
4. the paper: abstraction, law, phase change, negative compiler result, and the metric lesson.
