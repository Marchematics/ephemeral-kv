# Claim ledger

A row is a **measured claim** only when it has a matched baseline and stored artifact.
Accounting and design properties are labelled separately.

## Established / recorded constraints

| # | statement | evidence | status |
|---|---|---|---|
| A1 | Under the declared G1 cost model, active recompilation beats full re-prefill but does not beat DRAM/NVMe KV movement by 512 turns | `benchmarks/kill_gate_crossover.py` | accounting only |
| A2 | Under the declared matched-cost G3 model, full-KV transfer/re-prefill grows 32x from 32K to 1M while fixed-4K active rematerialization stays flat; 1M/2K is cheaper than 32K/16K in the active-state model | `artifacts/g3-mobility-accounting.json` | accounting only |\n| A3 | The predicted crossover is a phase boundary, not a universal win: faster fabrics push it later, larger KV/token pushes it earlier, and smaller active sets push it earlier | `artifacts/g3-phase-space-accounting.json` | accounting only |
| N1 | "text is source / KV is rebuildable artifact" is not sufficient novelty | `docs/NOVELTY.md` | overlap audit |
| N2 | "bounded active KV working set" is not sufficient novelty | `docs/NOVELTY.md` | overlap audit |

## Open measured claims

| # | question | required evidence | gate |
|---|---|---|---|
| Q1 | Is next-turn working state sparse and indexed lookup sublinear on real agent traces? | structural trace receipt exists; frozen-LM quality runner added; end-task quality still required | G2 |
| Q2 | Is the cold-route mobility tax <=1.25x from 128K to 1M at fixed active set? | hardware primitive runner added; end-to-end real model timing still required | G3 |
| Q3 | Can 1M/2K be cheaper to move than 32K/16K on hardware? | matched remote-route measurement | G3 |
| Q4 | Does replacing history-sized cold-route cost with measured active-set cost change the best routing decision under realistic skew/failure? | multi-worker p99/SLO goodput | G4 |
| Q5 | Can a worker/model be replaced without transferring history-sized model state? | failover/rollout experiment | G5 |

No row above Q1--Q5 is a result yet.
