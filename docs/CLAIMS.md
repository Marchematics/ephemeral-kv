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
| Q1 | Is next-turn working state sparse and indexed lookup sublinear on real agent traces, *and* does the retrieved view preserve the turn? | Split, and the split is the finding. **Sparsity: yes.** On the 15,000-session public corpus the lexical/provenance index selects a median 12% of a 32K-128K history and its active fraction falls as history grows; on the longest sessions (median history 82,440 tokens, max 156,137) a 4,096-token active view is **5.0%** of history. **Fidelity: not at that size.** Teacher-forced next-turn scoring with a frozen Llama-3.2-1B shows a dose-response to the active budget - token-accuracy delta -10.1 pp (4,096 tokens), -6.0 pp (8,192), -0.26 pp (16,384) in the 8K-32K bucket - and at long histories 4,096 tokens costs -8.3 pp, i.e. the loss is **flat in history** but above the 2 pp tolerance until the view is ~16K tokens, which is 82% of an 8K-32K history. The gate therefore returns `kill` for small fixed budgets and `advance` only when the view approaches the history size; end-task (not teacher-forced) quality is still required | `artifacts/g2-thoughtworks-structural-v1.json`, `artifacts/g2-model-quality-llama1b-long-v1.json`, `-b8192`, `-b16384`, `-longhist-b4096` | G2 |
| Q2 | Is the cold-route mobility tax <=1.25x from 128K to 1M at fixed active set? | hardware primitive runner added; end-to-end real model timing still required | G3 |
| Q3 | Can 1M/2K be cheaper to move than 32K/16K on hardware? | matched remote-route measurement | G3 |
| Q4 | Does replacing history-sized cold-route cost with measured active-set cost change the best routing decision under realistic skew/failure? | multi-worker p99/SLO goodput | G4 |
| Q5 | Can a worker/model be replaced without transferring history-sized model state? | failover/rollout experiment | G5 |

No row above Q1--Q5 is a result yet.
