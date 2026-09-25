# Anticipated objections, and what answers them

Written from the reviewer's side.  Each row is an objection strong enough to sink the paper if
unanswered, the answer, and the receipt the answer rests on.  If an objection has no receipt, the
claim it targets does not belong in the paper - and two of them were removed for exactly that
reason (the withdrawn compaction number, and the compiler claim the project started from).

| # | objection | answer | receipt |
|---|---|---|---|
| 1 | "Next-token fidelity is not task success; you are grading yourself on a proxy." | Agreed, and the paper says so: the end task is the primary metric and fidelity is reported as a constraint. The end task is file-level localisation against the recorded patch, with an offline *action-level* rescoring as a stricter variant (does the turn intend to edit a recorded file). Task success - what KVMem reports as DeepSWE 43.8% -> 48.4% - is **not measured here** and is listed as a limitation rather than approximated. | `g2b-action-metric-v1.json`; §4.3, §5, §6 |
| 2 | "8,192 tokens is not small; a 256K-window model can hold your whole state trivially." | The budget is not asserted, it is measured: a 4,096-token total with a 3,072-token window scores -2.40 pp of fidelity, 4,608 with a 3,584-token window holds 0.00 pp, and above that neither metric improves. The comparison against compaction is run at **equal budget and equal window**, so the state size is not doing hidden work. | `g2-compiler-window3k-b{4096,6144,12288}-v1.json`; `g2b-patch-localization-{window6656,compact}-b8192-n48.json` |
| 3 | "Your compiler baseline is a strawman - real systems summarise better than that." | The uncomfortable answer for the paper's original story: on the decision, compaction is **not worse** - it ties the same-window retrieval arm (paired interval includes zero), and the arm's receipt records that it called its summariser (223 calls / 395 reuses on the fidelity arm). The paper therefore claims no win over compaction. Scope: one summariser family (the served model writes the summaries), one split. | `g2-compiler-compact-b8192-v1.json`; the paired tests in §4.3 |
| 4 | "Your negative compiler result is probably an implementation artefact." | The ablation is stage-by-stage on identical instances: dedup alone ties plain retrieval (+/- 0.02), and the two harmful stages are identified by name with their costs (path collapse -8.5 pp of fidelity, snippet re-selection -20 pp). An earlier *bug* in the compaction arm did produce a false negative result, which is why receipts now record what the arm actually did (`summariser_calls`) and the audit asserts it. | `g2-compiler-consolidate-b8192-{,nopath-}v1.json`; `benchmarks/paper_numbers.py`; §A.2 |
| 5 | "Your routing result is simulation, not deployment." | Correct, and stated in the claim: arrival models and model geometries are declared, all costs are measured hardware primitives, and the region where the baseline wins is published alongside (balanced load with every session resident against a KV-moving tier does not advance). The only real distributed measurement is the process-level failover. | `g4-quality-join-v2.json`; `g5-failover-two-workers-samemodel-v1.json`; §6 |
| 6 | "One corpus, one model family." | True: SWE-style agent traces, Llama-3.2-1B for quality and Qwen2.5 for primitives and cross-model recovery. The structural claims (dead state, bounded state) are stated as properties of *these* traces, and cross-checkpoint quality is bounded in the ledger rather than generalised. | `CLAIMS.md` Q1 limits; `g5-failover-*.json` |
| 7 | "Several of your results are ties, and ties are not results." | They are reported as ties, with intervals, and labelled as bounds the claim must clear rather than as findings (the stricter action-level end task; compaction equivalence; the compiler against retrieval). The positive claims are the law, the inversion, capacity, recovery and the routing region - each with a receipt and a stated scope. | §2.2, §4.3, §A.1 |
| 8 | "The state bound is asserted, not derived - why 4.6-8K and not less?" | Because it is measured from below and above: 4,096 with a 3,072-token window fails the fidelity allowance (-2.40 pp), 4,608 with a 3,584-token window holds it at 0.00 pp, and 12,288 does not improve either metric. The bound is also a *design choice* rather than an intrinsic ceiling - lexical evidence mass grows with session age on this corpus, and the paper says so. | `g2-compiler-window3k-b*.json`; `g2-evidence-mass-v2.json` |
| 9 | "How do I know each arm ran the configuration the paper describes, rather than a nearby one that happens to give the same answer?" | Because the receipt records the flags that produced it (`config`), the audit lists any receipt that predates that block, and the one live instance found this way is in the appendix: the compaction baseline's runner passed a 3,072-token window while its receipt had been produced with a 6,656-token one, and the two are indistinguishable from the numbers (both hold the surface) or from the view size alone (3,072+512 and 5,120+1,536 both cap the view at 6,656).  The script was corrected, and re-running it reproduced the previous receipt exactly - all 48 rows, 223 summariser calls, view p50 6,650 - while adding the configuration. | `check_scripts.py --config-audit`; §A.2; `g2-compiler-compact-b8192-v1.json` |

## What the abstract's sentences rest on

Each sentence of the abstract maps to a receipt, which is the paper's defence against the most
common review outcome for a systems paper: a headline the artifacts do not support.

| abstract sentence | receipt |
|---|---|
| "a 6-8K execution state holds the next turn at 0.00 pp" | `g2-compiler-tailstate-compiledfar-tf0.5-b8192-v1.json` (0.00 pp), floor from `-b4096/-b6144` |
| "end-task score statistically indistinguishable from the best retrieval baseline" | 96 paired sessions: `g2b-patch-localization-{windowcompiler-b8192-n96,raw-b4096-n96}.json` |
| "state stays at 7-8K as history grows 64K -> 156K, and 40 -> 96 turns" | `g2-killer-table-v4.json` |
| "32x older costs 2.4x less to move (5.1x at 4K)" | `g2-killer-table-v4.json` inversion rows, from G3 measured primitives |
| "one worker holds 16x more sessions" | `g5-capacity-planning-v1.json` |
| "recovery and a model revision rebuild from a durable index instead of moving 12-128 GiB" | `g5-failover-two-workers-samemodel-v1.json`, `g5-failover-Qwen2.5-*.json` |
| "54 cells at a quality-admissible state (52 at 8,192 across all three replay regimes - balanced, slow-worker and worker-loss - and 2 at 16,384), every one of them clearing the 1.5x goodput bar" | `g4-quality-join-v2.json`, `g4-all-phase-summary-v2.json` |
| "a semantic compiler does not beat plain retrieval" | `g2b-patch-localization-raw-b4096-n96.json` against `-compiled-b8192-n48.json`; equal-budget ties in §4.3 |
| "collapsing a file costs 8.5 pp; snippet re-selection 20 pp" | `g2-compiler-consolidate-b8192-v1.json` against `-nopath-v1.json`; `g2-compiler-protectwhole-b8192-v1.json` |
| "teacher-forced fidelity is saturated by keeping the newest evidence whole" | `g2-compiler-recency-b8192-v1.json` (+0.29 pp), its own figure 6 |

## The three objections the paper cannot answer

Recorded here rather than left for a reviewer to find:

1. **Task success is unmeasured.** The end task (which files the next turn touches) is a proxy for
   whether the agent fixes the bug. It is a better proxy than next-token fidelity, and it is not
   task success.
2. **The routing consequence is simulated.** No live cluster re-places a session; the replay uses
   measured primitives and declared workloads, and one real distributed measurement exists (the
   process-level failover).
3. **The corpus reaches 96 turns and 156K tokens.** The paper's own killer table would want 300-500
   turns and 1M; the 1M rows are composed from measured primitives with one extrapolated input,
   and that is labelled wherever it appears.
