# Kill gates

EphemeralKV is now a **session-mobility** project. Five gates decide whether the
central claim survives. Every gate has a failure condition written before its
measurement.

The key systems quantity is the **mobility tax**

```text
mobility_tax = completion_time(cold remote route) - completion_time(warm local route)
```

Traditional full-KV mobility makes this tax grow with accumulated history. EphemeralKV
only matters if a remote turn can instead materialize a small active working set from a
durable index, making the tax primarily a function of the active set.

## G1 — Reject the easy tiered-storage story

**Question.** Does `discard + recompile` simply beat DRAM/NVMe KV restore on latency?

**Current evidence.** `benchmarks/kill_gate_crossover.py` has an accounting layer using
declared rates. Under that model, active recompilation beats a full re-prefill but
does **not** beat DRAM/NVMe KV movement by 512 turns. This is an accounting signal, not
a hardware measurement.

**Decision.** Do not use "recompute is cheaper than offload" as the paper thesis.
Hardware measurement can still quantify the crossover, but G3/G4 must carry the work.

**Status.** Accounting completed; measurement layer open.

## G2 — A real agent turn has a bounded retrievable working set

**Question.** Can a model-independent durable index find the history needed by the next
agent turn without scanning the entire transcript or materially harming task quality?

**Design.**
Use public multi-turn agent traces (SWE-Bench/OpenHands/BFCL-style plus at least one
long-memory agent benchmark). Build an append-only span store and at least two
non-QCC compilers:

* lexical/provenance compiler (BM25, file/tool/result IDs, recency);
* embedding compiler.

For each turn measure:

* history tokens and selected active tokens;
* index nodes/postings visited and lookup latency;
* task success / answer quality relative to a matched full-history or strongest feasible
  baseline;
* how active-set size and lookup time scale with session age.

**Advance rule.**
At long histories, the active fraction must fall rather than track history; p95 lookup
must be sublinear in stored tokens in practice; and the selected-view quality loss must
stay within 2 percentage points on the primary task metric or be recovered by a
fallback.

**Kill rule.**
If either lookup cost or required selected tokens scales approximately linearly with
history on realistic traces, the mobility abstraction collapses.

**Status.** Both halves are measured on the public corpus
(`thoughtworks/agentic-coding-trajectories`, 15,000 sessions, rebuilt from the parquet by
`benchmarks/build_trace_sessions.py`).

*Structural half* (`g2_trace_index.py`): the index selects a median 12% of a 32K-128K
history, and the fraction falls as history grows rather than tracking it.

*Model half* (`g2_model_quality.py`, frozen Llama-3.2-1B, teacher-forced next-turn target,
both arms scored with the same truncation): the fidelity of the retrieved view is a
**dose-response in the active budget**, measured on the same 192 examples:

| active budget | 8K-32K token-accuracy delta | active fraction at 8K-32K | verdict |
|---:|---:|---:|---|
| 4,096 | -10.1 pp | 0.21 | kill |
| 8,192 | -6.0 pp | 0.41 | kill |
| 16,384 | -0.3 pp | 0.82 | advance |

and on the longest sessions (median history 82,440 tokens, max 156,137, turns filtered to
a preceding history of at least 32,768 tokens) a 4,096-token view is **5.0% of history**
for -8.3 pp.

The reading is therefore split and has to be reported that way: **the working set is
sparse and stops tracking history, but at 4-8K tokens it does not preserve the turn**; the
loss is flat in history (so this is not a scaling failure) and disappears only once the
view is a large fraction of a short history. What the gate kills is the *combination*
"small fixed budget **and** teacher-forced parity"; end-task agent quality (not
teacher-forced continuation) remains open, and the two things that could move the
required budget are a stronger compiler (embedding/provenance rather than lexical) and a
workload whose turns depend on less of the transcript.

## G3 — History-free mobility on one machine

**Question.** With active work fixed, does the measured cold-route penalty stop scaling
with session age?

**Design.**
Measure the same next turn at `32K / 128K / 512K / 1M` accumulated history under:

1. warm local KV;
2. full-KV transfer from CPU/remote tier;
3. full re-prefill;
4. EphemeralKV: indexed lookup + active-set prefill.

Sweep active sets `2K / 4K / 8K / 16K`. Report TTFT, wall clock, bytes moved, GPU peak
memory, CPU time, and index time separately.

`benchmarks/kill_gate_mobility.py` defines the matched-cost accounting target.\n`benchmarks/g3_phase_space.py` sweeps KV bytes/token, fabric bandwidth, and active-set\nsize so the real experiment covers regimes where full-KV movement should win as well as\nregimes where rematerialization should win.

**Advance rule.**
For a fixed active set, the measured EphemeralKV mobility tax from 128K to 1M should
grow by at most **1.25x**, while at least one full-history cold route grows materially
with history. The 1M/2K case should be cheaper to move than the 32K/16K case.

**Kill rule.**
If EphemeralKV's cold-route penalty still grows close to linearly with history, there is
no new mobility regime.

**Status.** Hardware primitives are measured on the A10G, Qwen2.5-0.5B (12,288 B of KV
per token, the only 1M-feasible Full-KV geometry on this card):

| quantity | 32K | 128K | 512K | 1M |
|---|---:|---:|---:|---:|
| one-way H2D of the history's KV payload | 17.3 ms | 68.9 ms | 276.9 ms | **555.6 ms** |
| full re-prefill (this stack) | 1.59 s (20.6K tok/s) | 14.32 s (9.2K tok/s) | **infeasible** | infeasible |

H2D is linear at 23.2-23.4 GB/s; full re-prefill is superlinear (4x the history costs 9x
the time) and does not fit at 512K. The ephemeral side is measured too - active-set
prefill at 38.7K / 43.1K / 40.4K / 33.8K tok/s for 2K / 4K / 8K / 16K tokens (0.053 /
0.095 / 0.203 / 0.485 s, repeats=3 after the first call's warm-up, which by itself cost
1.075 s at 2K and is exactly the artefact a single repeat would have reported) - plus the
structural lookup, 0.087 / 0.150 / 0.246 ms p50 for <=8K / 8K-32K / 32K-128K histories.

Composed, the two statements the gate asks for are:

* **the inversion holds on hardware**: 1M history with a 2K active set costs ~0.10 s
  against 32K history with a 16K active set at ~0.49 s, a 5x separation driven by the
  active set rather than the history;
* **the tax stops tracking history**: the active prefill depends only on the active token
  count, and lookup grows 2.8x while the corpus grows ~16x (sublinear), so the <=1.25x
  band is satisfied *within the measured buckets*; a trace with genuine 1M-token
  histories does not exist in the public corpus, so the 1M lookup row is extrapolated
  from postings growth, not measured.

The honest limit: on this *local* 23 GB/s link, moving the full 12 GiB KV costs 0.556 s,
which still loses to a 2K ephemeral route by 5.6x but beats a 16K one - so the ephemeral
advantage at 1M is a function of the active set, and the crossover against a full move
sits near 120 GB/s effective bandwidth for a 2K active set. End-to-end multi-worker
routing is G4.

## G4 — Break sticky routing at cluster level

**Question.** Can cheap mobility reverse the current agent-serving preference for sticky
routing?

**Design.**
At 4--8 data-parallel workers, replay sessions with realistic tool gaps and skew. Compare:

* strict session-sticky routing;
* queue/load-aware routing with full-KV migration;
* KV-aware routing with an escape hatch;
* **Ephemeral mobility-cost routing**: use the same locality-vs-queue decision, but\n  replace the history-sized cold-route penalty with the measured active-set\n  rematerialization cost.

The scheduling rule is intentionally simple:

```text
choose worker i minimizing:
    queue_delay(i) + (0 if warm(i) else predicted_mobility_tax(request, i))
```

Run balanced load, hotspot bursts, one-worker slowdown, one-worker failure, and recovery.
Report p50/p95/p99 TTFT/JCT, SLO goodput, throughput, HBM/session, bytes moved, and route
migration rate.

**Best-paper gate.**
Under at least two realistic skew/failure regimes, changing the cold-route cost law\nshould improve SLO goodput by >=1.5x or p99 by >=30% against the strongest\nsticky/cache-aware baseline,
while regressing balanced-load median latency by <=5%.

**Kill rule.**
If a strong sticky/cache-aware baseline remains better across the measured operating
envelope, the core paper claim is false.

**Status.** Open.

## G5 — Recovery without session ownership

**Question.** Does worker/model replacement preserve session availability without moving
history-sized KV state?

**Design.**
Kill or drain the worker that served a long-running session, then resume the next turn
on another worker. Repeat across a compatible model replica and across a model
revision/adapter change where old KV is invalid. Compare:

* sticky worker restart / cold full prefill;
* tiered-KV restore;
* Ephemeral rematerialization from transcript + index.

The transcript/index path is expected to be model-independent; this is an enabling
property, not the novelty by itself.

**Advance rule.**
Failover recovery cost must follow the active set, not accumulated history, and the
session must not require a durable model-specific KV object for correctness.

**Kill rule.**
If recovery needs the history-sized KV or equivalent model-bound backing state, the
"session has no home" abstraction is not realized.

**Status.** Open.

## What kills the project outright

* G2: live state or index work scales roughly linearly with history on realistic agents.
* G3: remote materialization remains history-sized.
* G4: cheap mobility does not translate into a cluster-level p99/goodput win against
  strong sticky/cache-aware routing.
* Quality requires QCC-specific behavior; Paper B must stand without Paper A.
