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

**Status.** Open.

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

`benchmarks/kill_gate_mobility.py` is the accounting receipt that defines the expected
scaling and routing threshold before hardware measurement.

**Advance rule.**
For a fixed active set, the measured EphemeralKV mobility tax from 128K to 1M should
grow by at most **1.25x**, while at least one full-history cold route grows materially
with history. The 1M/2K case should be cheaper to move than the 32K/16K case.

**Kill rule.**
If EphemeralKV's cold-route penalty still grows close to linearly with history, there is
no new mobility regime.

**Status.** Accounting artifact present; hardware measurement open.

## G4 — Break sticky routing at cluster level

**Question.** Can cheap mobility reverse the current agent-serving preference for sticky
routing?

**Design.**
At 4--8 data-parallel workers, replay sessions with realistic tool gaps and skew. Compare:

* strict session-sticky routing;
* queue/load-aware routing with full-KV migration;
* KV-aware routing with an escape hatch;
* **Ephemeral soft affinity**: use a warm prefix when it wins, otherwise rematerialize
  the active set on the worker with the best predicted completion time.

The scheduling rule is intentionally simple:

```text
choose worker i minimizing:
    queue_delay(i) + (0 if warm(i) else predicted_mobility_tax(request, i))
```

Run balanced load, hotspot bursts, one-worker slowdown, one-worker failure, and recovery.
Report p50/p95/p99 TTFT/JCT, SLO goodput, throughput, HBM/session, bytes moved, and route
migration rate.

**Best-paper gate.**
Under at least two realistic skew/failure regimes, soft affinity should improve SLO
goodput by >=1.5x or p99 by >=30% against the strongest sticky/cache-aware baseline,
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
