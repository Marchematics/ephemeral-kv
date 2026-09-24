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

**End-task half, first receipt (`benchmarks/g2b_patch_localization.py`).** 2,736 sessions
of the public corpus qualify (`patch_present`, `max_isl >= 32,768`, and a recoverable patch
file set), and the metric is file-level localization of the *generated* next turn against
the recorded patch's files - a decision an evicted history can actually break.  On 24
sessions (median history 32,897 tokens, 16,384-token view at an active fraction of 0.498),
generating 96 greedy tokens per arm:

| reference | arm | precision | recall | F1 |
|---|---|---:|---:|---:|
| recorded patch files | full history | 0.104 | 0.029 | 0.045 |
| recorded patch files | **active view** | 0.313 | 0.218 | **0.219** |
| recorded next turn's files | full history | 0.000 | 0.000 | 0.000 |
| recorded next turn's files | **active view** | 0.250 | 0.236 | **0.222** |

So bounded retrieval does not cost the end-task decision here - it improves it, in the same
direction as the fidelity receipt but larger, which is what a focused view should do to a
model that is otherwise diluted by 33K tokens of raw transcript.  The absolute level is low
for both arms (a 96-token continuation names few of the recorded files), so this is a
direction, not a headline: more generated tokens, or a plan-style prompt, would sharpen it.

Two harness bugs had to be fixed to make the metric possible at all, and both affected the
earlier quality receipts:

* `_content` rendered only a message's `content`, dropping `tool_calls_json` - so a view
  contained a command's *output* but not the command, and the recorded next turn's own file
  mentions were invisible (every example scored null against them);
* the trace builder dropped `ground_truth_meta_json`, the corpus's only end-task signal.

**Previous next step for the end-task half (`g2b`).** Teacher-forced fidelity is a proxy; the
corpus supports a real downstream behaviour instead.  `ground_truth_meta_json` carries
`patch_present` and `resolved` per session, and the *patch's file set* is recoverable from
the transcript's tool outputs and tool-call arguments (`diff --git a/<path> b/<path>` in
`git diff` output, `Modified File:` summaries, editor commands), not from assistant prose -
a scan of assistant `content` alone finds only a handful of the long sessions.  The metric
to measure is therefore **patch localization**: generate the next turn under the full
history and under the active view, extract the file paths each continuation names, and
score precision/recall/F1 against the recorded patch's files.  That is a decision an
evicted-history policy can actually break, and it needs no sandbox.  (The builder dropped
`ground_truth_meta_json` until now, which is why this was not possible earlier.)

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


## Where the gates stand together

Two of them now bound the same quantity from opposite sides, and the gap between them is the
project's central open problem rather than a detail:

| gate | what it fixes |
|---|---|
| G2 (quality) | the active view must hold **16,384 tokens** with dedup (-1.94 pp, fraction 0.200 at 82K histories) to preserve the next turn, and 32,768 without dedup |
| G4 (routing) | the cold-route law changes the routing decision only at **2,048 active tokens**, and only when a worker disappears and its sessions must be re-materialized |

So the quality side needs an order of magnitude more active state than the routing side needs
to win, and that is where the paper's remaining work is: the compiler, not the router.  Two
levers are already measured to move it in the right direction - recency + oversized-span
truncation + identifier provenance took the long-history loss at a fixed 4,096-token view
from -21.98 pp to -7.17 pp, and content-identity dedup then halved the budget that holds - so
the question is whether a *snippet-level* or embedding compiler can reach 2-4K without losing
the turn.  If it cannot, the honest system claim is the one G4 already supports: session
mobility pays for recovery (worker loss) and for capacity, not for steady-state routing on
coding-agent traces with their current compilers.

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

**Status.** Replay simulator implemented (`benchmarks/g4_routing_replay.py`): every cost is
read from the G3 receipts, the workload (arrivals, tool gaps, per-session history growth)
is declared because the public corpus has no timestamps, and the plan's gate is evaluated
in the artifact rather than in prose. Policies: `strict_sticky`, `sticky_saturated` (the
KV-aware escape hatch production ships), `full_kv_move`, `full_reprefill`, `ephemeral`,
over balanced load, a slow worker, and a worker that disappears and must have its sessions
re-materialized elsewhere.

**Result: 4 of 48 cells advance, and they are all forced mobility with a 2,048-token
active set** (goodput x1.97 against the strongest baseline, p50 2.46 s against 2.48 s).
Everything else is `not_established`, and the reason is the same tension G2 found from the
other side:

* with a **16,384-token** active set - what G2 measured as fidelity-preserving on long
  histories - an ephemeral cold route costs 0.485 s of prefill, while moving a KV payload
  on this 23.2-23.4 GB/s fabric costs 0.017 s (32K history) to 0.556 s (1M).  The
  ephemeral route is therefore *worse* in every regime measured (goodput x0.16-0.75),
  including after a worker failure;
* lowering the fabric to 10 or 2 GB/s does not rescue it while sessions stay warm: a cold
  route is only paid on placement, migration, or displacement, so the cost law is rarely
  exercised (at 2 GB/s only `worker_failure` + 2K advances, balanced is still
  `not_established` at x1.17);
* `strict_sticky` is the only policy the law clearly beats (goodput 0.00-0.29 against
  0.33-0.42), which is the easy version of the claim and not the gate.

So G4's honest state is **conditional and open**: mobility pays when the active set is
small *and* sessions are forced to move, and the project's binding constraint is the joint
requirement - small active sets are what make routing cheap, large ones are what make the
next turn faithful. Closing G4 means closing that gap (a stronger compiler, or a workload
whose turns need less of the transcript), not tuning the router.

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

**Status. Measured, first receipt** (`benchmarks/g5_rollout_resume.py`, 24 examples from
the longest sessions, median history **130,131 tokens**, active budget 16,384):

| route after a model rollout | cost |
|---|---|
| durable transcript + index: lookup | 0.25 ms |
| ephemeral resume: active-set prefill on the **new** model | **0.55 s** (Qwen2.5-0.5B), **1.04 s** (Llama-3.2-1B) |
| full-KV resume: re-prefill the history on the new model | **6.31 s** (optimistic linear scaling of a superlinear measured curve; **infeasible** at 512K+) |

and the durable view is model-independent in the way the claim needs: the *same* index and
the *same* active view, scored with a model that never saw the session, keeps the next
turn's fidelity (**Llama-3.2-1B: token-accuracy delta +0.48 pp, NLL -0.010**), while the
model that built the view scores +7.25 pp - the active view beating its own full-history
baseline at 130K tokens, where the full context is past what that checkpoint handles.

So a rollout costs a bounded prefill of the active set rather than a history-sized
transfer or recompute, and worker failure loses an optimisation rather than a session.
The caveats are in the artifact: the budget is applied with the primary model's tokenizer,
the re-prefill figure is a flagged optimistic scaling, and this is teacher-forced fidelity
rather than end-task success.


## What kills the project outright

* G2: live state or index work scales roughly linearly with history on realistic agents.
* G3: remote materialization remains history-sized.
* G4: cheap mobility does not translate into a cluster-level p99/goodput win against
  strong sticky/cache-aware routing.
* Quality requires QCC-specific behavior; Paper B must stand without Paper A.

**Correction (same round, after fixing the renderer).**  Every fidelity number above was
produced by a harness whose `_content` rendered only a message's `content` and dropped
`tool_calls_json` - so neither arm saw the agent's own commands, only their outputs.  With
the renderer fixed and everything else identical (long slice, recency+provenance compiler,
48 examples, 131,072-token ceiling), the 16,384-token view loses **3.46 pp** of token
accuracy in the 32K-128K bucket (NLL +0.164) instead of the 1.08 pp the earlier receipt
showed, i.e. above the 2 pp tolerance:

| receipt | overall accuracy delta | 32K-128K bucket | NLL delta |
|---|---:|---:|---:|
| before the renderer fix | 0.00 pp | -1.08 pp | +0.063 |
| after the renderer fix | -2.00 pp | **-3.46 pp** | +0.131 |
| after the fix, 32,768-token view | 0.00 pp | **-0.49 pp** | -0.005 |

So the fidelity-preserving active budget is **32,768 tokens on this corpus with this
compiler** (active fraction 0.372 at 82K-token histories, against 0.82 for the same fidelity
in the 8K-32K range), and the earlier "16K holds" statement is superseded rather than
deleted: it was measured on a harness that made the full-history arm weaker than it is.  The
corrected dose-response is:

| active budget | 32K-128K accuracy delta | active fraction | renderer |
|---:|---:|---:|---|
| 4,096 | -10.1 pp | 0.21 | before the fix |
| 8,192 | -6.0 pp | 0.41 | before the fix |
| 16,384 | -3.46 pp | 0.200 | corrected |
| **32,768** | **-0.49 pp** | **0.372** | corrected |

and the end-task metric is the *more* forgiving of the two at 16,384 (localization F1 0.219
for the active view against 0.045 for full history), which is worth stating plainly: token
accuracy is the harsher gate, and the downstream file decision survives a smaller view.

**Collapsing duplicate spans halves the required budget.**  Coding trajectories repeat
themselves - a file is `cat`-ed again after an edit, the same test output appears twice - and
measured on six long sessions **17.5% of the spans in a compiled view are exact duplicates of
another span in the same view** (26.8% share an 80-character prefix).  `compile_view(...,
dedup=True)` orders each group of identical texts most-recent-first and keeps only the newest
copy, so the same token budget buys more distinct content.  Under the corrected renderer, on
the same 48 examples:

| active budget | dedup | 32K-128K accuracy delta | active fraction | inside the 2 pp rule |
|---:|---|---:|---:|---|
| 8,192 | yes | -6.94 pp | 0.100 | no |
| 16,384 | no | -3.46 pp | 0.200 | no |
| **16,384** | **yes** | **-1.94 pp** | **0.200** | **yes, at the boundary** |
| 32,768 | no | -0.49 pp | 0.372 | yes |

So the fidelity-preserving view on this corpus is **16,384 tokens at an active fraction of
0.200** (82K-token histories) rather than the 32,768 the pre-dedup compiler needed - a
factor of two from one compiler change, and in the direction the mobility claim needs, since
the routing side pays for active *tokens* while the quality side is what sets them.  The
-1.94 pp sits at the tolerance boundary (n=44 in that bucket), so it is reported as holding
*at* the rule rather than comfortably inside it.  The direction the mobility claim
needs - the active *fraction* falling as history grows (0.200 at 82K-token histories against
0.82 in the 8K-32K range) - is unchanged, because it does not depend on the renderer.
