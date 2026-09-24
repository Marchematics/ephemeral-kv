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

## The claim this project is now built around

```text
history size  =/=>  mobility cost
```

and the causal chain that has to hold for it to be a new abstraction rather than a cache
policy:

```text
compiler -> small executable state -> history-independent mobility -> routing phase change
```

The durable session is `history + index`; the execution state is `compile(history, q)`; local KV
is a **disposable execution artifact**.  The compiler is not a summariser - "compress old history
and feed it back" is already a baseline other systems compare against - it is an **executable
state compiler**, the same shape as a database turning a log into a materialised view:

| input | output |
|---|---|
| million-token transcript + current action | current facts, current file/tool state, unresolved constraints, the exact evidence that must stay verbatim, provenance pointers |

with superseded state, repeated tool output, repeated reasoning and intermediate failed versions
*removed* rather than shortened.

**The five hard results that decide whether this is a Best-paper project**, in the order they
have to be earned:

| # | result | bar | status |
|---|---|---|---|
| 1 | semantic compiler shrinks the required view | **16K -> <= 8K**, ideally 4-8K | extractive family measured: **not met** (-5.44 pp at 8K against a 2 pp allowance) |
| 2 | fidelity against full history | **<= 2 pp** | not met at 8K yet (-1.94 pp is what 16K gives) |
| 3 | real agent task (patch localization) | no regression, ideally better than raw retrieval | **met**: 0.292 at 8K compiled against 0.237 raw and 0.045 for full history |
| 4 | G4 winning region | clearly wider than **4/48**, no longer only the 2K extreme | open, re-measured after #1 |
| 5 | cluster consequence | >= 1.5x SLO goodput or >= 30% p99 in several realistic regimes | open, after #4 |

and one deeper result that separates "strong OSDI" from Best: a **phase diagram inversion** in
which mobility cost is a function of the compiled state rather than the history,
`dM/dL ~= 0`, so that an old session can be *cheaper* to move than a young one and session age
stops meaning anything for placement:

```text
session age   raw history   executable state
20 turns           40K             6.1K
100 turns         210K             6.8K
300 turns         640K             7.0K
500 turns           1M             7.3K
```

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

**The killer table, measured (and what it is missing).** The mobility law is only half the
claim; the other half is that the *quality* a fixed state buys does not decay as the session
grows.  Bucketing the G2 receipts by session length (`benchmarks/g2_killer_table.py`) gives

| raw history p50 | compiled state p50 | teacher-forced fidelity (tail_state, 8,192 budget) |
|---:|---:|---:|
| 64,401 | 7,829 | **+0.00 pp** (NLL +0.254) |
| 82,972 | 7,899 | **+0.00 pp** (NLL +0.003) |
| 155,574 | 8,203 | **+1.43 pp** (NLL -0.288) |

so across a 2.4x growth in raw history the state that must move grows by 5% and the fidelity
delta does not move at all - `dM/dL ~ 0` on the fidelity half of the claim, measured rather
than asserted.  The *decision* column of this table is the piece still missing: every
end-task receipt is drawn from ~32.9K-token histories, because the filter takes the first
examples above the threshold, and the 64K-128K sessions in the corpus (338 of them) have not
been measured yet.  Until they are, the paper can claim a flat *fidelity* law at fixed state
and a *decision* law only at one session length.

Composed with the hardware numbers, the inversion the paper wants is:

| session | compiled state | mobility (lookup + active prefill) | full-KV move |
|---|---:|---:|---:|
| 32K, raw 16K view (what fidelity needed before the compiler) | 16,384 | 0.4855 s | 0.0173 s |
| **1M, compiled** | **8,192** | **0.2031 s** | 0.5530 s |
| **1M, compiled** | **4,096** | **0.0954 s** | 0.5530 s |
| 262K, compiled | 8,192 | 0.2031 s | 0.1383 s |

The first two rows are the counter-intuitive one: a session **32x older** is **2.4x cheaper**
to move (5.1x at a 4K state), and the crossover where a full-KV move becomes the better cold
route sits around 262K-1M tokens on a 23.3 GB/s link, not around 32K.  Session age stops
predicting placement cost; the compiled state does.


**The requirement is content volume, not selection granularity.**  A second compiler
change was tried and did not help: `snippet=True` keeps the query-relevant lines of an
oversized span (with context and an elision marker) instead of its prefix, which is the
right response to a two-thousand-line tool dump.  Under the corrected renderer, on the same
48 examples:

| active budget | compiler | 32K-128K accuracy delta | active fraction |
|---:|---|---:|---:|
| 4,096 | dedup + snippet | -13.14 pp | 0.050 |
| 8,192 | dedup | -6.94 pp | 0.100 |
| 8,192 | dedup + snippet | -6.70 pp | 0.100 |
| 16,384 | no dedup | -3.46 pp | 0.200 |
| **16,384** | **dedup** | **-1.94 pp** | 0.200 |
| 32,768 | no dedup | -0.49 pp | 0.372 |

Snippet selection is within noise of plain truncation at 8,192 (-6.70 against -6.94 pp), so
*which part* of a span is kept is not what binds; the loss grows steeply with less distinct
content (about -7 pp at 8K, -13 pp at 4K against -1.9 pp at 16K).  Removing *duplicated*
content helps by a factor of two in budget, and re-selecting *within* a span does not help
at all.  The next lever therefore has to reduce how much distinct content the turn needs -
a semantic/embedding compiler that summarises or compresses spans, or a workload whose turns
depend on less of the transcript - rather than any further re-ranking of the same text.

**The agent's next decision needs half the view the next-token loss does.**  The budget sweep
from the previous section used the teacher-forced metric; the end-task metric is the one that
decides what a router may actually use, so it was swept over the same 24 paired sessions
(median history 32,897 tokens, dedup+snippet compiler, 96 generated tokens):

| active budget | active fraction | F1 vs the patch's files | F1 vs the next turn's files |
|---:|---:|---:|---:|
| 4,096 | 0.125 | 0.042 | 0.111 |
| **8,192** | 0.249 | **0.237** | **0.264** |
| 16,384 | 0.498 | 0.142 | 0.139 |
| full history | 1.000 | 0.045 | 0.000 |

with paired bootstrap over the same sessions: 4,096 -> 8,192 is **+0.196, 95% CI [+0.024,
+0.361], 9 wins against 1 loss**, while 8,192 -> 16,384 is +0.095 with a CI that includes
zero ([-0.026, +0.231]).  So the end-task metric **collapses at 4K, holds from 8K, and does
not resolve above it at n=24** - and at every budget where it holds, the bounded view beats
feeding the model the whole 33K-token transcript (0.045).

That gives three requirements for the same session, measured, and they are not the same
number: the router needs ~2K active tokens to want to move, the agent's next decision holds
from ~8K, and teacher-forced token accuracy needs ~16K.  The honest system claim follows the
middle one (the decision is what a user sees), and it is still four times what the routing
side needs - so the compiler gap is halved, not closed.

## The semantic compiler, first iteration: what the ablation says

The routing gate and the fidelity gate together said the remaining work is the *compiler*: the
agent's next decision holds from 8,192 active tokens, teacher-forced fidelity needs 16,384, and
re-ranking the same text does not move either.  This iteration built the compiler the plan
called for - retrieve evidence, consolidate it, keep what must be verbatim - and measured it
against the same 48 examples and the same receipts.

`ephemeralkv/consolidate.py` implements the two stages: :func:`consolidate` groups the retrieved
spans by the state they describe (file path, content identity), keeps the newest, and records
the turns it replaced as provenance; :func:`select_verbatim` cuts a unit that does not fit by
dropping prose before anything executable (code, tracebacks, diffs, paths, numbers stay
byte-for-byte), and marks every omission so a compiled view never looks complete when it is not.
The compiled arm retrieves twice its budget and compiles down, so the comparison is exactly
"16K of raw evidence into 8K of compiled state".

At a **8,192-token budget**, 48 examples, 32K-128K histories:

| compiler variant | token-accuracy delta | NLL delta |
|---|---:|---:|
| raw truncation (the previous receipt) | -6.94 pp | +0.476 |
| evidence consolidation, latest state per file | **-13.90 pp** | +0.640 |
| **evidence consolidation without collapsing file states** | **-5.44 pp** | +0.550 |
| consolidation keeping the replaced views' verbatim lines | -7.38 pp | +0.320 |
| *16,384-token view, for reference (the current requirement)* | *-1.94 pp* | *+0.151* |

Three things follow, and the first is a correction to the design rather than to the numbers:

1. **Collapsing a file to its latest state is actively harmful** (-13.90 pp, twice the loss of
   plain truncation): earlier views of the same file carry content the next turn needs - a
   region the later view no longer shows, the context of a change - so "latest state wins" is
   the wrong rule.  The plan's example (four views of `cache.py` becoming one state plus
   provenance) is the intuition that the measurement rejects.
2. **The best extractive variant is only ~1.5 pp better than raw truncation** (-5.44 against
   -6.94) and nowhere near the 2 pp gate, so the gate is **not met**: the extractive family -
   retrieval, duplicate collapse, state collapse, verbatim-priority truncation - does not get
   8,192 tokens to fidelity parity on this corpus.
3. Keeping the replaced views' verbatim lines does not rescue it either (-7.38 pp), which says
   the missing information is not only "executable lines".

**The gate splits, and the split is informative.** With the same 8,192-token compiled view, the
*end-task* metric does not just hold, it improves - and the improvement is large enough to beat
the larger raw views:

| view | F1 vs the patch's files | F1 vs the next turn's files |
|---|---:|---:|
| 8,192 raw (dedup + snippet) | 0.237 | 0.264 |
| **8,192 compiled, no state collapse** | **0.292** | **0.278** |
| 16,384 raw | 0.142 | 0.139 |
| full history | 0.045 | 0.000 |

So against the gate's three conditions: the view *is* 8,192 tokens (mechanically), the agent's
next decision **improves** (0.292 against 0.237), and teacher-forced fidelity **fails**
(-5.44 pp against the 2 pp allowance).  The two metrics disagree about the same view, and the
disagreement is the finding: teacher-forced next-token accuracy penalises any context change
that alters surface realisation, while the decision the agent actually has to make is better
served by a focused compiled view than by twice as much raw evidence.

**The executable-state compiler was built and it does not help either - and the reason is a
measurement, not a bug.** `ephemeralkv/statecompile.py` replays the log: full dumps *set* a
file's content, unified diffs and `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks *apply* to
it, a command's repeated runs collapse to its latest result, repeated reasoning keeps only its
newest occurrence, and everything keeps provenance.  At 8,192 tokens:

| compiler variant | teacher delta (32K-128K) | end-task F1 (patch files) |
|---|---:|---:|
| raw truncation (dedup + snippet) | -6.94 pp | 0.237 |
| consolidate, latest view per file | -13.90 pp | - |
| **consolidate, no state collapse** | **-5.44 pp** | **0.292** |
| consolidate + replaced views' verbatim lines | -7.38 pp | - |
| **materialise (replay the log into current state)** | -7.21 pp | 0.231 |
| *16,384-token view (what the gate is trying to halve)* | *-1.94 pp* | *0.142* |

So the 16K -> 8K gate is **not met** by any of five variants on the teacher-forced metric (best
-5.44 pp against a 2 pp allowance), while the *end-task* metric is comfortably viable at 8K
(0.292 against 0.237 raw, and better than twice the raw evidence).  And the diagnostic explains
the failed variants: over three long examples the retrieved evidence classifies as

| span type | share |
|---|---:|
| loose text (reasoning, acknowledgements, chatter - no file, no command) | **74%** |
| command results | 14% |
| file events (dump, diff, edit) | **6%** |

A state compiler can only consolidate the 6%.  The binding constraint is therefore **what the
retrieval selects**, not how the selected text is compressed, and that diagnosis predicted that
selection *state-first* - every materialised file state, the latest result of each command, and
only then loose text - should recover the loss.  It was built (`state_first_units` in
`ephemeralkv/statecompile.py`) and it does the opposite:

| compiler variant | teacher delta (32K-128K) | end-task F1 (patch files) |
|---|---:|---:|
| raw truncation (dedup + snippet) | -6.94 pp | 0.237 |
| consolidate, no state collapse | -5.44 pp | 0.292 |
| materialise (replay the log into current state) | -7.21 pp | 0.231 |
| **state-first selection** | **-15.38 pp** | 0.242 |

State-first is the **worst** variant measured - worse than raw truncation by 8.4 pp, with NLL
+0.936 against raw's +0.476 - and the reason is the same 6% that motivated it.  A state-first
view spends its budget on *whole file dumps*: they are large (one dump can exceed the entire
budget, which is why the control below had to skip rather than stop on the newest span) and
mostly irrelevant to the query, and promoting them evicts the query-relevant spans the lexical
ranking had found.  The view becomes query-agnostic.  The diagnosis was right that retrieval
selects mostly chatter and wrong that promoting state would help: the chatter is what the next
turn's surface form is conditioned on, and state is a small, expensive minority of the evidence.
Note also that the end-task metric stays flat across all of these (0.231-0.292): which files to
touch survives an 8K view built almost any way, while next-token fidelity does not.

Seven extractive variants have now been measured at 8,192 tokens - raw truncation, snippet
selection, three consolidation rules, log replay, and state-first selection - and the best of
them is -5.44 pp against the 2 pp allowance.  The family's ceiling is therefore not a tuning
question, and the remaining question is *why*: is the gate about which text is kept, or about
how many tokens of text are kept at all?  `--compile-mode recency` answers it directly by
removing selection entirely and keeping the last N tokens of history.

### The control that reframes the gate: 8,192 tokens of plain recency

Plain recency - no retrieval, no ranking, no consolidation, whole spans in their original order
- does what no compiler variant managed:

| active budget | 32K-128K (n=44) | >=128K (n=4) | active fraction p50 |
|---|---:|---:|---:|
| **8,192** | **+0.29 pp** | 0.00 pp | 0.097 |
| 16,384 | 0.00 pp | +0.71 pp | 0.174 |
| 32,768 | +0.66 pp | +0.71 pp | 0.295 |
| full history | reference | reference | 1.000 |

At 8,192 tokens - 9.7% of the median history - recency is indistinguishable from full history
(+0.29 pp, NLL -0.048), and the 16K and 32K rows are the same within noise.  So the fidelity
half of the gate *is* reachable at 8K, and it is reached by an arm with no compiler at all.
Two things follow, and the first is a correction.

**The earlier verdict was wrong, and the reason is specific.** In the lexical arm only
`recency_spans=3` spans were protected, and each was capped at `max_span_fraction=0.25` of the
budget, so the two or three most recent tool results - the evidence the next turn's surface form
directly continues - were truncated to 2,048 tokens or dropped outright.  Every compiler variant
inherited that, and the consolidation and state stages then rewrote what survived.  The 5-15 pp
losses were never about the far field; they were about the tail.  `tail_state` below tests that
causally by keeping the tail whole and spending the remainder on compiled far-field state.

**And teacher-forced next-turn fidelity cannot tell a compiler from truncation.**  8,192 tokens
of recency is already at parity, so "we halve the budget at equal fidelity" is not a defensible
headline: the trivial policy does it.  The claim has to be about what recency *cannot* do, which
is whatever needs evidence from far back in the session - and that is exactly what the end-task
metric measures.  This is the sharpest form of the metric disagreement first seen in the ladder:
the proxy metric that the compiler literature uses is nearly saturated by recency on these
traces, while the decision the agent has to make is not.

## Where the gates stand together

### The two metrics want different views, and that is the finding

Every arm measured at an 8,192-token view on the same corpora, both metrics on the same
examples:

| view (8,192 tokens) | teacher-forced fidelity | end-task F1 (patch files) |
|---|---:|---:|
| full history | reference | 0.045 |
| **plain recency** (whole spans, newest first) | **+0.29 pp** (NLL -0.048) | 0.154 |
| tail_state, 60% verbatim tail + compiled far field | 0.00 pp (NLL -0.016) | 0.104 |
| raw lexical (dedup + snippet) | -6.94 pp | 0.237 |
| **evidence consolidation, no state collapse** | -5.44 pp | **0.292** |
| log replay into materialised state | -7.21 pp | 0.231 |
| state-first selection | -15.38 pp | 0.242 |
| 16,384-token reference view | -1.94 pp | 0.142 |

Read down the columns: **fidelity is bought by the newest span, the decision by query-focused
evidence, and no view buys both.**  The fidelity column is nearly saturated by recency - which
is why a paper whose headline is "we halve the budget at equal fidelity" would be claiming
something a one-line truncation policy already does - while the decision column is carried by
the compiler (0.292 against 0.154 for recency and 0.045 for full history) and *destroyed* by
spending budget on generic recency padding (tail_state 0.104, worse than recency itself, because
older recency spans displace the evidence the decision needs).

The proxy metric the compiler literature uses cannot see this.  On these traces it is dominated
by the surface form of the turn being continued: keep the newest span whole and it is at parity
even with 90% of the session gone.  The end task can see it, because the file a patch will touch
is often named only in evidence from far back.  That is the paper's measurement contribution,
and it also sets the design target: an 8K view that is *the current turn plus the query's
evidence*, with no third category in the budget (`--compile-mode tail_query`, measured next).

### What each gate needs

| measurement | active budget it needs |
|---|---|
| G4 (routing) | **2,048-4,096 tokens** - with the widened grid, ephemeral mobility advances in every forced-mobility cell at both sizes |
| G2 end-task (what a user sees) | **~8,192 tokens** for the best decision (0.292); recency reaches only 0.154 |
| G2 teacher-forced (the proxy) | **8,192 tokens if the newest span is kept whole** (+0.29 pp); ~16,384 with lexical selection of the tail |

The old reading of this table - "the proxy needs twice the decision, and the compiler is the
open problem" - was wrong in an instructive way.  The proxy needs 8K, not 16K, and the 5-15 pp
losses that looked like a compiler failure were the tail being truncated to 2,048 tokens by
`max_span_fraction=0.25` in every lexical arm.  What the compiler actually buys is the *decision*
at a fixed budget, and what remains open is whether one view can hold both columns at once:
that is what `tail_query` measures, and if it cannot, the honest system claim is the frontier
above plus the regime G4 supports - mobility pays for recovery and capacity, not for
steady-state routing at the fidelity the proxy demands.

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

**Result: 39 of 264 cells advance** in the widened grid, against 4 of 48 in the first one.  The
first grid's boundary - "only forced mobility with a 2,048-token active set" - turned out to be
an artefact of the grid rather than of the cost law: its active-set axis contained nothing
between 2,048 and 16,384, so the region could only ever be reported at its extreme, and its
longest session was 262K tokens.  Adding the sizes the compiler can actually run at, and the
session scale the durability argument is about:

| active set | advancing cells |
|---|---:|
| 2,048 | 17/66 |
| 4,096 | 16/66 |
| 8,192 | 4/66 |
| 16,384 | 2/66 |

| session mix | advancing cells |
|---|---:|
| corpus (32K-262K) | 26/132 |
| with a 1M-token mix | 13/132 |

Every forced-mobility cell at 2,048 **and** 4,096 now advances (goodput x1.97 and x1.6 against
the strongest baseline), where the old grid could only show the 2K column; and at million-token
sessions - where moving the history's KV payload costs 555.6 ms against 0.053-0.095 s of
rematerialisation - the capacity regimes open as well.  The one extrapolated input is the 1M
lookup row (no trace in the public corpus is that long); every other cost is a G3 measurement,
and each receipt records the history mix it sampled.

The regimes that still do not advance are balanced load and short sessions, and the reason is
unchanged:

* with a **16,384-token** active set - what G2 measured as fidelity-preserving before the
  compiler existed - an ephemeral cold route costs 0.485 s of prefill, while moving a KV payload
  on this 23.2-23.4 GB/s fabric costs 0.017 s (32K history) to 0.556 s (1M), so the ephemeral
  route is worse wherever the payload is small (goodput x0.16-0.75);
* lowering the fabric to 10 or 2 GB/s does not rescue it while sessions stay warm: a cold route
  is only paid on placement, migration, or displacement, so the cost law is rarely exercised;
* in those cells the law still clearly beats only `strict_sticky`, which is the easy version of
  the claim and not the gate.

So G4 is now **measured and positive in the regimes it should be**, and its remaining coupling
to G2 is explicit: the 4,096 column advances only if the compiler holds the decision at 4,096,
which is exactly what the `tail_state`/`tail_query` arms at 4,096 are for.

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
