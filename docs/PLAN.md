# Kill gates

Five experiments decide whether this project is worth building. Each has a decision
rule written *before* it runs, and each is designed to fail cheaply on a single 24 GiB
GPU rather than after a serving system has been written.

## G1 — Crossover exists (the core gate)

**Question.** Past what session length does `discard + recompile a working set` beat
`keep and restore`?

**Design.** Simulate sessions of `T = 8, 32, 128, 512` turns with a fixed per-turn
addition. For each `T`, measure wall-clock and peak memory for four policies on the
same next request:

| policy | durable state | per-turn work |
|---|---|---|
| `keep_hbm` | full KV in HBM | none |
| `keep_dram` | full KV in host DRAM, moved in/out | transfer |
| `keep_nvme` | full KV on disk, streamed | transfer |
| `recompute_full` | transcript | full re-prefill of the history |
| `recompile_active` | transcript + index | compile a working set for the request only |

**Decision rule.** The project lives if `recompile_active` beats `keep_*` on total
time-to-response (JCT) for some `T <= 512`, or beats `recompute_full` by a factor that
grows with `T`. It dies if recompile's cost tracks history length (there is no
crossover) or if the working set is not small (gates G4/G5 fail).

**Status.** Not run. Harness: `benchmarks/kill_gate_crossover.py` (skeleton).

## G2 — Working-set sparsity holds on real agent traffic

**Question.** In real multi-turn agent trajectories, how much of the history is
actually live for the next request?

**Design.** Take public agent traces (SWE-Bench/OpenHands/BFCL-style) and, for each
turn, measure the fraction of history tokens whose removal changes the answer
(leave-one-out on chunks, or attention-mass attribution as a cheaper proxy).

**Decision rule.** The project lives if the live fraction is small and roughly flat in
session length (say <10% at 500K history). It dies if the live set grows with history,
because then recompilation cannot be cheap.

**Status.** Not run.

## G3 — Model-state reuse is *not* the only cheap path

**Question.** Is the restored KV actually faster than a fresh compile of a *small*
working set, on real hardware, including transfer and scheduling?

**Design.** Same as G1 but on the serving stack (continuous batching, several concurrent
sessions) instead of a single-request harness: measure TTFT, JCT, goodput and session
capacity at a fixed memory ceiling.

**Decision rule.** Lives if session capacity at a fixed HBM ceiling rises materially
without hurting TTFT at the same SLA; dies if scheduling overhead eats the gain.

**Status.** Not run.

## G4 — Longer request, smaller footprint (the sharpest claim)

**Question.** Can a request with a longer history occupy *less* GPU memory than one
with a shorter history?

**Design.** Pair requests: `history 1M / working set 2K` against
`history 32K / working set 16K`. Measure resident HBM per request under the same engine.

**Decision rule.** Lives if the inversion is reproducible; dies if the compiler's
overhead or fragmentation keeps the footprint ordered by history length.

**Status.** Not run.

## G5 — Model-agnostic durability

**Question.** Does a durable transcript survive a model change without invalidating the
session, where a durable KV cannot?

**Design.** Compile for checkpoint A, then serve the next turn with checkpoint B (and
with a different adapter / precision / rope configuration). Compare quality against
"KV compiled by the same model".

**Decision rule.** Lives if the transcript path shows the expected portability; this is
mostly a design property, so the experiment is a demonstration rather than a risk.

**Status.** Not run.

## What would kill the project outright

* No crossover on any `T <= 512` (G1), or
* live history growing with session length (G2), or
* the compile cost of a small working set being dominated by anything other than the
  working set (G1/G3).
