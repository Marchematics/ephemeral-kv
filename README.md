# EphemeralKV

**A long-lived LLM session should have an identity, not a home.**

Agent-serving stacks increasingly make a session *sticky*: a follow-up turn is routed back to the
worker that already owns its KV prefix, because moving or rebuilding a history-sized KV cache is
expensive.  This project tests a different abstraction, in three objects rather than one:

```text
durable session     the transcript plus a model-independent index; append-only, grows
execution state     compile(history, q): a bounded view, rebuilt on demand
local KV            a disposable artifact of having computed that state once
```

If a cold route costs `lookup + prefill(active_set)` instead of `move_or_recompute(history)`, then
session age stops determining where the next turn may run - not because KV is cheap to move, but
because the thing that moves is small.

## What this repository reports

Measured on real coding-agent traces; every number below is asserted by `benchmarks/paper_numbers.py`
against a checked-in receipt, and the CPU-only path in
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) re-derives all of them without a GPU.

* **The state is bounded and does not track age.**  4,096 tokens holds the surface inside the 2 pp
  fidelity allowance (-0.96 pp); 6,144 holds it at parity (0.00 pp).  Across an **8.9x range of raw
  history** (67K to 984K tokens, the long end composed from real sessions) and **45x of turns**, the
  state stays between 4,588 and 8,439 tokens and 18-40 KB of transfer per bucket; at the floor
  configuration it
  is 3,922-3,928 tokens.
* **Placement stops depending on age.**  A session 32x older costs **2.4x less to move** (5.1x at a
  4K state), and one worker holds **32x more sessions** (427 / 40 / 16 per worker for 0.5B / 8B /
  70B geometries) - the same number whether the sessions are 64K tokens or a million.
* **Routing has a phase change at an admissible state.**  Of 724 replay cells, **157 advance at a
  state whose measured quality point passes** - 77 at the 4,096-token floor, 18 at 6,144, 59 at
  8,192, 3 at 16,384 - across **all five regimes** the replay models: balanced, slow-worker hotspot,
  worker loss, a flash crowd and a heavy-tailed session-size mix.  **117 clear the 1.5x SLO-goodput
  bar** with a finite ratio (median 1.61x, max 5.06x) and 40 face a baseline that completes no work
  at all.  **23 clear the 30% p99 bar** (best +68.7%), every one of them in the flash-crowd regime,
  where all workers evict at once and a history-sized cold route queues behind the spike - and that
  tail claim is swept over the burst's own parameters rather than quoted from one setting: it holds
  at the 4,096-token state in 6 of 9 burst shapes (advancing in the same 6) and at 8,192 in 1 of 9, so it is
  stated as conditional on the workload rather than as a property of the regime.  On a workload whose
  sessions **actually reach a million tokens** - the grid's history choices are targets, and with its
  8 turns per session nothing exceeded 113K, now recorded per receipt as `realised.history_max` - the
  region advances in **16 of 20 cells and every one clears the 30% p99 bar**, in four of the five
  regimes at both the 4,096- and 8,192-token states.
* **Recovery moves state, not history.**  A killed worker's replacement rebuilds 8,192 tokens in
  0.91-1.42 s with identical token accuracy, reading ~32 KB instead of the 0.38 GiB of KV the owner
  held; the durable object also resumes on a **different** model.

## What is not claimed

* **A semantic compiler does not pay here.**  Content dedup is a statistical tie with plain
  retrieval, collapsing a file to its latest state costs 8.5 pp of fidelity, re-selecting the newest
  output's lines costs 20 pp, and at the 4,096-token floor a materialised far field scores -2.48 pp
  against consolidation's -2.40 pp while widening the window reaches -0.96 pp.  What carries quality
  is the window and retrieval, not a compiler.
* **Compaction is equivalent, not worse.**  Measured under the same budget and window, with the
  summary in every scored context; the paper claims no win over it.
* **Task success is unmeasured.**  The end task is next-turn file localisation against the recorded
  patch, plus an offline action-level rescoring.  Its **executable** form - 48 mid-session turns,
  three arms parsed into tool invocations - is measured and is deliberately two-sided: the bounded
  state **beats plain retrieval decisively** (+0.292, CI [+0.146, +0.438]) and **regresses against
  the full transcript** (-0.250, CI [-0.417, -0.063]), because the views carry a median of 44, 6 and
  1 prior actions respectively.  A view ranked for *evidence* does not show the model the format it
  is being asked to produce.
* **Past 156K tokens the histories are composed** out of whole real sessions, and the 8B/70B
  geometries are declared rather than measured.  Both are labelled wherever they appear.

## Where to read

| document | what it is |
|---|---|
| [`docs/PAPER.md`](docs/PAPER.md) | the manuscript: abstract, laws, system, evaluation, limitations, claim-to-receipt map |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | the corpora's derivation, the CPU-only verification path, the GPU runners, the cold-start drill |
| [`docs/VERDICT.md`](docs/VERDICT.md) | the claim/limit ledger: every claim with its measured value and scope |
| [`docs/CLAIMS.md`](docs/CLAIMS.md) | the accounting ledger for structural and cost claims |
| [`docs/PAPER_SPEC.md`](docs/PAPER_SPEC.md) | the gate table and the open items before submission |
| [`docs/REVIEW.md`](docs/REVIEW.md) | the objections a reviewer will raise, with the receipt that answers each |
| [`docs/NOVELTY.md`](docs/NOVELTY.md) | the overlap audit: headlines deliberately not claimed |

## Repository layout

```text
benchmarks/   harnesses and audits; each writes a machine-readable JSON receipt
artifacts/    150+ checked-in receipts, including the ones the figures are built from
scripts/      the runner scripts that produced them, with the exact flags
figures/      each figure as CSV (numbers, traceable to receipts) and SVG (the drawing)
ephemeralkv/  the library: durable span index, consolidation, state compilation
tests/        CPU tests; no GPU and no downloads
docs/         the manuscript and its ledgers
```

## Verifying what is here

```bash
python -m pytest tests/ -q                    # library semantics
python benchmarks/check_receipts.py            # every cited receipt exists, is complete, is tracked
python benchmarks/check_scripts.py             # every script's promised output exists, with the
                                              # flags its receipt recorded
python benchmarks/paper_numbers.py             # re-derives and asserts every headline number
python benchmarks/make_figures.py --outdir /tmp/figs && diff -r /tmp/figs figures
```

The corpora (4.9 GB of public agent traces) and the model checkpoints are not committed; both are
rebuildable, and `docs/REPRODUCING.md` gives the commands.  The measured arms need a GPU: they are in
`scripts/`, one runner per experiment family.

## Relationship to QCC

Deliberately separate from [`qcc-transformer`](https://github.com/Marchematics/qcc-transformer): QCC
asks how much live state *one query* needs, while this project asks when a long-lived session should
be free to move between workers.  No QCC code is copied here, and no claim depends on another
project's compiler.
