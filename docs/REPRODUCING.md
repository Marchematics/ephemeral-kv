# Reproducing this repository

What is here, what is not, and the exact commands that re-derive every number in the paper.  Nothing
in this file requires a GPU until the last section.

## What is in the repository, and what is not

**In:** the harnesses (`benchmarks/`), the runner scripts that produced the receipts (`scripts/`),
the receipts themselves (`artifacts/`, 115 files), the figures (`figures/`, CSV plus SVG), the
library (`ephemeralkv/`) and its test suite (`tests/`), and the manuscript and its ledgers
(`docs/`).

**Not in:** the corpora and the model checkpoints.

* `data/` (4.9 GB) is generated, and `.gitignore`d.  The five derived corpora are *subsets of one
  downloaded corpus*, and both steps are scripted, so nothing about them has to be taken on trust:

  | file | rows | rule |
  |---|---:|---|
  | `sessions.jsonl` | 15,000 | downloaded source corpus (`benchmarks/build_trace_sessions.py`) |
  | `sessions-long.jsonl` | 5,758 | `max_isl >= 32768` |
  | `sessions-64k.jsonl` | 1,010 | `max_isl >= 65536` |
  | `sessions-128k.jsonl` | 6 | `max_isl >= 131072` |
  | `sessions-patch-long.jsonl` | 2,736 | patch filter, `max_isl >= 32768` |
  | `sessions-patch-64k.jsonl` | 338 | patch filter, `max_isl >= 65536` |

  The *patch filter* is the one the patch-localisation harness applies to its own examples: the
  row's ground-truth metadata says a patch is present and the patch's file set is recoverable from
  the transcript (`benchmarks/g2b_patch_localization.py`).  `data/sessions-patch-64k.jsonl` is the
  corpus behind the long-history column and the failover receipts; before
  `benchmarks/build_derived_corpora.py` existed, the rules above lived only in the shell history.

* Model checkpoints (2.4 GB Llama-3.2-1B-Instruct, 2.9 GB Qwen2.5-1.5B, 954 MB Qwen2.5-0.5B, 2.0 GB
  Phi-3.5-mini-instruct under `/root/qcc/models/`) are named by absolute path inside the runner
  scripts.  Those paths are this machine's; nothing else in the repository depends on them.  Note
  that `/root/qcc/models/Llama-3.2-3B-Instruct` is a 145-byte stub, not a checkpoint - it fails to
  load, and no receipt uses it.

## Rebuilding the corpora

```bash
python benchmarks/build_trace_sessions.py --out data/sessions.jsonl   # needs pyarrow + network
python benchmarks/build_derived_corpora.py                            # needs only the file above
python benchmarks/build_derived_corpora.py --verify                   # rules against what is there
```

`build_trace_sessions.py` documents the source dataset, the session count and the `no_proxy` detail
its HTTP client needs.  `--verify` is the check to run after copying corpora between machines: it
re-derives each corpus's session-id set from the source and compares it with the file present.

## The CPU-only verification path

Every headline number in the paper is re-derived from the receipts, and every figure is regenerated
from them, with no model and no GPU:

```bash
python -m pytest tests/ -q                       # 83 tests, library semantics
python benchmarks/check_receipts.py               # every cited receipt: exists, complete, tracked
python benchmarks/check_scripts.py                # every script's promised output exists
python benchmarks/check_scripts.py --config-audit # receipts that do not record their own flags
python benchmarks/paper_numbers.py                # re-derives and asserts every headline number
python benchmarks/make_figures.py --outdir /tmp/figs
python benchmarks/make_svg_figures.py --dir /tmp/figs
diff -r /tmp/figs figures                        # the figures are reproducible, not pasted
```

What each one is for:

* **`paper_numbers.py`** is the value audit: it reads the receipts and asserts the numbers the
  manuscript quotes, prints `PASS`/`FAIL` per claim and exits non-zero on any failure.  If a receipt
  is regenerated and a number moves, this is where it shows up.
* **`check_receipts.py`** is the receipt audit.  It reports a cited receipt that is missing, one
  whose payload is a *partial* write (the harnesses write incrementally, so a killed run leaves a
  partial artifact at its final path), and one that exists only in the working directory rather than
  in the repository.
* **`check_scripts.py`** is the other direction - script to output - and would have caught the
  failure that cost two rounds: a runner naming an `--out` path whose arm never started, because the
  harness rejected a flag and the queue had no `set -e`.  `--config-audit` lists receipts that do not
  record the flags that produced them; those predate the `config` block.
* **`make_figures.py`** rebuilds each figure's CSV from the receipts, and **`make_svg_figures.py`**
  draws the SVGs from those CSVs.  Missing inputs are an error rather than a figure with an arm
  silently dropped.

## The GPU path

The runner scripts in `scripts/` are the exact invocations that produced the receipts - same
instances, same flags, same output paths - grouped by experiment:

| scripts | what they produce | rough cost |
|---|---|---|
| `run_budget_lower.sh`, `run_window3k_raw.sh`, `run_fixed_window.sh`, `run_tailstate_sweep3.sh`, `run_tail_rerun.sh` | the budget floor, the window sweeps, the compiler ladder (figures 2, 3, 6) | 1-3 h each |
| `run_joint_n96.sh` | the central quality claim at 96 paired instances | ~2 h |
| `run_compaction.sh` | the model-written compaction baseline | ~3 h |
| `run_window_plus_compiler.sh` | the window/budget split, the equal-budget references | ~3 h |
| `run_g4_*.sh` | the routing replay grids (figure 4) | minutes each |
| `run_remaining.sh`, `run_longhist2.sh` | equal-budget references, the long-history column | 1-2 h each |

Practical notes, all of which cost time when learned the hard way:

* The GPU (a single 24 GiB A10G) is shared, and the corpora are large: a compaction arm at 33K-82K
  tokens of history takes 1-2 minutes per example and is sensitive to co-tenant load.  The runners
  wait for each other with `pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2"`; the pattern is
  anchored because an unanchored `pgrep -f` also matches the shell that launched the script, which
  deadlocks the wait.
* A killed run leaves a partial receipt at the final path.  That is expected, and
  `check_receipts.py` reports it; re-run the arm rather than quoting a partial.
* The runner scripts derive the repository root from their own location
  (`cd "$(dirname "$0")/.."`, `export PYTHONPATH="$PWD"`), so a script run from a clone reads the
  clone.  The interpreter path and the checkpoint paths are this machine's and need editing
  elsewhere.

## The cold-start drill

The check that the repository is self-contained: clone it somewhere else and run the CPU-only path
there.

```bash
rm -rf /tmp/coldstart && git clone -q <this repo> /tmp/coldstart && cd /tmp/coldstart
python -m pytest tests/ -q
python benchmarks/check_receipts.py
python benchmarks/check_scripts.py
python benchmarks/paper_numbers.py
python benchmarks/make_figures.py --outdir /tmp/coldfigs
python benchmarks/make_svg_figures.py --dir /tmp/coldfigs
diff -r /tmp/coldfigs figures
```

A clone has no `data/` and no checkpoints, so anything that needs them is out of scope for the
drill; everything else must pass.  The drill is what found four cited receipts that existed only in
the working directory, ten script outputs that were never tracked, figure-generation inputs that
were missing from the clone, and runners that resolved `PYTHONPATH` to the original checkout.

## What is not measured here

Stated in full in `docs/PAPER.md` Section 6; the ones a reader is most likely to look for:

* **task success is not measured.**  The end task is file-level localisation of the next turn against
  the recorded patch, plus an offline action-level rescoring; DeepSWE-style task success is the
  stronger evidence and is left as future work;
* **the routing consequence is simulated** from measured primitives (lookup, H2D bandwidth,
  active-set prefill) with declared arrival models, not measured on a cluster;
* **the corpus stops at 96 turns and 156K tokens.**  The 1M rows are composed with one extrapolated
  lookup input, and are labelled as such wherever they appear;
* **the 8B/70B capacity geometries are declared** from published KV geometries; the 0.5B geometry is
  measured.
