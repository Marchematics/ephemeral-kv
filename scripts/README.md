# Reproduction scripts

These are the runner scripts that produced the receipts in `artifacts/`, copied here so the
evidence path is inside the repository rather than in a working directory.  Each script is a
sequence of harness invocations with the exact flags used; the artifact it writes is the artifact
the paper cites.  They are numbered by the family they belong to rather than in execution order:

* `run_tailstate_sweep3.sh`, `run_tail_rerun.sh`, `run_window_plus_compiler.sh`,
  `run_fixed_window.sh`, `run_window3k_raw.sh`, `run_budget_lower.sh` - the compiler ladder, the
  window sweeps and the budget floor (figures 2, 3 and 6);
* `run_joint_n96.sh` - the central quality claim at 96 paired instances;
* `run_compaction.sh` - the model-written compaction baseline;
* `run_g4_grid2.sh`, `run_g4_geometry.sh`, `run_g4_pressure.sh`, `run_g4_hotspot.sh` - the routing
  replay grids whose union is figure 4;
* `run_g4_join.sh` - the join between those grids and the measured quality points, which is what
  turns a cost result into a quality-admissible region (and the file a reader checks when asking
  which active sizes the region may claim);
* `run_figure_inputs.sh` - the decision arms figure 3 is drawn from and the equal-budget references
  around them, reconstructed from each receipt's own `config` block;
* `run_floor_3584.sh` - the arms that moved the state floor from 6,144 to 4,096 tokens: a
  3,584-token window inside a 4,096-token total holds the surface (-0.96 pp) where a 3,072-token
  window at the same total does not (-2.40 pp).  This is what put the state on the routing replay's
  4,096-token column, whose 63 advancing cells took the admissible region from 54 to 117;
* `run_g4_size6144.sh` - the 6,144-token column, kept as the configuration a deployment would run
  when it wants the whole 2 pp fidelity allowance in reserve;
* `run_composed.sh`, `run_killer_composed.sh` - the composed long histories (145K-1.07M tokens) and
  the killer table over them: state tokens and bytes, the decision, and the fidelity reference.  A
  composed session is real transcripts concatenated, which is labelled in the corpus and the
  receipts because a million-token session is not something the corpus contains;
* `run_remaining.sh`, `run_longhist2.sh` - the equal-budget references and the long-history column.

They assume the corpora under `data/` (not in the repository: 734 MB - 2.3 GB of public agent
traces) and the model checkpoints named inside each script.  Two environment facts are worth
knowing before running them: the GPU is shared, so a script waits for other harness processes with
`pgrep -f "…benchmarks/g2_model_quality.py|…benchmarks/g2b_patch_localization.py"` - anchored to
the two GPU harnesses, because an unanchored pattern matches the launching shell (deadlocking the
wait) and a broader one also matches the CPU-only harnesses, which then serialise the GPU queue
behind measurements that do not use it, and the harnesses write their artifact
incrementally, so a killed run leaves a *partial* receipt at the final path - `benchmarks/`
`check_receipts.py` reports any cited receipt whose payload is partial.

Each script derives the repository root from its own location
(`cd "$(dirname "$0")/.."`, `export PYTHONPATH="$PWD"`), so a script run from a clone reads the
clone: hardcoding the original checkout path made a clone's drill exercise the wrong tree.  The
interpreter path (`/root/qcc/venv/bin/python`) and the model paths are the local environment's and
have to be adjusted for a different machine.

**Every output a script writes is tracked in `artifacts/`**, including the sweep arms that only
feed a figure's CSV, because a figure is regenerated *from* its receipts: `benchmarks/make_figures.py`
fails (exit 1) rather than writing a figure with an arm missing.  Receipts written after the
`config` block was added record the flags that produced them; `benchmarks/check_scripts.py`
reports the ones that predate it (`--config-audit`), which is what makes an arm's exact
configuration - not just its numbers - checkable from the repository alone.
