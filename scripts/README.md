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
* `run_remaining.sh`, `run_longhist2.sh` - the equal-budget references and the long-history column.

They assume the corpora under `data/` (not in the repository: 734 MB - 2.3 GB of public agent
traces) and the model checkpoints named inside each script.  Two environment facts are worth
knowing before running them: the GPU is shared, so a script waits for other harness processes with
`pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2"` (an unanchored pattern also matches the
shell that launched the script, which deadlocks the wait), and the harnesses write their artifact
incrementally, so a killed run leaves a *partial* receipt at the final path - `benchmarks/`
`check_receipts.py` reports any cited receipt whose payload is partial.
