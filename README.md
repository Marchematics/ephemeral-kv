# EphemeralKV

**The conversation is durable. The KV cache is disposable.**

A long-lived LLM session currently accumulates model state: every turn appends keys
and values, and serving systems treat that KV as an object worth *keeping* — pinned in
HBM, tiered to DRAM, offloaded to NVMe, restored on the next turn. This project tests
the opposite abstraction:

> The durable state of a session is its **transcript plus a cheap semantic index**.
> Model state is a **compile product** of the current request against that transcript,
> and it can die when the turn ends.

If that holds, the consequences are measurable and counterintuitive:

1. **Discard + recompile can be cheaper than restore.** Past the crossover, rebuilding
   a small working set from the transcript beats loading a large KV from HBM/DRAM/NVMe.
2. **Longer sessions get *relatively* cheaper.** Active state tracks the request's
   working set, not the session history, so a 1M-token agent session can hold an active
   HBM footprint close to a 128K session's.
3. **A longer request can occupy less GPU memory than a shorter one** — because the
   *working set*, not the history, sets the footprint.

## Relationship to QCC

This is a separate project from [qcc-transformer](https://github.com/Marchematics/qcc-transformer),
and deliberately so: QCC asks *how much state does one query need* (a single-request
context-compilation question). EphemeralKV asks *whether model state should be durable
at all* (an OS state-lifecycle question). **Nothing here may depend on QCC**: the
compiler is a pluggable interface, and a BM25 or embedding compiler is a first-class
participant. A QCC-shaped compiler is one column in the results, never a requirement.

## Status

Scaffold plus the kill-gate plan. No result is claimed yet: every headline above is a
hypothesis with a decision rule in [`docs/PLAN.md`](docs/PLAN.md), and the project is
designed to be killed cheaply if the gates fail — see
[`docs/CLAIMS.md`](docs/CLAIMS.md) for the ledger format.

## Repository layout

```
benchmarks/   one script per measurement; each writes JSON under artifacts/
docs/        PLAN.md (kill gates), CLAIMS.md (ledger), README.md (index)
tests/       CPU tests; no GPU and no downloads
```

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python benchmarks/kill_gate_crossover.py --help
```
