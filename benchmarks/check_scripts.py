#!/usr/bin/env python
"""Does every reproduction script's output actually exist, and is it complete?

The receipt audit checks that artifacts the *documents* cite exist.  This checks the other
direction, and it is the check that would have caught the failure that cost two rounds: a runner
script listing an `--out artifacts/...` path whose arm never started, because a harness rejected a
flag and the queue had no `set -e`.  A script that has been "queued" and one that is still running
look identical; a missing output file does not.

Reports, per script: outputs that exist, outputs that are partial writes, and outputs that are
missing (which may simply mean the script has not been run yet - the point is that the question is
asked).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OUT_RE = re.compile(r"--out\s+\"?([^\s\"\\]+\.json)\"?")

# Arms that were designed, run and then superseded by a later variant of the same measurement.
# Listing them here is the documented way to say "this output is not expected": anything else
# missing is an arm that never produced its receipt.
SUPERSEDED = {
    "artifacts/g2b-patch-localization-tailstate-tf0.6-b8192-v1.json": "superseded by -v3 (v1-v2 predate the target-rebinding fix)",
    "artifacts/g2-compiler-tailstate-tf0.4-b8192-v1.json": "superseded by the tf0.35/0.5/0.6 sweep",
    "artifacts/g2b-patch-localization-tailstate-tf0.75-b4096-v1.json": "superseded by the tail_query design at 4,096",
    "artifacts/g2b-patch-localization-tailstate-tf0.9-b2048-v1.json": "superseded by the tail_query design at 2,048",
    "artifacts/g2-compiler-tailstate-tf0.75-b4096-v1.json": "superseded by the window3k arms",
    "artifacts/g2-compiler-tailstate-tf0.9-b2048-v1.json": "superseded by the window3k arms",
    "artifacts/g2b-patch-localization-compiled-b8192-long-v1.json": "superseded by -long-v2 on the pre-filtered corpus",
    "artifacts/g2-compiler-tailstate-tf0.5-b8192-long-v1.json": "superseded by the window3k long-history arm",
    "artifacts/g2b-patch-localization-protectwhole-b8192-long-v1.json":
        "the long-history decision column is covered by the raw and compiled long arms; "
        "protect-whole was measured at the short length, where it scores -20.33 pp of fidelity",
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scripts", default="scripts")
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--config-audit", action="store_true",
                   help="list complete receipts that do not record the flags that produced them")
    args = p.parse_args(argv)

    rows = []
    for script in sorted(Path(args.scripts).glob("*.sh")):
        text = script.read_text()
        outputs = []
        for match in OUT_RE.finditer(text):
            path = match.group(1)
            if "$" in path:
                continue            # a shell template, not a fixed output
            if path not in outputs:
                outputs.append(path)
        for path in outputs:
            target = Path(path)
            recorded = False
            if path in SUPERSEDED:
                state = "superseded"
            elif not target.exists():
                state = "missing"
            else:
                try:
                    payload = json.loads(target.read_text())
                    state = "partial" if payload.get("partial") else "complete"
                    # A receipt that does not record the flags that produced it cannot be
                    # reproduced, and from the rows alone some arms are indistinguishable: the
                    # compaction arm caps its view at 6,656 whether the window was 3,072 tokens
                    # with a 512-token summary or 5,120 with 1,536.
                    recorded = isinstance(payload.get("config"), dict)
                except (json.JSONDecodeError, OSError):
                    state = "unreadable"
            rows.append({"script": script.name, "output": path, "state": state,
                         "config_recorded": recorded})

    counts = {"complete": 0, "partial": 0, "missing": 0, "unreadable": 0, "superseded": 0}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    for row in rows:
        if args.only_missing and row["state"] in ("complete", "superseded"):
            continue
        if args.config_audit and (row["state"] != "complete" or row["config_recorded"]):
            continue
        note = "   [no config recorded]" if row["state"] == "complete" and not row["config_recorded"] else ""
        print(f"{row['state']:<10} {row['script']:<32} {row['output']}{note}")
    complete = [r for r in rows if r["state"] == "complete"]
    with_config = sum(1 for r in complete if r["config_recorded"])
    print(f"\n{len(rows)} outputs across the scripts: " +
          ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"complete receipts recording their run configuration: {with_config}/{len(complete)} "
          f"(the rest predate the config block; --config-audit lists them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
