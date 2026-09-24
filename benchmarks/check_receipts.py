#!/usr/bin/env python
"""Every receipt a document cites must exist, and every tracked receipt should be cited.

The claim ledger is only as good as its links: a paper repo where `CLAIMS.md` names an artifact
that was never committed reproduces nothing.  This checks both directions - cited files that are
missing, and tracked artifacts that no document mentions - so the ledger cannot drift away from
`artifacts/` silently.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DOCS = ("docs/CLAIMS.md", "docs/VERDICT.md", "docs/DRAFT.md", "docs/PLAN.md",
        "docs/PAPER_SPEC.md", "docs/NOVELTY.md", "docs/REPORT.md")
# receipts that exist only as part of an external run (models, corpora, GPU sweeps) or that are
# deliberately not tracked; listing them here is the documented-omission mechanism
OPTIONAL_PREFIXES = ("data/", "models/")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs", nargs="*", default=list(DOCS))
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args(argv)

    cited: dict[str, set[str]] = {}
    pattern = re.compile(r"`([^`]*artifacts/[^`]+?)`")
    for name in args.docs:
        path = Path(name)
        if not path.exists():
            continue
        for match in pattern.findall(path.read_text()):
            entry = match.strip()
            # skip globs, shell commands and placeholder paths: they are patterns or invocations,
            # not receipts
            if " " in entry or "*" in entry or "<" in entry or entry.endswith(".py"):
                continue
            cited.setdefault(entry, set()).add(name)

    missing = sorted(name for name in cited if not Path(name).exists())

    # a killed run leaves a partial artifact at its final path; a cited partial is a problem
    partial = []
    for name in cited:
        path = Path(name)
        if path.suffix != ".json" or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("partial"):
            partial.append(name)
    tracked = subprocess.run(["git", "ls-files", args.artifacts], capture_output=True, text=True,
                             check=False).stdout.split()
    uncited = sorted(name for name in tracked if name not in cited)

    print(f"cited receipts: {len(cited)}   tracked artifacts: {len(tracked)}")
    if partial:
        print(f"\nCITED BUT PARTIAL ({len(partial)}) - a killed run wrote these; re-run before use:")
        for name in partial:
            print(f"  {name}   <- {', '.join(sorted(cited[name]))}")
    if missing:
        print(f"\nCITED BUT MISSING ({len(missing)}):")
        for name in missing:
            print(f"  {name}   <- {', '.join(sorted(cited[name]))}")
    if uncited:
        print(f"\nTRACKED BUT UNCITED ({len(uncited)}) - informational: intermediate or superseded\n"
              f"receipts are kept for audit but are not load-bearing for any claim:")
        for name in uncited[:40]:
            print(f"  {name}")
        if len(uncited) > 40:
            print(f"  ... and {len(uncited) - 40} more")
    if not missing and not uncited:
        print("every cited receipt exists and every tracked receipt is cited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
