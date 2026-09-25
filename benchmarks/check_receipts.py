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
        "docs/PAPER_SPEC.md", "docs/NOVELTY.md", "docs/REPORT.md", "docs/REVIEW.md",
        "docs/PAPER.md", "docs/REPRODUCING.md")
# The figure generator is a citing document too: it names the receipts each figure is built from,
# and a figure whose input is untracked cannot be regenerated in a clone even though the figure's
# own CSV is committed.
FIGURE_SOURCES = ("benchmarks/make_figures.py",)
# receipts that exist only as part of an external run (models, corpora, GPU sweeps) or that are
# deliberately not tracked; listing them here is the documented-omission mechanism
OPTIONAL_PREFIXES = ("data/", "models/")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs", nargs="*", default=list(DOCS))
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args(argv)

    # A document may also cite a family of receipts with a brace glob
    # (`g2-compiler-window3k-b{4096,6144,12288}-v1.json`).  Left unexpanded, those citations were
    # invisible: the arms they name looked uncited, and - worse - a member of the family could go
    # missing without the receipt audit noticing, because the glob matches nothing on disk either.
    def expand(token: str) -> list[str]:
        match = re.search(r"\{([^{}]*)\}", token)
        if not match:
            return [token]
        return [expansion for alternative in match.group(1).split(",")
                for expansion in expand(token[:match.start()] + alternative + token[match.end():])]

    cited: dict[str, set[str]] = {}
    # Two citation styles have to be recognised.  A document may name a receipt in full
    # (`artifacts/x.json`) or, as the manuscript's claim map and ledger do, by its bare filename
    # (`x.json`).  Matching only the first style is what let a load-bearing receipt look uncited:
    # the audit's "cited" set was missing everything the paper's own table names.
    full = re.compile(r"`([^`]*artifacts/[^`]+?)`")
    # braces and commas are allowed so a family citation is captured whole and then expanded
    bare = re.compile(r"`([A-Za-z0-9][A-Za-z0-9._+{},-]*\.json)`")
    quoted = re.compile(r"\"([^\"]*artifacts/[^\"]+?\.json)\"")
    sources = [(name, full, bare) for name in args.docs]
    sources += [(name, quoted, None) for name in FIGURE_SOURCES]
    for name, first, second in sources:
        path = Path(name)
        if not path.exists():
            continue
        text = path.read_text()
        found = list(first.findall(text))
        if second is not None:
            for candidate in second.findall(text):
                for name_ in expand(candidate):
                    # a bare filename is a citation only if it resolves; a brace glob is a citation
                    # of its members whether or not they are present, so that a missing member is
                    # reported rather than silently skipped
                    if (Path(args.artifacts) / name_).exists() or "{" in candidate:
                        found.append(f"{args.artifacts}/{name_}")
        for match in found:
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

    # A receipt that exists only in the working directory is invisible to a reader: the cold-start
    # drill (clone the repository elsewhere and run this) is what exposed that class.
    untracked = sorted(name for name in cited
                       if Path(name).exists() and name not in tracked)
    print(f"cited receipts: {len(cited)}   tracked artifacts: {len(tracked)}")
    if untracked:
        print(f"\nCITED BUT UNTRACKED ({len(untracked)}) - present here, absent from a clone:")
        for name in untracked:
            print(f"  {name}   <- {', '.join(sorted(cited[name]))}")
    if partial:
        print(f"\nCITED BUT PARTIAL ({len(partial)}) - written incrementally, so this is a run that "
              f"was killed or is still in flight; re-run before quoting:")
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
