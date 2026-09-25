#!/usr/bin/env python
"""Derive the corpora the harnesses read from the downloaded trace corpus.

`build_trace_sessions.py` documents where `data/sessions.jsonl` comes from.  The files the G2/G5
runners actually open are *subsets* of it, and until this script existed the rules that produced
them lived only in the commands that were typed once: a reader could rebuild the 15,000-session
source and still not rebuild `data/sessions-patch-64k.jsonl`, which is the corpus behind the
long-history column and the failover receipts.

The rules, all on the source corpus's own metadata:

| output                       | rows  | rule                                                          |
|------------------------------|-------|---------------------------------------------------------------|
| `sessions-long.jsonl`        | 5758  | `max_isl >= 32768`                                            |
| `sessions-64k.jsonl`         | 1010  | `max_isl >= 65536`                                            |
| `sessions-128k.jsonl`        | 6     | `max_isl >= 131072`                                           |
| `sessions-patch-long.jsonl`  | 2736  | patch filter and `max_isl >= 32768`                           |
| `sessions-patch-64k.jsonl`   | 338   | patch filter and `max_isl >= 65536`                           |

`max_isl` is the longest single input length the recorded agent ran with, which is why the length
cuts are on it rather than on `total_tokens` (a session can total 100K tokens in turns that never
exceed 8K each, and such a session does not exercise a long context).

The *patch filter* is the one `g2b_patch_localization.py` applies to its own examples, not a
re-implementation: the row's ground-truth metadata says a patch is present **and** the patch's file
set is recoverable from the transcript's tool outputs (`patch_files_from_messages`).  A subset that
kept sessions whose recorded patch cannot be read back would put examples in the corpus that the
harness then skips, so the corpus and the harness filter for the same thing.

Rows are written with all their fields, so nothing a harness reads (`ground_truth_meta_json`
carries the end-task ground truth) is dropped on the way through.  The files are large - 4.9 GB
together - and are not committed; what is committed is this script and the receipts produced from
its output.

Usage:
    python benchmarks/build_derived_corpora.py                       # write into data/
    python benchmarks/build_derived_corpora.py --verify              # check the rules against
                                                                     # the files already there
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g2_trace_index import messages_from_row           # noqa: E402
from g2b_patch_localization import patch_files_from_messages  # noqa: E402

# name -> (row count, minimum max_isl, needs the patch filter)
CORPORA = {
    "sessions-long.jsonl": (5758, 32768, False),
    "sessions-64k.jsonl": (1010, 65536, False),
    "sessions-128k.jsonl": (6, 131072, False),
    "sessions-patch-long.jsonl": (2736, 32768, True),
    "sessions-patch-64k.jsonl": (338, 65536, True),
}


def is_patch_session(row: dict) -> bool:
    """The patch filter, applied exactly as the patch-localisation harness applies it."""
    try:
        meta = json.loads(row.get("ground_truth_meta_json") or "{}")
    except json.JSONDecodeError:
        return False
    if not meta.get("patch_present"):
        return False
    try:
        messages = messages_from_row(row)
    except ValueError:
        return False
    return bool(patch_files_from_messages(messages))


def select(source: Path, min_isl: int, patch: bool):
    with source.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("max_isl") or 0) < min_isl:
                continue
            if patch and not is_patch_session(row):
                continue
            yield row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="data/sessions.jsonl")
    p.add_argument("--outdir", default="data")
    p.add_argument("--verify", action="store_true",
                   help="compare each rule's session-id set with the file already present")
    p.add_argument("--only", default="", help="one corpus name, for a quick check")
    args = p.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"source corpus not found: {source} (see benchmarks/build_trace_sessions.py)")
        return 2

    failed = False
    for name, (want_rows, min_isl, patch) in CORPORA.items():
        if args.only and args.only != name:
            continue
        target = Path(args.outdir) / name
        if args.verify:
            if not target.exists():
                print(f"{name:<28} absent (nothing to verify against)")
                failed = True
                continue
            existing = {json.loads(line)["session_id"] for line in target.open() if line.strip()}
            derived = {row["session_id"] for row in select(source, min_isl, patch)}
            ok = existing == derived
            failed = failed or not ok
            print(f"{name:<28} rule rows {len(derived):>5}  file ids {len(existing):>5}  "
                  f"{'same id set' if ok else 'MISMATCH'}  "
                  f"(expected {want_rows} rows)")
            continue
        rows = 0
        with target.open("w") as out:
            for row in select(source, min_isl, patch):
                out.write(json.dumps(row) + "\n")
                rows += 1
        note = "" if rows == want_rows else f"  (the paper's runs used {want_rows})"
        print(f"wrote {target} ({rows} rows){note}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
