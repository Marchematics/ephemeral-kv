#!/usr/bin/env python
"""Compose long histories out of real sessions, so the cost law can be measured past the corpus.

The corpus's longest real session is ~156K tokens, so every row of the paper's mobility table past
that point is *composed from measured primitives* rather than measured.  That is the honest limit
the paper states - but the law it reports (a bounded execution state whose size does not track the
history) is exactly the claim a reader wants to see at 500K or 1M, not at 156K.

This script builds those sessions out of the real ones.  A composed session is a concatenation of
whole sessions from `data/sessions-patch-long.jsonl`, keeping the *last* session's final turn as the
end task's target, so the decision ground truth (the recorded patch's file set on the last turn) is
unchanged and still real.  Sessions are taken from the same repository where the corpus allows it -
a session that edits the same files over and over is what makes supersede semantics meaningful - and
the row records its provenance either way:

    session_id           composed::<bucket>::<repo>::<n>
    composed_from        the source session ids, in order
    composed_same_repo   whether every part is from one repository
    total_tokens         the sum of the parts' token counts
    n_turns              the sum of the parts' turn counts

What this is not: a claim that real sessions are this long.  Composed rows measure how the *state*
and the *decision* respond to history length with everything else held real (real transcripts, real
tool output, real patches); they are labelled in every receipt and figure that uses them.

Usage:
    python benchmarks/build_composed_long_sessions.py                 # write the buckets
    python benchmarks/build_composed_long_sessions.py --verify        # check what is on disk
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.g2_trace_index import messages_from_row          # noqa: E402
from benchmarks.g2b_patch_localization import patch_files_from_messages  # noqa: E402

# bucket label -> (target tokens, source rows per example cap)
BUCKETS = {"128k": 131072, "256k": 262144, "512k": 524288, "1m": 1048576}
EXAMPLES_PER_BUCKET = 12
SLACK = 1.3             # a composition may overshoot its target by this much


def repo_of(row: dict) -> str:
    try:
        meta = json.loads(row.get("ground_truth_meta_json") or "{}")
    except json.JSONDecodeError:
        return "unknown"
    instance = meta.get("instance_id") or ""
    return instance.split(".")[0] or "unknown"


def compose(parts: list[dict], bucket: str, index: int) -> dict:
    last = parts[-1]
    messages = []
    for part in parts:
        messages.extend(json.loads(part["messages_json"]))
    repos = {repo_of(part) for part in parts}
    # The end task's ground truth is the *final* turn's patch, so it has to be read from the last
    # part alone.  Derived from the concatenated transcript it would collect the file set of every
    # part (97 files at 1M instead of the last patch's), which makes the task easier the longer the
    # history is - a confound between length and ground-truth size that would flatter the result.
    target_files = sorted(patch_files_from_messages(json.loads(last["messages_json"])))
    return {
        "session_id": f"composed::{bucket}::{sorted(repos)[0]}::{index}",
        "composed_target_files": target_files,
        "composed_from": [part["session_id"] for part in parts],
        "composed_same_repo": len(repos) == 1,
        "composed_bucket": bucket,
        "source_dataset": "composed from " + ", ".join(sorted({p["source_dataset"] for p in parts})),
        "source_id": last.get("source_id"),
        "agent_framework": "composed",
        "ground_truth_meta_json": last.get("ground_truth_meta_json"),
        "messages_json": json.dumps(messages),
        "n_turns": sum(int(part.get("n_turns") or 0) for part in parts),
        "total_tokens": sum(int(part.get("total_tokens") or 0) for part in parts),
        "max_isl": max(int(part.get("max_isl") or 0) for part in parts),
    }


def build(source: Path) -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in source.open() if line.strip()]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row.get("total_tokens") or 0) > 0:
            by_repo[repo_of(row)].append(row)
    for group in by_repo.values():
        group.sort(key=lambda r: r["session_id"])

    out: dict[str, list[dict]] = {bucket: [] for bucket in BUCKETS}
    used: set[str] = set()
    for bucket, target in BUCKETS.items():
        # first pass: same-repository compositions, which is the realistic case
        for repo, group in sorted(by_repo.items()):
            if len(out[bucket]) >= EXAMPLES_PER_BUCKET:
                break
            parts: list[dict] = []
            total = 0
            for row in group:
                if row["session_id"] in used:
                    continue
                parts.append(row)
                total += int(row["total_tokens"])
                if total >= target:
                    break
            if parts and total >= target and total <= target * SLACK:
                out[bucket].append(compose(parts, bucket, len(out[bucket])))
                used.update(part["session_id"] for part in parts)
        # second pass: mix repositories for the buckets a single repo cannot reach
        if len(out[bucket]) < EXAMPLES_PER_BUCKET:
            pool = [row for group in by_repo.values() for row in group
                    if row["session_id"] not in used]
            pool.sort(key=lambda r: r["session_id"])
            index = 0
            while len(out[bucket]) < EXAMPLES_PER_BUCKET and index + 1 < len(pool):
                parts, total = [], 0
                while index < len(pool) and total < target:
                    row = pool[index]
                    index += 1
                    if row["session_id"] in used:
                        continue
                    parts.append(row)
                    total += int(row["total_tokens"])
                if parts and total >= target and total <= target * SLACK:
                    out[bucket].append(compose(parts, bucket, len(out[bucket])))
                    used.update(part["session_id"] for part in parts)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="data/sessions-patch-long.jsonl")
    p.add_argument("--outdir", default="data")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)

    if args.verify:
        for bucket in BUCKETS:
            path = Path(args.outdir) / f"sessions-composed-{bucket}.jsonl"
            if not path.exists():
                print(f"{bucket:>5}: absent")
                continue
            rows = [json.loads(line) for line in path.open() if line.strip()]
            tokens = [row["total_tokens"] for row in rows]
            turns = [row["n_turns"] for row in rows]
            same = sum(1 for row in rows if row.get("composed_same_repo"))
            print(f"{bucket:>5}: {len(rows):>2} rows  raw history p50 {statistics.median(tokens):>9,.0f} "
                  f"(min {min(tokens):,})  turns p50 {statistics.median(turns):>6,.0f}  "
                  f"same-repo {same}/{len(rows)}")
        return 0

    source = Path(args.source)
    if not source.exists():
        print(f"source corpus not found: {source}")
        return 2
    built = build(source)
    for bucket, rows in built.items():
        path = Path(args.outdir) / f"sessions-composed-{bucket}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        if rows:
            tokens = [row["total_tokens"] for row in rows]
            turns = [row["n_turns"] for row in rows]
            print(f"wrote {path} ({len(rows)} rows: raw history p50 {statistics.median(tokens):,.0f}, "
                  f"turns p50 {statistics.median(turns):,.0f})")
        else:
            print(f"wrote {path} (0 rows: the corpus has no combination that reaches this bucket)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
