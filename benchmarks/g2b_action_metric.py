#!/usr/bin/env python
"""A stronger end task than file mention: does the turn intend to *edit* the right file?

The end-task metric used in the main receipts is file-level mention: the recorded patch's files
appear in the generated turn.  That is weaker than the question a deployment cares about - would
the agent act on the right file - and weaker than the task-success metric other systems report.
This scores the same receipts one notch higher, offline, without any new model runs:

* `mention`   - a recorded path appears anywhere in the generated turn (the existing metric);
* `edit`      - the turn names an edit-type tool or command (str_replace_editor, apply_patch,
                edit/create/insert, sed -i, cat >, patch, git apply);
* `action_hit`- both, i.e. the turn intends to modify a file the recorded patch touched.

It is not task success - it does not check whether the edit is correct - but it separates "the
model was thinking about the file" from "the model was about to change it", and it costs nothing
because the continuations are already in the artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

EDIT_PATTERNS = (
    "str_replace", "apply_patch", "git apply", "sed -i", "cat >", "tee ", "<<'EOF'", '<<"EOF"',
    '"edit"', '"create"', '"insert"', '"str_replace"', "'edit'", "'create'", "'insert'",
    "patch -p", ">>",
)
PATH_RE = re.compile(r"[\w./-]+\.[A-Za-z0-9]{1,6}")


def paths_in(text: str) -> set[str]:
    return {match.group(0) for match in PATH_RE.finditer(text or "")}


def score(continuation: str, recorded: list[str]) -> dict:
    basenames = {Path(name).name for name in recorded}
    mentioned = {name for name in basenames if name in (continuation or "")}
    edit_intent = any(pattern in (continuation or "") for pattern in EDIT_PATTERNS)
    # a path inside the turn that ends with one of the recorded basenames counts as targeting it
    targeted = {name for name in basenames
                if any(p.endswith(name) for p in paths_in(continuation))}
    return {"mention": bool(mentioned), "edit_intent": edit_intent,
            "action_hit": bool(targeted) and edit_intent}


def paired(diffs: list[float], repeats: int = 20000, seed: int = 0):
    if not diffs:
        return None
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
                   for _ in range(repeats))
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("receipts", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = []
    for name in args.receipts:
        path = Path(name)
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload.get("partial"):
            print(f"skipping partial receipt {name}")
            continue
        arm = path.stem.replace("g2b-patch-localization-", "")
        paired_active, paired_full = [], []
        counts = {"full": [0, 0, 0], "active": [0, 0, 0]}
        for row in payload["rows"]:
            recorded = row.get("recorded_files") or []
            if not recorded or "next_turn" not in (row.get("full") or {}):
                continue
            for side in ("full", "active"):
                scored = score((row.get(side) or {}).get("continuation", ""), recorded)
                counts[side][0] += scored["mention"]
                counts[side][1] += scored["edit_intent"]
                counts[side][2] += scored["action_hit"]
                (paired_active if side == "active" else paired_full).append(scored["action_hit"])
        n = len(paired_active)
        if not n:
            continue
        diffs = [a - b for a, b in zip(paired_active, paired_full)]
        ci = paired(diffs)
        rows.append({
            "arm": arm, "instances": n,
            "full_action_rate": round(sum(paired_full) / n, 3),
            "active_action_rate": round(sum(paired_active) / n, 3),
            "paired_action_delta": round(sum(diffs) / n, 3),
            "ci95": [round(ci[0], 3), round(ci[1], 3)] if ci else None,
            "active_edit_intent": counts["active"][1],
            "full_edit_intent": counts["full"][1],
        })
    payload = {"schema": "ephemeral-kv-action-metric-v1", "kind": "offline_rescoring",
               "interpretation": ("action_hit = the generated turn names an edit-type tool or "
                                  "command and targets a file the recorded patch touched"),
               "rows": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{'arm':<44} {'n':>3} {'full':>6} {'active':>7} {'paired':>7}  ci95")
    for row in rows:
        print(f"{row['arm'][:42]:<44} {row['instances']:>3} {row['full_action_rate']:>6.3f} "
              f"{row['active_action_rate']:>7.3f} {row['paired_action_delta']:>+7.3f}  {row['ci95']}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
