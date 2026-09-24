#!/usr/bin/env python
"""Paired bootstrap over the same sessions for two end-task arms.

The end-task corpus is 24 paired sessions, and differences of ~0.05 F1 are common noise at that
size (the 4,096 -> 8,192 comparison needed a bootstrap to separate +0.196 from zero).  Any claim
that one compiler arm beats another has to be paired on the same instances, so this reports the
paired mean difference, its 95% interval, and the win/loss split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def rows(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {r["instance_id"]: r["active"]["f1"] for r in payload["rows"] if "active" in r}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="baseline receipt")
    p.add_argument("--b", required=True, help="candidate receipt")
    p.add_argument("--label-a", default="a")
    p.add_argument("--label-b", default="b")
    p.add_argument("--repeats", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    a, b = rows(Path(args.a)), rows(Path(args.b))
    shared = sorted(set(a) & set(b))
    if not shared:
        print("no shared instances")
        return 1
    diffs = [b[i] - a[i] for i in shared]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    rng = random.Random(args.seed)
    means = []
    for _ in range(args.repeats):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    mean = sum(diffs) / len(diffs)
    print(f"{args.label_a} -> {args.label_b}: n={len(shared)} paired mean {mean:+.3f} "
          f"95% CI [{lo:+.3f}, {hi:+.3f}] wins {wins} losses {losses} "
          f"({'separated from zero' if lo > 0 or hi < 0 else 'includes zero'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
