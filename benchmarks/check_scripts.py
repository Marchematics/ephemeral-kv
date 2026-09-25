#!/usr/bin/env python
"""Does every reproduction script's output actually exist, and is it complete?

The receipt audit checks that artifacts the *documents* cite exist.  This checks the other
direction, and it is the check that would have caught the failure that cost two rounds: a runner
script listing an `--out artifacts/...` path whose arm never started, because a harness rejected a
flag and the queue had no `set -e`.  A script that has been "queued" and one that is still running
look identical; a missing output file does not.

Reports, per script: outputs that exist, outputs that are partial writes, and outputs that are
missing (which may simply mean the script has not been run yet - the point is that the question is
asked).  It also compares the flags in each script's invocation against the `config` block the
receipt recorded: the check that catches a runner whose flags have drifted away from the run the
paper cites.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OUT_RE = re.compile(r"--out\s+\"?([^\s\"\\]+\.json)\"?")

# harness -> {flag: (config key, is_boolean)}, read from the harness source itself so the audit does
# not keep a second copy of the flag names that can drift from the harness's own
HARNESSES = {"benchmarks/g2_model_quality.py": {}, "benchmarks/g2b_patch_localization.py": {}}
ARG_RE = re.compile(r'add_argument\("(--[a-z0-9-]+)"([^\n]*)')


def harness_flags(path: str) -> dict:
    """{flag: (config key, boolean?)} for one harness, from its argparse declarations."""
    flags = {}
    text = Path(path).read_text()
    for flag, rest in ARG_RE.findall(text):
        key = flag[2:].replace("-", "_")
        default = re.search(r"default=([^,)]+)", rest)
        literal = default.group(1).strip() if default else ""
        boolean = literal in ("True", "False") or "store_true" in rest
        flags[flag] = (key, boolean)
    return flags


def shell_arrays(text: str) -> dict:
    """`name=(a b c)` assignments, so an invocation using "${name[@]}" can be expanded."""
    joined = text.replace("\\\n", " ")
    return {match.group(1): match.group(2).split()
            for match in re.finditer(r"(\w+)=\(([^)]*)\)", joined, re.S)}


def invocations(text: str) -> list[list[str]]:
    """Shell commands in a script, with line continuations and arrays expanded."""
    arrays = shell_arrays(text)
    joined = text.replace("\\\n", " ")
    for name, values in arrays.items():
        for form in (f"${{{name}[@]}}", f"${{{name}[*]}}", f'"${{{name}[@]}}"'):
            joined = joined.replace(form, " ".join(values))
    return [[token.strip("\"'") for token in piece.split()]
            for piece in re.split(r"[\n;]", joined) if "--out" in piece]


def flags_of(tokens: list[str], flags: dict) -> dict:
    """The flags in one invocation, as {config key: value}."""
    out: dict = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        name, _, inline = token.partition("=")
        if name not in flags:
            index += 1
            continue
        key, boolean = flags[name]
        if boolean:
            out[key] = True
        elif inline:
            out[key] = inline
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            out[key] = tokens[index + 1]
            index += 1
        index += 1
    return out


def compare(config: dict, passed: dict, flags: dict) -> list[str]:
    """Flags whose scripted value disagrees with the configuration the receipt recorded."""
    problems = []
    for flag, (key, boolean) in sorted(flags.items()):
        if key not in config:
            continue
        want = config[key]
        if boolean:
            got = bool(passed.get(key, False))
        elif key in passed:
            raw = passed[key]
            got = type(want)(raw) if not isinstance(want, bool) else raw not in ("False", "0")
        else:
            continue            # not passed: the harness default applies, and config matches it
        if got != want:
            problems.append(f"{flag}: script {got!r} vs receipt {want!r}")
    return problems

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

    flag_problems = []
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
            payload_config = None
            if state == "complete":
                try:
                    payload_config = json.loads(target.read_text()).get("config")
                except (json.JSONDecodeError, OSError):
                    payload_config = None
            if isinstance(payload_config, dict):
                # the invocation that writes this output, and the flags it passes
                for tokens in invocations(text):
                    if not any(path in token for token in tokens):
                        continue
                    harness = next((token for token in tokens
                                    if token.endswith(".py") and "benchmarks/" in token), None)
                    if harness is None or harness not in HARNESSES:
                        continue
                    if not HARNESSES[harness]:
                        HARNESSES[harness] = harness_flags(harness)
                    passed = flags_of(tokens, HARNESSES[harness])
                    for problem in compare(payload_config, passed, HARNESSES[harness]):
                        flag_problems.append({"script": script.name, "output": path,
                                              "problem": problem})
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
    checked = sum(1 for r in rows if r["state"] == "complete" and r["config_recorded"])
    print(f"script flags compared against the recorded configuration: {checked} outputs, "
          f"{len(flag_problems)} disagreement(s)")
    for problem in flag_problems:
        print(f"  MISMATCH {problem['script']} -> {problem['output']}: {problem['problem']}")
    # a disagreement means the script no longer re-runs the arm the paper cites, which is how the
    # compaction baseline's window drifted from 6,656 to 3,072 without any number moving
    return 1 if flag_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
