#!/usr/bin/env python
"""Next-action reproduction: does the state let the agent perform the recorded action?

The end-task metric this project uses elsewhere asks whether a 96-token continuation *names*
the files of the recorded patch.  That is a localization score, and it is weak evidence for a
systems claim: naming a file is not doing the work.  This harness scores the same continuations
as **executable actions** - the tool invocation an agent framework would actually dispatch.

The recorded next turn of a coding trajectory is a tool call, in one of two shapes:

    ```\\nstr_replace_editor view /testbed/funcy/decorators.py\\n```
    [{"function": {"arguments": "{\\"command\\": \\"cd /testbed && git diff\\"}", "name": "execute_bash"}}]

so a continuation can be parsed into `(tool, verb, target)` and compared with the recorded
`(tool, verb, target)`.  Three numbers come out, in order of strictness:

    tool     the same tool is invoked (bash against bash, the editor against the editor)
    target   the same file is addressed, for actions that address one
    exact    tool, verb and target all agree

This is not task success either - the action is not executed against a repository - but it would
be a step closer than file mentions: the decision the state has to support, in the form the
runtime consumes.

**Scope, measured rather than assumed: on the receipts that exist this instrument is
uninformative, and recording that is the point of keeping it.**  Those receipts generate at
`examples[-1]`, the last assistant turn of a session, and on this corpus that turn is a
session-ending submission in **48 of 48** cases (47 `submit`, one task tracker) with **no file
target at all** - the work is done, so the agent writes a summary and stops.  `target` therefore
has nothing to compare against, and the tool rate only asks whether a model that is *finishing* a
session also emits a `submit` call (measured: it does not - 8.7% with full history, 7.4% at the
bounded state).  An action-reproduction metric has to score **mid-session** action turns, the
editor call that changes a file, and no receipt contains a continuation for one of those.
Measuring it needs a harness that generates at chosen action turns, which is a different
measurement rather than a re-scoring of these.

Ground truth is rebuilt from the corpus rather than re-measured: `examples[-1]` of a session
does not depend on the token budget, so one pass per (corpus, min-history-tokens, tokeniser)
yields the recorded action for every session a receipt could have scored.  The build is verified
against each receipt's own recorded next-turn file set *and* its recorded history size, so a
misaligned cache fails loudly instead of silently scoring the wrong turn.

Usage
-----
    # build (and reuse) the ground-truth cache, then score a receipt
    python benchmarks/g2c_action_reproduction.py \\
        --receipt artifacts/g2b-patch-localization-window3584-b4096-n48.json \\
        --out artifacts/g2c-action-final-turn-window3584-n48.json
    # aggregate every action receipt into one table
    python benchmarks/g2c_action_reproduction.py --table artifacts/g2c-action-*.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import glob
import json
import random
import re
from pathlib import Path
from statistics import mean

from benchmarks.g2b_patch_localization import (_FILE_TOKEN, _matches, mentioned_files,
                                              patch_files_from_messages)
from benchmarks.g2_trace_index import _content, messages_from_row

# Ground-truth caches live with the corpus rather than beside the receipts.  They are keyed by the
# corpus, the history filter and the tokeniser - the only inputs `examples[-1]` depends on - and
# they are large (6.8 MB for the 2,734 sessions of `sessions-patch-long`), so a receipt that cites
# this instrument should not drag the whole transcript's token counts into the repository.
TRUTH_DIR = Path("data")

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)```", re.S)
# a tool call in the JSON shape several frameworks record: name plus escaped arguments.  The
# arguments are a *JSON string containing JSON*, so the pattern has to be escape-aware -
# `"((?:[^"\\]|\\.)*)"` - or it stops at the first escaped quote and truncates the payload
# (measured: `"command": "cd /x && git diff"` parsed as a command whose head was `{"command":`).
_JSON_NAME = re.compile(r'"name"\s*:\s*"([A-Za-z_][\w.\-]*)"')
_JSON_ARG = re.compile(r'"(?:arguments|args|input)"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
_JSON_ARG_OBJ = re.compile(r'"(?:arguments|args|input)"\s*:\s*(\{.*?\})\s*[,}]', re.S)
_JSON_KEYVAL = re.compile(r'"(command|cmd|path|file_path|filename|file)"\s*:\s*"((?:[^"\\]|\\.)*)"')
_JSON_KEYVAL_OBJ = re.compile(r'"(command|cmd|path|file_path|filename|file)"\s*:\s*"((?:[^"\\]|\\.)*)"')
_BARE = re.compile(r"^\s*(str_replace_editor|execute_bash|bash|submit)\b.*$", re.M)

EDITOR_VERBS = {"view", "create", "str_replace", "insert", "undo_edit"}

# one alias per tool family, so `execute_bash` and `bash` agree but the editor and the shell do
# not: invoking the wrong tool is a different failure from invoking the right one on the wrong file
_TOOL_FAMILY = {
    "str_replace_editor": "editor",
    "str_replace_editor.": "editor",
    "edit_file": "editor",
    "file_editor": "editor",
    "str_replace": "editor",
    "execute_bash": "bash",
    "bash": "bash",
    "run_command": "bash",
    "shell": "bash",
    "submit": "submit",
    "finish": "submit",
}


def tool_family(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().strip("`'\"").lower()
    return _TOOL_FAMILY.get(key, key)


def _path_token(text: str) -> str | None:
    """The file a shell command addresses, if it names one.

    Only tokens that look like a *file* count: the shared `_FILE_TOKEN` rule means `cd
    /workspace/pandas__pandas__5.19 && git diff` has no target (the checkout directory is not a
    file) while `python tests/test_io.py` does.  Without that rule every `cd <checkout> && ...`
    command would trivially "match" on the directory both sides share, and the target rate would
    measure the framework rather than the state.
    """
    for token in text.split():
        token = token.strip("'\"`,;()")
        if _FILE_TOKEN.fullmatch(token):
            return token
    return None


def _unescape_arguments(blob: str) -> str:
    """Decode a JSON string body, keeping the text when it is truncated mid-escape.

    Continuations are cut at a token limit, so an arguments payload is often unterminated; the
    partial text still names the tool and usually the file, which is what is being scored.
    """
    try:
        return json.loads('"' + blob + '"')
    except Exception:
        return blob.encode().decode("unicode_escape", errors="ignore")


def _keyvals(blob: str) -> dict:
    decoded = _unescape_arguments(blob)
    pairs = dict(_JSON_KEYVAL.findall(blob)) or dict(_JSON_KEYVAL_OBJ.findall(decoded))
    if not pairs:
        pairs = {k: _unescape_arguments(v) for k, v in _JSON_KEYVAL.findall(decoded)}
    else:
        pairs = {k: _unescape_arguments(v) for k, v in pairs.items()}
    return pairs


def _action_from_command(command: str) -> dict | None:
    """Parse one command line: `tool verb path ...` or a shell command."""
    line = command.strip().splitlines()[0].strip() if command.strip() else ""
    if not line:
        return None
    parts = line.split()
    head = parts[0].strip("'\"`")
    family = tool_family(head)
    if family == "editor":
        verb = parts[1] if len(parts) > 1 and parts[1] in EDITOR_VERBS else None
        target = None
        if verb and len(parts) > 2:
            target = parts[2].strip("'\"`")
        return {"tool": "editor", "raw_tool": head, "verb": verb, "target": target,
                "command": line}
    if family == "submit":
        return {"tool": "submit", "raw_tool": head, "verb": None, "target": None, "command": line}
    # anything else is a shell command: the family is the interpreter, the target is the path it
    # touches.  `cd /x && git diff` and `git diff` are the same action on different paths, which
    # is what the target comparison is for.
    return {"tool": "bash", "raw_tool": head, "verb": head,
            "target": _path_token(line), "command": line}


def parse_action(text: str) -> dict | None:
    """The first executable action in a continuation, or None if there is not one."""
    if not text:
        return None
    body = re.sub(r"^\s*<assistant>\s*", "", text)

    # JSON tool calls first: they are unambiguous, and a continuation that carries one usually
    # also carries prose that mentions tools
    name = _JSON_NAME.search(body)
    if name:
        family = tool_family(name.group(1))
        # the arguments precede the name in some frameworks and follow it in others
        arg = _JSON_ARG.search(body) or _JSON_ARG_OBJ.search(body)
        if arg:
            arg_blob = arg.group(1)
        else:
            arg_blob = body[name.end():name.end() + 800]
        keyval = _keyvals(arg_blob)
        value = keyval.get("command") or keyval.get("cmd")
        target = (keyval.get("path") or keyval.get("file_path")
                  or keyval.get("filename") or keyval.get("file"))
        if family == "editor" or (value and "str_replace_editor" in str(value)):
            parsed = _action_from_command(value or "")
            if parsed:
                return parsed
        if family == "bash":
            command = value if value is not None else arg_blob
            parsed = _action_from_command(str(command))
            if parsed:
                parsed["tool"] = "bash"
                parsed["raw_tool"] = name.group(1)
                return parsed
        return {"tool": family, "raw_tool": name.group(1), "verb": None,
                "target": target.strip() if isinstance(target, str) else None,
                "command": str(value or arg_blob)[:400]}

    fence = _FENCE.search(body)
    if fence:
        parsed = _action_from_command(fence.group(1))
        if parsed:
            return parsed
    bare = _BARE.search(body)
    if bare:
        return _action_from_command(bare.group(0))
    return None


def score_action(continuation: str, recorded: dict | None) -> dict:
    """Compare a continuation with the recorded next action."""
    generated = parse_action(continuation)
    out = {
        "generated": generated,
        "recorded": recorded,
        "parseable": generated is not None,
        "tool": None,
        "target": None,
        "exact": None,
    }
    if generated is None or recorded is None:
        return out
    out["tool"] = generated["tool"] == recorded["tool"]
    # the target comparison only applies when the recorded action addresses a file; a `cd x && ...`
    # whose recorded counterpart is `submit` is not a localization failure
    if recorded.get("target") and generated.get("target"):
        out["target"] = bool(_matches(generated["target"].lstrip("/"),
                                      recorded["target"].lstrip("/")))
    out["exact"] = bool(out["tool"] and (out["target"] is not False)
                        and (generated.get("verb") == recorded.get("verb")
                             or recorded.get("verb") is None))
    return out


# --------------------------------------------------------------------------------------------
# ground truth


def build_groundtruth(jsonl: Path, min_history_tokens: int, token_counter, limit: int = 0) -> list[dict]:
    """The recorded next action for every session a receipt on this corpus could have scored.

    Returns a *list in corpus order*, because that is the only key the receipts share.  An
    `instance_id` is not unique here (384 of 2,293 are reused: the corpus carries several
    trajectories for the same task) and the earlier receipts predate the `session_id` field being
    recorded at all, so a dictionary keyed by either silently scores the wrong turn for a duplicate
    - measured: 3 of 48 rows disagreed with the receipt's own record, and the rebuilt turn's
    history was larger than the whole transcript, which is how the collision was found.  Position
    is exact: the harness walks the file in order and stops at `max_examples`.

    The filters are replicated from `g2b_patch_localization.main`: rows without a recorded patch
    and sessions whose history never passes `min_history_tokens` are skipped, and the target is the
    last qualifying assistant turn.  `history_tokens` is returned with it so the pairing can be
    verified rather than assumed.
    """
    out: list[dict] = []
    seen = 0
    with jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = json.loads(row.get("ground_truth_meta_json") or "{}")
            if not meta.get("patch_present"):
                continue
            try:
                messages = messages_from_row(row)
            except ValueError:
                continue
            if not (row.get("composed_target_files") or patch_files_from_messages(messages)):
                continue
            instance_id = meta.get("instance_id")
            texts = [_content(m) for m in messages]
            counts = _count_all(token_counter, texts)
            history_tokens = 0
            spans = 0
            last: str | None = None
            last_hist = 0
            # the newest user/tool event *is* the query, empty or not: an empty one makes the
            # harness skip the turn rather than reach further back, and this must agree with it
            query = ""
            for msg, text, count in zip(messages, texts, counts):
                role = str(msg.get("role", "unknown"))
                if role == "assistant" and text.strip() and spans >= 8 and query:
                    if history_tokens >= min_history_tokens:
                        last, last_hist = text, history_tokens
                if role in {"user", "tool"}:
                    query = text
                history_tokens += max(1, int(count))
                spans += 1
            if last is None:
                continue
            out.append({"instance_id": instance_id, "target": last,
                        "history_tokens": last_hist,
                        "session_id": row.get("session_id")})
            seen += 1
            if limit and seen >= limit:
                break
    return out


def _count_all(token_counter, texts: list[str]) -> list[int]:
    """Token count per text, batched when the counter supports it."""
    batch = getattr(token_counter, "batch", None)
    if batch is not None:
        return batch(texts)
    return [int(token_counter(t)) for t in texts]


class _TokenizerCounter:
    """`len(encode(text))` per text, with one batched call per session."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def batch(self, texts: list[str]) -> list[int]:
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]


def _tokenizer_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", Path(model).name or "tokenizer")


def groundtruth_path(jsonl: str, min_history_tokens: int, tokenizer_tag: str) -> Path:
    stem = Path(jsonl).stem
    return TRUTH_DIR / f"g2c-next-action-truth-{stem}-{min_history_tokens}-{tokenizer_tag}.json"


def load_groundtruth(jsonl: str, min_history_tokens: int, tokenizer_tag: str, token_counter,
                     rebuild: bool = False) -> list[dict]:
    path = groundtruth_path(jsonl, min_history_tokens, tokenizer_tag)
    if path.exists() and not rebuild:
        payload = json.loads(path.read_text())
        if payload.get("schema") == "ephemeral-kv-g2c-next-action-truth-v2":
            return payload["sessions"]
    cache = build_groundtruth(Path(jsonl), min_history_tokens, token_counter)
    path.write_text(json.dumps(
        {"schema": "ephemeral-kv-g2c-next-action-truth-v2",
         "config": {"jsonl": jsonl, "min_history_tokens": min_history_tokens,
                    "tokenizer": tokenizer_tag},
         "sessions": cache}, indent=2) + "\n")
    print(f"[g2c] built ground truth for {len(cache)} sessions -> {path}")
    return cache


# --------------------------------------------------------------------------------------------
# rescoring

_TOKENIZERS: dict[str, object] = {}


def _load_tokenizer(model: str):
    """The tokeniser a receipt scored with, or None when it cannot be loaded.

    The history filter that selects the ground-truth turn is expressed in this tokeniser's tokens,
    so the cache has to be built with it.  A missing checkpoint degrades to whitespace counts and
    the alignment check below is then the only thing standing between the cache and a wrong turn,
    which is why a mismatch is fatal rather than a warning.
    """
    if not model:
        return None
    if model not in _TOKENIZERS:
        try:
            from transformers import AutoTokenizer

            _TOKENIZERS[model] = AutoTokenizer.from_pretrained(model, use_fast=True)
        except Exception as exc:                                     # pragma: no cover
            print(f"[g2c] tokenizer unavailable for {model}: {exc}")
            _TOKENIZERS[model] = None
    return _TOKENIZERS[model]


def rescore(receipt_path: Path, rebuild_truth: bool = False) -> dict:
    receipt = json.loads(receipt_path.read_text())
    config = receipt.get("config") or {}
    jsonl = config.get("jsonl")
    min_history_tokens = int(config.get("min_history_tokens") or 0)
    model = config.get("model") or ""
    if not jsonl:
        raise SystemExit(f"{receipt_path}: receipt has no config.jsonl")
    tokenizer = _load_tokenizer(model)
    tag = _tokenizer_tag(model)
    if tokenizer is not None:
        counter = _TokenizerCounter(tokenizer)
    else:
        counter = lambda text: max(1, len(str(text).split()))
    truth = load_groundtruth(jsonl, min_history_tokens, tag if tokenizer is not None else "split",
                             counter, rebuild=rebuild_truth)

    rows, mismatches, missing = [], 0, 0
    for position, row in enumerate(receipt.get("rows") or []):
        if position >= len(truth):
            missing += 1
            continue
        entry = truth[position]
        recorded_text = entry["target"]
        # Two independent checks that the rebuilt turn is the one this receipt scored.  The file
        # mentions are what the receipt recorded about its own ground truth; the history size is
        # the quantity that selected the turn in the first place.  Either disagreeing means the
        # cache is off by a turn and every number below would be about a different question.
        stored = (row.get("full") or {}).get("next_turn")
        if row.get("history_tokens") is not None \
                and int(row["history_tokens"]) != int(entry["history_tokens"]):
            mismatches += 1
            continue
        if stored is not None and \
                sorted(mentioned_files(recorded_text)) != stored.get("recorded"):
            mismatches += 1
            continue
        recorded = parse_action(recorded_text)
        scored = {"instance_id": row.get("instance_id"), "position": position,
                  "recorded_action": recorded,
                  "recorded_text": recorded_text[:400]}
        for arm in ("full", "active"):
            if arm in row:
                scored[arm] = {**score_action((row[arm] or {}).get("continuation", ""), recorded),
                               "f1": (row[arm] or {}).get("f1")}
        rows.append(scored)

    if mismatches:
        raise SystemExit(
            f"{receipt_path}: {mismatches} rows disagree with the rebuilt ground truth - the "
            "cache does not describe the turns this receipt scored")
    summary = summarize(rows)
    return {
        "schema": "ephemeral-kv-g2c-action-reproduction-v1",
        "kind": "executable_action_measurement",
        "source_receipt": str(receipt_path),
        "config": {"jsonl": jsonl, "min_history_tokens": min_history_tokens,
                   "rows_in_receipt": len(receipt.get("rows") or []),
                   "rows_scored": len(rows), "rows_without_groundtruth": missing,
                   "source_config": config},
        "summary": summary,
        "rows": rows,
        "interpretation": (
            "the generated continuation parsed as a tool invocation and compared with the "
            "recorded next action; `target` is scored only where the recorded action addresses "
            "a file, so a session-ending `submit` is not a localization failure.  On this corpus "
            "the recorded next action is a session-ending submission in almost every row, so the "
            "tool rate measures whether a finishing agent emits a `submit` call and nothing about "
            "mid-session action reproduction - see the module docstring"),
    }


def _rate(rows, arm, key):
    vals = [r[arm][key] for r in rows if arm in r and r[arm].get(key) is not None]
    return mean(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    kinds: dict[str, int] = {}
    for row in rows:
        action = row.get("recorded_action")
        if action:
            kinds[action["tool"]] = kinds.get(action["tool"], 0) + 1
    out = {"examples": len(rows),
           "recorded_actions_parseable": sum(1 for r in rows if r.get("recorded_action")),
           # the receipt records only the *last* turn of each session, and on a coding corpus that
           # turn is the agent submitting.  Publishing the mix makes the degenerate scope visible
           # in the artifact: a `target` rate over rows that address no file is not a result.
           "recorded_action_kinds": dict(sorted(kinds.items())),
           "recorded_actions_with_a_file_target": sum(
               1 for r in rows if (r.get("recorded_action") or {}).get("target"))}
    for arm in ("full", "active"):
        present = [r for r in rows if arm in r]
        if not present:
            continue
        out[arm] = {
            "parseable": _rate(present, arm, "parseable"),
            "tool": _rate(present, arm, "tool"),
            "target": _rate(present, arm, "target"),
            "exact": _rate(present, arm, "exact"),
            "f1": _rate(present, arm, "f1"),
        }
    return out


def bootstrap_ci(diffs: list[float], repeats: int = 20000, seed: int = 0):
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(repeats):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]


def paired(receipts: list[dict], key: str = "target") -> dict:
    """Paired full-vs-active over the instances both arms scored, per receipt and pooled."""
    per = []
    pooled: dict[str, list[float]] = {}
    for payload in receipts:
        rows = [r for r in payload["rows"] if "full" in r and "active" in r]
        diffs = [float(bool(r["active"].get(key))) - float(bool(r["full"].get(key)))
                 for r in rows if r["active"].get(key) is not None
                 and r["full"].get(key) is not None]
        if not diffs:
            continue
        ci = bootstrap_ci(diffs)
        per.append({"receipt": Path(payload["source_receipt"]).name, "metric": key,
                    "n": len(diffs), "mean_diff": mean(diffs),
                    "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                    "wins": sum(1 for d in diffs if d > 0),
                    "losses": sum(1 for d in diffs if d < 0)})
        for r, d in zip([r for r in rows if r["active"].get(key) is not None
                         and r["full"].get(key) is not None], diffs):
            pooled.setdefault(r["instance_id"], []).append(d)
    if pooled:
        diffs = [mean(v) for v in pooled.values()]
        ci = bootstrap_ci(diffs)
        per.append({"receipt": "POOLED over instances", "metric": key, "n": len(diffs),
                    "mean_diff": mean(diffs),
                    "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                    "wins": sum(1 for d in diffs if d > 0),
                    "losses": sum(1 for d in diffs if d < 0)})
    return {"metric": key, "comparisons": per}


def table(paths: list[str]) -> int:
    payloads = [json.loads(Path(p).read_text()) for p in sorted(paths)]
    header = f"{'receipt':<52} {'n':>4} {'full tool/tgt/exact':>21} {'active tool/tgt/exact':>23}"
    print(header)
    print("-" * len(header))
    for payload in payloads:
        s = payload["summary"]
        f, a = s.get("full") or {}, s.get("active") or {}

        def fmt(d):
            return "/".join("--" if d.get(k) is None else f"{d[k]:.3f}"
                            for k in ("tool", "target", "exact"))
        print(f"{Path(payload['source_receipt']).name:<52} {s['examples']:>4} "
              f"{fmt(f):>21} {fmt(a):>23}")
    print()
    for key in ("tool", "target", "exact"):
        pair = paired(payloads, key)
        for row in pair["comparisons"]:
            if row["receipt"].startswith("POOLED"):
                print(f"paired active-full {key}: n={row['n']} mean={row['mean_diff']:+.4f} "
                      f"CI={row['ci95']} W/L={row['wins']}/{row['losses']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--receipt", action="append", default=[],
                   help="a g2b patch-localization receipt to rescore (repeatable)")
    p.add_argument("--out", default="", help="write one action receipt per --receipt")
    p.add_argument("--table", nargs="*", default=None,
                   help="aggregate existing action receipts into one table")
    p.add_argument("--rebuild-groundtruth", action="store_true")
    args = p.parse_args(argv)

    if args.table is not None:
        paths = args.table or sorted(glob.glob("artifacts/g2c-action-*.json"))
        if not paths:
            print("no action receipts found")
            return 1
        return table(paths)

    if not args.receipt:
        p.error("nothing to do: pass --receipt or --table")

    written = []
    for receipt in args.receipt:
        payload = rescore(Path(receipt), rebuild_truth=args.rebuild_groundtruth)
        if args.out and len(args.receipt) == 1:
            out = Path(args.out)
        else:
            stem = Path(receipt).stem.replace("g2b-patch-localization-", "")
            out = Path("artifacts") / f"g2c-action-{stem}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(out)
        s = payload["summary"]
        f, a = s.get("full") or {}, s.get("active") or {}
        print(f"[g2c] {out.name}: n={s['examples']} "
              f"full={f.get('target')} active={a.get('target')} (target match)")
    if written:
        print("wrote", ", ".join(str(w) for w in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
