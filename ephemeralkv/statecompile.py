"""Executable-state compilation: replay the event log, keep the materialised state.

The first compiler iteration grouped spans by file and kept the *latest view of that file*.
That is the wrong rule and the measurement said so plainly: -13.90 pp at an 8,192-token budget
against -6.94 pp for plain truncation, because a later view of a file is usually a diff or one
region, not the file.  What a coding session actually carries is an **event log** - full dumps,
unified diffs, search/replace edits, test runs - and what the next turn needs is the
**materialised current state** plus the unresolved constraints, not a history of views.

So this module replays the log:

* a full dump (`cat -n`, `view`) *sets* a file's content;
* a unified diff *applies* to it (hunks matched by context, applied bottom-up);
* an editor block (`<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE`) *replaces* a region;
* a test/build run updates that command's **latest** result, replacing earlier runs of the same
  command rather than accumulating them;
* everything that names no file and is not an execution result (reasoning, acknowledgements)
  keeps only its newest occurrence, because a trajectory repeats itself.

The output is a set of :class:`~ephemeralkv.consolidate.Unit` objects: materialised file state
(verbatim by construction), the current unresolved result (verbatim), and provenance for
everything that was superseded.  Nothing is paraphrased - that is the point.  A summary that
rewrites a traceback or an API signature is useless to the agent that has to act on it, while a
materialised file state is exactly what the agent would have seen had the session been shorter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ephemeralkv.consolidate import (Unit, _normalise_line, content_fingerprint,
                                     must_stay_verbatim, path_of, select_verbatim)

# these are matched against *whole spans*, so their anchors need re.M - without it a block
# that does not start at position 0 never matches (measured: an editor block routed to the
# reasoning branch and its edit was lost)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)
_DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
_EDITOR_FILE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (\S+)", re.M)
_SEARCH = re.compile(r"^<{5,} SEARCH", re.M)
_DIVIDER = re.compile(r"^={5,}", re.M)
_REPLACE = re.compile(r"^>{5,} REPLACE", re.M)
_GUTTER = re.compile(r"^\s*(\d+)\t")
# commands whose *latest* result is what matters, keyed by a normalised signature
_COMMAND = re.compile(r"(?:^|\s)(pytest|python -m pytest|python|npm|make|cargo|go test|"
                      r"git (?:diff|status|log)|ls|cat|grep)\b[^\n]*")


@dataclass
class FileState:
    """A file's materialised content and the turns that built it."""

    path: str
    lines: list[str] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    superseded: list[int] = field(default_factory=list)
    events: int = 0
    verbatim: bool = True

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Result:
    """The latest result of a repeated command (a test run, a build, a status)."""

    signature: str
    text: str
    turn: int
    superseded: list[int] = field(default_factory=list)


def _dump_lines(text: str) -> list[str] | None:
    """Content of a full-file dump, or None if this is not one."""
    lines = text.splitlines()
    numbered = [line for line in lines if _GUTTER.match(line)]
    if len(numbered) >= max(3, len(lines) // 2):
        return [_GUTTER.sub("", line).rstrip() for line in lines if _GUTTER.match(line)]
    return None


def _apply_hunks(base: list[str], text: str) -> tuple[list[str], int]:
    """Apply unified-diff hunks to `base`, newest context wins on mismatch."""
    out = list(base)
    applied = 0
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = _HUNK.match(lines[index])
        if not match:
            index += 1
            continue
        old_start = max(0, int(match.group(1)) - 1)
        index += 1
        removed, added = [], []
        while index < len(lines) and not _HUNK.match(lines[index]) and \
                not lines[index].startswith("diff --git"):
            line = lines[index]
            if line.startswith("-"):
                removed.append(line[1:])
            elif line.startswith("+"):
                added.append(line[1:])
            elif line.startswith(" "):
                removed.append(line[1:])
                added.append(line[1:])
            index += 1
        # locate the removed block by content near the declared position, then replace it
        target = None
        probe = [line for line in removed if line.strip()]
        for candidate in range(max(0, old_start - 5), min(len(out), old_start + 5) + 1):
            window = out[candidate:candidate + len(removed)]
            if window and probe and window[:len(probe)] == probe[:len(probe)]:
                target = candidate
                break
        if target is None:
            target = min(old_start, len(out))
        out[target:target + len(removed)] = added
        applied += 1
    return out, applied


def _apply_editor(base: list[str], text: str) -> tuple[list[str], int]:
    """Apply `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks to `base`."""
    out = list(base)
    applied = 0
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not _SEARCH.match(lines[index]):
            index += 1
            continue
        index += 1
        search, replace, mode = [], [], "search"
        while index < len(lines) and not _REPLACE.match(lines[index]):
            line = lines[index]
            if _DIVIDER.match(line):
                mode = "replace"
            elif mode == "search":
                search.append(line)
            else:
                replace.append(line)
            index += 1
        index += 1
        if not search:
            continue
        found = None
        for candidate in range(len(out) - len(search) + 1):
            if out[candidate:candidate + len(search)] == search:
                found = candidate
                break
        if found is not None:
            out[found:found + len(search)] = replace
            applied += 1
    return out, applied


def _signature(text: str) -> str | None:
    match = _COMMAND.search(" ".join(text.split())[:400])
    return match.group(0).strip() if match else None


def materialise(spans) -> tuple[dict[str, FileState], dict[str, Result], list[Unit]]:
    """Replay `spans` in turn order into file states, latest results and loose evidence."""
    files: dict[str, FileState] = {}
    results: dict[str, Result] = {}
    loose: list[Unit] = []

    for span in sorted(spans, key=lambda s: (s.turn, s.span_id)):
        text = span.text
        path = path_of(text)
        editor = _EDITOR_FILE.search(text)
        if editor and not path:
            path = editor.group(1)
        dump = _dump_lines(text) if path else None

        if path and dump is not None:
            state = files.setdefault(path, FileState(path=path))
            if state.lines:
                state.superseded.append(state.turns[-1] if state.turns else span.turn)
            state.lines = list(dump)
            state.turns.append(span.turn)
            state.events += 1
            continue

        if path and (text.lstrip().startswith("diff --git") or "@@ " in text):
            state = files.setdefault(path, FileState(path=path))
            state.lines, applied = _apply_hunks(state.lines, text)
            if applied:
                state.turns.append(span.turn)
                state.events += applied
                continue

        if path and _SEARCH.search(text):
            state = files.setdefault(path, FileState(path=path))
            state.lines, applied = _apply_editor(state.lines, text)
            if applied:
                state.turns.append(span.turn)
                state.events += applied
                continue

        # a command's output often names files ("1 failed in tests/test_cache.py"), so this
        # branch cannot be gated on "no path" - it is reached only after the branches that
        # actually consume file state have declined
        signature = _signature(text)
        if signature is not None:
            key = " ".join(signature.split()[:3])
            previous = results.get(key)
            superseded = ([previous.turn] if previous else []) + \
                (previous.superseded if previous else [])
            results[key] = Result(signature=key, text=text, turn=span.turn,
                                  superseded=superseded)
            continue

        # no file and no command: reasoning, acknowledgements, questions.  Keep the newest
        # occurrence of each distinct text and drop earlier repeats.
        fingerprint = content_fingerprint(text)
        for unit in loose:
            if content_fingerprint(unit.text) == fingerprint:
                unit.turns.append(span.turn)
                unit.superseded.append(span.turn)
                break
        else:
            loose.append(Unit(text=text, tokens=span.token_estimate, verbatim=True,
                              turns=[span.turn], kind="span"))

    return files, results, loose


def compile_executable_state(spans, token_budget: int, tokens_of,
                             query_terms: set[str] | None = None,
                             recent_turns: int = 2) -> tuple[list[Unit], dict]:
    """Materialise the log, then fit the *state* into the budget.

    Priority: the newest files the session is working on (a file whose last event is recent, or
    whose path appears in the query), then unresolved results, then recent conversation.  A file
    state that does not fit is cut with :func:`select_verbatim`, which drops prose before code
    and marks the omission.
    """
    files, results, loose = materialise(spans)
    query_terms = query_terms or set()
    newest_turn = max((max(f.turns) for f in files.values() if f.turns), default=0)
    newest_turn = max(newest_turn, max((r.turn for r in results.values()), default=0))
    newest_turn = max(newest_turn, max((max(u.turns) for u in loose if u.turns), default=0))

    def file_rank(state: FileState) -> tuple:
        in_query = any(term and term in state.path for term in query_terms)
        recent = bool(state.turns) and state.turns[-1] >= newest_turn - recent_turns
        return (0 if in_query else 1, 0 if recent else 1, -max(state.turns or [0]))

    ordered: list[Unit] = []
    for state in sorted(files.values(), key=file_rank):
        ordered.append(Unit(text=f"# {state.path} (current state)\n{state.text}",
                            tokens=max(1, state.events), verbatim=True,
                            turns=list(state.turns), superseded=list(state.superseded),
                            kind="state"))
    for result in sorted(results.values(), key=lambda r: -r.turn):
        ordered.append(Unit(text=f"# latest `{result.signature}` result\n{result.text}",
                            tokens=max(1, len(result.text.split())), verbatim=True,
                            turns=[result.turn], superseded=list(result.superseded),
                            kind="result"))
    for unit in sorted(loose, key=lambda u: -max(u.turns or [0])):
        ordered.append(unit)

    out, stats = _fit(ordered, token_budget, tokens_of)
    stats.update({"files": len(files), "results": len(results), "loose": len(loose),
                  "materialised_events": sum(f.events for f in files.values()),
                  "superseded_events": sum(len(f.superseded) for f in files.values())})
    return out, stats


def _fit(units: list[Unit], token_budget: int, tokens_of) -> tuple[list[Unit], dict]:
    """Fit units into the budget, cutting prose before code inside any unit that overflows."""
    out: list[Unit] = []
    used = 0
    stats = {"units_in": len(units), "units_out": 0, "compiled": 0, "tokens_used": 0}
    for unit in units:
        if used >= token_budget:
            break
        room = token_budget - used
        cost = max(1, int(tokens_of(unit.text) if callable(tokens_of) else unit.tokens))
        if cost > room:
            text, verbatim = select_verbatim(unit.text, room, tokens_of)
            if not text.strip():
                continue
            cost = max(1, int(tokens_of(text) if callable(tokens_of) else room))
            cost = min(cost, max(1, room))
            stats["compiled"] += 1
            out.append(Unit(text=text, tokens=cost, verbatim=verbatim, turns=list(unit.turns),
                            kind=unit.kind, superseded=list(unit.superseded)))
        else:
            out.append(unit)
        used += cost
    stats["units_out"] = len(out)
    stats["tokens_used"] = used
    stats["budget"] = token_budget
    return out, stats

# --------------------------------------------------------------------------- #
# state-first selection
# --------------------------------------------------------------------------- #

def classify(text: str) -> str:
    """What kind of evidence a span is: file state, a command result, or loose text.

    Measured over three long examples: 74% of the spans a lexical retriever returns are loose
    text (reasoning, acknowledgements, chatter), 14% are command results and only 6% are file
    events - which is why a state compiler that only had the lexical top-k to work with behaved
    exactly like plain truncation.  Selection has to be state-first, and that needs the span
    kinds at index time.
    """
    if path_of(text) and (_dump_lines(text) is not None or _SEARCH.search(text)
                          or text.lstrip().startswith("diff --git") or "@@ " in text):
        return "file_event"
    if _signature(text) is not None:
        return "command_result"
    return "loose"


def state_first_units(index, query: str, token_budget: int, tokens_of,
                      loose_spans: int = 12, recent_spans: int = 4) -> tuple[list[Unit], dict]:
    """Build the view from state-carrying evidence, not from lexical top-k.

    Order of preference: materialised file states (all of them - there are few, and each one
    stands for every event that produced it), then the latest result of each command, then the
    newest loose text, then loose text that overlaps the query.  The point is that state
    evidence competes for the budget on its *kind*, not on how many query terms happen to be in
    it - which is what let chatter crowd out file state before.
    """
    from ephemeralkv.index import terms as _terms

    query_terms = set(_terms(query))
    spans = list(index.spans)
    file_spans = [span for span in spans if classify(span.text) == "file_event"]
    result_spans = [span for span in spans if classify(span.text) == "command_result"]
    loose = [span for span in spans if classify(span.text) == "loose"]

    # results must go through the replayer too, otherwise the "latest result of each command"
    # channel is computed and then dropped (measured: the file state appeared in the view and
    # the test result did not)
    files, results, _ = materialise(file_spans + result_spans)
    newest_turn = max((max(f.turns) for f in files.values() if f.turns),
                      default=0)
    query_paths = {term for term in query_terms if "." in term}

    units: list[Unit] = []
    for state in sorted(files.values(),
                        key=lambda f: (0 if any(p in f.path for p in query_paths) else 1,
                                       -max(f.turns or [0]))):
        units.append(Unit(text=f"# {state.path} (current state)\n{state.text}",
                          tokens=max(1, len(state.text.split())), verbatim=True,
                          turns=list(state.turns), superseded=list(state.superseded),
                          kind="state"))
    for result in sorted(results.values(), key=lambda r: -r.turn):
        units.append(Unit(text=f"# latest `{result.signature}` result\n{result.text}",
                          tokens=max(1, len(result.text.split())), verbatim=True,
                          turns=[result.turn], superseded=list(result.superseded),
                          kind="result"))

    def overlap(span) -> float:
        tokens = set(_terms(span.text))
        return len(tokens & query_terms) / max(1, len(query_terms))

    loose_sorted = sorted(loose, key=lambda s: (-overlap(s), -s.turn))
    for span in loose_sorted[:loose_spans]:
        units.append(Unit(text=span.text, tokens=span.token_estimate, verbatim=True,
                          turns=[span.turn], kind="span"))
    for span in sorted(loose, key=lambda s: -s.turn)[:recent_spans]:
        if span.turn not in {u.turns[0] for u in units if u.kind == "span"}:
            units.append(Unit(text=span.text, tokens=span.token_estimate, verbatim=True,
                              turns=[span.turn], kind="recent"))

    out, stats = _fit(units, token_budget, tokens_of)
    stats.update({"kind_file_spans": len(file_spans), "kind_results": len(result_spans),
                  "kind_loose": len(loose), "files": len(files),
                  "query_paths": sorted(query_paths)[:4]})
    return out, stats
