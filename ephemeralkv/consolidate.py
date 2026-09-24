"""Stage 2 and 3 of the compiler: consolidate evidence, then keep what must be verbatim.

The G2 receipts say where the remaining gap is.  Retrieval is not the problem - the durable
index finds the right spans (structural receipt: a median 12% of a 32K-128K history) - and
re-ranking the same text does not help either (snippet selection was within noise of plain
truncation, -6.70 against -6.94 pp at 8,192 tokens).  What *did* halve the budget was
removing duplicated content.  This module generalises that observation from "identical text"
to "the same *state* seen repeatedly", which is what a coding trajectory is full of:

    turn 12  cache.py uses an LRU list
    turn 19  cache.py now uses a dict of deques
    turn 27  cache.py's eviction still fails the test
    turn 35  cache.py switched to an OrderedDict

A retrieval-only compiler keeps all four copies and spends four spans' worth of budget on one
file.  The consolidator keeps the **latest** state, records the provenance of the ones it
replaced, and spends the freed budget on other evidence.  It is deliberately *extractive* -
no summarisation model - because the third stage exists to protect exactly the parts that a
summary would paraphrase away: code, error messages, paths, identifiers and numbers.

Two stages, and the order matters:

* :func:`consolidate` - group by the state they describe (file path, tool signature), collapse
  superseded versions and near-duplicate text, and emit provenance lines;
* :func:`select_verbatim` - when a consolidated unit still has to be cut to fit, keep every
  line that must survive byte-for-byte and cut prose instead.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

_PATH = re.compile(r"[\w./-]+\.(?:py|pyx|js|ts|tsx|jsx|md|txt|json|ya?ml|toml|cfg|ini|c|cc|"
                   r"cpp|h|hpp|rs|go|java|rb|sh|sql|html|css)")
# tool outputs that dump a file's *state* rather than a transient message
_DUMP = re.compile(r"^\s*\d+\t|^\s*\d+\s{2,}\S|\*\*\* (?:Update|Add|Delete) File:|"
                   r"^diff --git a/|^@@ ")
_ERROR = re.compile(r"Traceback \(most recent call last\)|^[A-Za-z_.]*(?:Error|Exception)\b|"
                    r"\b(?:FAILED|AssertionError)\b|^\s*assert\b|error:")
_CODE = re.compile(r"^\s*(?:def |class |import |from |return |if |for |while |try:|except |"
                   r"elif |else:|elif\b|@|\}|\{|\)|#)|[=;{}()\[\]]\s*$|^\s*[+-]\s|"
                   r"^\s{4,}\S")
_NUMBER_OR_ID = re.compile(r"\b\d[\d_.]*\b|[0-9a-f]{8}-[0-9a-f]{3,}")


def _normalise_line(line: str) -> str:
    """Strip the decorations a tool adds so two views of one file compare equal."""
    line = re.sub(r"^\s*\d+\t", "", line)              # `cat -n` gutter
    line = re.sub(r"^\s*\d+\s{2,}", "", line)          # `view` gutter
    return line.rstrip()


def content_fingerprint(text: str) -> str:
    """Identity of the *content*, ignoring line numbers and whitespace runs."""
    body = " ".join(" ".join(_normalise_line(line).split())
                    for line in text.splitlines())
    return hashlib.md5(body.encode()).hexdigest()


def _strip_diff_prefix(path: str) -> str:
    """`a/pkg/mod.py` and `b/pkg/mod.py` are the same file as `pkg/mod.py`."""
    for prefix in ("a/", "b/"):
        if path.startswith(prefix) and path.count("/") > 1:
            return path[len(prefix):]
    return path


def path_of(text: str) -> str | None:
    """The file a span is about, if it names exactly one - otherwise None."""
    hits = {_strip_diff_prefix(m.group(0)) for m in _PATH.finditer(text)}
    return next(iter(hits)) if len(hits) == 1 else None


def must_stay_verbatim(line: str) -> bool:
    """Whether cutting or paraphrasing this line would destroy an executable detail.

    Code, tracebacks, diffs, paths and numbers are kept; ordinary prose is what gets cut when
    the budget binds.  This is the whole reason the compiler is extractive rather than
    abstractive: a summary that rewrites `dict[str, deque]` or a traceback is useless to the
    agent that has to act on it.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _DUMP.match(line) or _ERROR.search(line) or _CODE.match(line):
        return True
    if _PATH.search(line) or _NUMBER_OR_ID.search(line):
        return True
    return False


# --------------------------------------------------------------------------- #
# consolidation
# --------------------------------------------------------------------------- #

@dataclass
class Unit:
    """One compiled unit: text, whether it is verbatim, and where it came from."""

    text: str
    tokens: int
    verbatim: bool
    turns: list[int] = field(default_factory=list)
    kind: str = "span"                     # span | state | provenance | recent
    superseded: list[int] = field(default_factory=list)

    @property
    def provenance(self) -> str:
        if not self.turns:
            return ""
        turns = ",".join(str(t) for t in sorted(set(self.turns)))
        line = f"[turns {turns}]"
        if self.superseded:
            gone = ",".join(str(t) for t in sorted(set(self.superseded)))
            line += f" [superseded: {gone}]"
        return line


def _similar(a: str, b: str, threshold: float = 0.7) -> bool:
    """Cheap near-duplicate check: Jaccard over line shingles, with a size guard."""
    lines_a = {_normalise_line(x).strip() for x in a.splitlines() if x.strip()}
    lines_b = {_normalise_line(x).strip() for x in b.splitlines() if x.strip()}
    if not lines_a or not lines_b:
        return False
    if min(len(lines_a), len(lines_b)) / max(len(lines_a), len(lines_b)) < threshold:
        return False
    inter = len(lines_a & lines_b)
    union = len(lines_a | lines_b)
    return union > 0 and inter / union >= threshold


def consolidate(spans, *, query_terms: set[str] | None = None,
                collapse_paths: bool = True, keep_earlier_verbatim: bool = False) -> list[Unit]:
    """Collapse superseded and repeated evidence into one unit per state.

    Grouping is by the file a span is about (when it names exactly one) and otherwise by
    content identity.  Within a group the most recent span is the state; earlier ones become
    provenance.  Near-duplicates across groups are merged the same way.

    The returned units carry their turns, so a downstream consumer can always tell which
    turns a compiled statement came from - the property that separates this from
    summarisation.
    """
    units: list[Unit] = []
    by_path: dict[str, list] = {}
    loose: list = []
    for span in spans:
        path = path_of(span.text)
        (by_path.setdefault(path, []) if path else loose).append(span)

    for path, group in by_path.items():
        group = sorted(group, key=lambda s: (s.turn, s.span_id))
        keep = group[-1]
        superseded = [s.turn for s in group[:-1]]
        merged = keep.text
        if not collapse_paths:
            # ablation: keep every view of the file as its own unit.  Measured effect of
            # collapsing to the latest state at an 8,192-token budget: -13.90 pp against
            # -6.94 pp for plain truncation, i.e. the earlier views carry content the next
            # turn needs and "latest state wins" throws it away.
            for span in group:
                units.append(Unit(text=span.text, tokens=span.token_estimate,
                                  verbatim=True, turns=[span.turn], kind="state"))
            continue
        if keep_earlier_verbatim:
            # union of evidence: the latest state, plus every verbatim-critical line from the
            # views it replaced, so nothing executable is lost by collapsing
            seen = {_normalise_line(line).strip() for line in merged.splitlines()}
            extra = []
            for span_item in group[:-1]:
                for line in span_item.text.splitlines():
                    key = _normalise_line(line).strip()
                    if key and key not in seen and must_stay_verbatim(line):
                        seen.add(key)
                        extra.append(line)
            if extra:
                merged = merged + "\n" + "\n".join(extra)
        # a diff or a later dump is state; an earlier identical body adds nothing
        units.append(Unit(text=merged, tokens=keep.token_estimate, verbatim=True,
                          turns=[s.turn for s in group], superseded=superseded,
                          kind="state"))

    for span in sorted(loose, key=lambda s: (s.turn, s.span_id)):
        merged_into = None
        for unit in units:
            if content_fingerprint(unit.text) == content_fingerprint(span.text) or \
                    _similar(unit.text, span.text):
                merged_into = unit
                break
        if merged_into is not None:
            merged_into.turns.append(span.turn)
            merged_into.superseded.append(span.turn)
            continue
        units.append(Unit(text=span.text, tokens=span.token_estimate,
                          verbatim=True, turns=[span.turn], kind="span"))

    units.sort(key=lambda u: (min(u.turns) if u.turns else 0))
    return units


# --------------------------------------------------------------------------- #
# stage 3: selective verbatim retention
# --------------------------------------------------------------------------- #

def select_verbatim(text: str, keep_tokens: int, tokens_of) -> tuple[str, bool]:
    """Cut `text` to `keep_tokens`, dropping prose before anything executable.

    Returns the text and whether everything kept is verbatim-critical.  Lines are scored
    `must_stay_verbatim` first; when the verbatim lines alone exceed the allowance the cut
    falls back to keeping them in order (never a paraphrase), and when they fit, prose lines
    are added back in order until the allowance is spent.
    """
    lines = text.splitlines()
    if not lines:
        return text, True
    costs = [max(1, int(tokens_of(line) if callable(tokens_of) else 1)) for line in lines]
    total = sum(costs)
    if total <= keep_tokens:
        return text, all(must_stay_verbatim(line) for line in lines if line.strip())

    verbatim = [index for index, line in enumerate(lines) if must_stay_verbatim(line)]
    chosen: list[int] = []
    used = 0
    for index in verbatim:
        if used + costs[index] > keep_tokens:
            continue
        chosen.append(index)
        used += costs[index]
    for index, line in enumerate(lines):
        if index in chosen or not line.strip():
            continue
        if used + costs[index] > keep_tokens:
            break
        chosen.append(index)
        used += costs[index]
    chosen.sort()
    if not chosen:                                  # nothing fit: keep the head verbatim
        chosen = list(range(min(len(lines), max(1, keep_tokens))))
    out, previous = [], None
    for index in chosen:
        if previous is not None and index > previous + 1:
            out.append("... [compiled: unrelated lines omitted] ...")
        out.append(lines[index])
        previous = index
    if chosen[-1] < len(lines) - 1:
        # mark a cut tail as well: text that simply stops looks complete, which is the one
        # thing a compiled view must never imply
        out.append(f"... [compiled: {len(lines) - 1 - chosen[-1]} trailing lines omitted] ...")
    kept_prose = any(not must_stay_verbatim(lines[i]) for i in chosen)
    return "\n".join(out), not kept_prose


def compile_units(units: list[Unit], token_budget: int, tokens_of,
                  recency_turns: int = 1) -> tuple[list[Unit], dict]:
    """Fit consolidated units into `token_budget`, newest state first.

    Priority is the point of the whole pipeline: the most recent turn's evidence, then the
    consolidated file states (which already stand for every earlier view of the same file),
    then older material - and within a unit, prose is what gets cut.
    """
    ordered = sorted(units, key=lambda u: (-max(u.turns or [0]), u.kind != "state"))
    recent_cutoff = None
    if recency_turns and units:
        newest = max(max(u.turns) for u in units if u.turns)
        recent_cutoff = newest - recency_turns + 1
    out: list[Unit] = []
    used = 0
    stats = {"units_in": len(units), "units_out": 0, "compiled": 0, "superseded": 0}
    for unit in ordered:
        if used >= token_budget:
            break
        room = token_budget - used
        text = unit.text
        verbatim = unit.verbatim
        if unit.tokens > room:
            text, verbatim = select_verbatim(text, room, tokens_of)
            if text is None or not text.strip():
                continue
            stats["compiled"] += 1
            cost = min(room, max(1, int(tokens_of(text) if callable(tokens_of)
                                        else unit.tokens)))
        else:
            cost = unit.tokens
        keep = Unit(text=text, tokens=cost, verbatim=verbatim, turns=list(unit.turns),
                    kind=unit.kind, superseded=list(unit.superseded))
        out.append(keep)
        used += cost
        stats["superseded"] += len(set(unit.superseded))
    out.sort(key=lambda u: (min(u.turns) if u.turns else 0))
    stats["units_out"] = len(out)
    stats["tokens_used"] = used
    stats["budget"] = token_budget
    return out, stats


def render(units: list[Unit], with_provenance: bool = True) -> str:
    """The compiled context: each unit, followed by where it came from."""
    parts = []
    for unit in units:
        parts.append(unit.text)
        if with_provenance and unit.provenance:
            parts.append(unit.provenance)
    return "\n".join(parts)
