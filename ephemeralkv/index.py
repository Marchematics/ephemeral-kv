"""A tiny model-independent durable span index for EphemeralKV kill gates.

This is intentionally not a QCC implementation. It is a lexical/provenance baseline
whose purpose is to make one systems question measurable: can a cold worker identify
a bounded active text view without scanning an entire long-lived transcript?

The index is append-only and serializable. Query cost is reported as postings visited
so experiments can distinguish real indexed lookup from an O(history) scan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from collections import defaultdict

_TOKEN = re.compile(r"[A-Za-z0-9_./:-]+")


def terms(text: str) -> list[str]:
    return [x.lower() for x in _TOKEN.findall(text)]


@dataclass(frozen=True)
class Span:
    span_id: int
    turn: int
    role: str
    text: str
    token_estimate: int


@dataclass(frozen=True)
class LookupStats:
    query_terms: int
    postings_visited: int
    candidate_spans: int
    total_spans: int


class DurableSpanIndex:
    """Append-only lexical index over transcript spans.

    token_estimate is deliberately supplied by the caller. The index is tokenizer
    agnostic; a serving backend may use a model tokenizer when it materializes the
    selected view.
    """

    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._postings: dict[str, list[int]] = defaultdict(list)

    @property
    def spans(self) -> tuple[Span, ...]:
        return tuple(self._spans)

    def append(self, *, turn: int, role: str, text: str,
               token_estimate: int | None = None) -> Span:
        span_id = len(self._spans)
        if token_estimate is None:
            token_estimate = max(1, len(terms(text)))
        span = Span(span_id, turn, role, text, token_estimate)
        self._spans.append(span)
        for term in sorted(set(terms(text))):
            self._postings[term].append(span_id)
        return span

    def lookup(self, query: str, *, max_spans: int = 32) -> tuple[list[Span], LookupStats]:
        qterms = sorted(set(terms(query)))
        score: dict[int, float] = defaultdict(float)
        visited = 0
        for term in qterms:
            posting = self._postings.get(term, ())
            visited += len(posting)
            weight = 1.0 / max(1, len(posting))
            for span_id in posting:
                score[span_id] += weight

        ranked = sorted(score, key=lambda i: (score[i], self._spans[i].turn, i),
                        reverse=True)[:max_spans]
        selected = [self._spans[i] for i in ranked]
        stats = LookupStats(
            query_terms=len(qterms),
            postings_visited=visited,
            candidate_spans=len(score),
            total_spans=len(self._spans),
        )
        return selected, stats

    def compile_view(self, query: str, *, token_budget: int,
                     max_spans: int = 64) -> tuple[list[Span], LookupStats]:
        candidates, stats = self.lookup(query, max_spans=max_spans)
        kept: list[Span] = []
        used = 0
        for span in candidates:
            if used + span.token_estimate > token_budget:
                continue
            kept.append(span)
            used += span.token_estimate
        kept.sort(key=lambda s: (s.turn, s.span_id))
        return kept, stats

    def dumps(self) -> str:
        payload = {
            "schema": "ephemeral-kv-span-index-v1",
            "spans": [asdict(s) for s in self._spans],
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def loads(cls, raw: str) -> "DurableSpanIndex":
        payload = json.loads(raw)
        if payload.get("schema") != "ephemeral-kv-span-index-v1":
            raise ValueError("unsupported span-index schema")
        out = cls()
        for row in payload["spans"]:
            out.append(
                turn=int(row["turn"]),
                role=str(row["role"]),
                text=str(row["text"]),
                token_estimate=int(row["token_estimate"]),
            )
        return out
