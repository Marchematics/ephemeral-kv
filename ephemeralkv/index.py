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
                     max_spans: int = 64, recency_spans: int = 0,
                     recency_fraction: float = 0.5,
                     max_span_fraction: float = 1.0,
                     provenance_terms: int = 0) -> tuple[list[Span], LookupStats]:
        """Select the spans that fit `token_budget`.

        The lexical channel alone is not enough for a *continuation* target and the
        failure is easy to describe: a span larger than the budget is skipped outright, so
        one 20K-token tool output can leave the view with no recent tool output at all
        (the gate measured -10.1 pp accuracy at a 4,096-token budget).  Three optional
        channels address that, all off by default so the earlier receipts stay valid:

        * `recency_spans` - always keep the last N spans, within `recency_fraction` of the
          budget (a serving stack never evicts the current turn);
        * `max_span_fraction` - truncate an oversized span to this share of the budget
          instead of dropping it;
        * `provenance_terms` - pull in spans that share identifiers (paths, tool ids)
          with the top lexical hits, on top of the lexical ranking.
        """
        candidates, stats = self.lookup(query, max_spans=max_spans)
        kept: list[Span] = []
        used = 0

        def take(span: Span, room: int) -> None:
            nonlocal used
            if room <= 0 or span.token_estimate <= 0:
                return
            if span.token_estimate <= room:
                kept.append(span)
                used += span.token_estimate
                return
            share = len(span.text) * (room / max(1, span.token_estimate))
            head = span.text[: max(1, int(share))]
            kept.append(Span(span.span_id, span.turn, span.role, head,
                             max(1, int(room))))
            used += max(1, int(room))

        cap = int(token_budget * max_span_fraction) if max_span_fraction < 1.0 else None
        recent_ids: set[int] = set()
        if recency_spans > 0:
            recency_budget = int(token_budget * recency_fraction)
            recent_used = 0
            for span in reversed(self._spans[-recency_spans:]):
                room = min(recency_budget - recent_used,
                           token_budget - used,
                           cap if cap is not None else token_budget)
                if room <= 0:
                    break
                take(span, room)
                recent_used += min(span.token_estimate, room)
                recent_ids.add(span.span_id)

        if provenance_terms > 0:
            # identifiers shared with the query and with the best lexical hits: the cheap
            # version of "the file this turn is about", which lexical overlap alone misses
            seeds = {t for t in terms(query) if len(t) > 3}
            for span in candidates[:3]:
                seeds.update(t for t in terms(span.text)
                             if len(t) > 3 and ("/" in t or "." in t))
            extra: list[Span] = []
            for term in sorted(seeds)[:provenance_terms]:
                for span_id in self._postings.get(term, ())[:8]:
                    span = self._spans[span_id]
                    if span.span_id not in recent_ids and span not in candidates:
                        extra.append(span)
            candidates = candidates + extra[:provenance_terms]

        for span in candidates:
            if span.span_id in recent_ids:
                continue
            room = token_budget - used
            if room <= 0:
                break
            take(span, min(room, cap) if cap is not None else room)

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
