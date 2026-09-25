#!/usr/bin/env python
"""G4: does a history-free mobility cost change the best routing decision?

The G3 receipts measure what a cold route costs.  G4 asks the systems question that
follows: at 4--8 data-parallel workers, does replacing a history-sized cold-route penalty
with an active-set-sized one change *when a scheduler should leave the warm worker*?

The scheduler is deliberately trivial, because the routing heuristic is not the claim:

    choose worker i minimizing   queue_delay(i) + (0 if warm(i) else tax(request, i))

What changes between policies is only `tax`:

* `strict_sticky`      - never leave the warm worker (tax = infinity for cold workers);
* `full_kv_move`       - pay the measured H2D cost of the history's KV payload;
* `full_reprefill`     - pay the measured full-history prefill cost;
* `ephemeral`          - pay lookup + prefill of the *active* set (G3-measured).

Every cost constant is read from the G3 artifacts rather than declared, so a policy
comparison cannot quietly assume the answer.  What is *declared* is the workload: the
public corpus carries no timestamps, so turn arrivals and tool gaps come from an explicit
model (`--gap-model`), and that is labelled in the output.

Usage:
    python benchmarks/g4_routing_replay.py \
        --g3 artifacts/g3-hardware-primitives-qwen05b-v1.json \
        --g3-warm artifacts/g3-active-prefill-warm-qwen05b-v1.json \
        --g3-prefill artifacts/g3-fullprefill-qwen05b-v1.json \
        --out artifacts/g4-routing-replay-v1.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# histories to replay, and the active set each turn needs.  The 16K active set is what
# G2 measured as fidelity-preserving on long histories; 2K is the small-working-set
# version the mobility thesis would prefer.
HISTORY_CHOICES = (32768, 131072, 262144)
# A million-token session is the regime the durability argument is about: the KV payload to move
# grows with the *session* (12.00 GiB at 1M on this model, 555.6 ms on the measured local link)
# while rematerialisation grows with the *active set* (0.053 s at 2,048), so where the crossover
# sits has to be located rather than assumed.  The lookup row for a 1M history is postings-growth
# extrapolation - no trace in the public corpus is that long - and the receipt says so.
HISTORY_CHOICES_LONG = (32768, 131072, 262144, 1048576)
SLO_SECONDS = 2.0


@dataclass
class Costs:
    """Measured primitives, in seconds, read from G3 receipts."""

    kv_bytes_per_token: int
    bandwidth_gbs: float
    lookup_s: dict[int, float]              # by history bucket
    active_prefill_s: dict[int, float]      # by active tokens
    full_prefill_s: dict[int, float | None]  # by history tokens, None = infeasible

    def move_s(self, history: int) -> float | None:
        payload = history * self.kv_bytes_per_token
        if self.bandwidth_gbs <= 0:
            return None
        return payload / (self.bandwidth_gbs * 1e9)

    def full_prefill(self, history: int) -> float | None:
        best = None
        for tokens, seconds in sorted(self.full_prefill_s.items()):
            if seconds is None:
                continue
            if tokens <= history:
                best = seconds * (history / tokens)      # superlinear, so this is a floor
        return best

    def ephemeral(self, history: int, active: int) -> float:
        bucket = next((b for b in sorted(self.lookup_s) if history <= b), max(self.lookup_s))
        prefill = self.active_prefill_s.get(active)
        if prefill is None:
            prefill = self.active_prefill_s[max(self.active_prefill_s)] * active / max(
                self.active_prefill_s)
        return self.lookup_s[bucket] + prefill


def load_costs(g3_path: Path, warm_path: Path, prefill_path: Path) -> Costs:
    g3 = json.loads(Path(g3_path).read_text())
    warm = json.loads(Path(warm_path).read_text())
    prefill = json.loads(Path(prefill_path).read_text())
    kv = int(g3["config"]["kv_bytes_per_token"])
    bandwidth = statistics.median(
        row["full_kv_h2d"]["effective_GBps_p50"] for row in g3["history_rows"])
    # the structural receipt carries lookup by history bucket; fall back to G3's own
    # numbers if it is not supplied with the artifact set
    structural = None
    for candidate in (g3_path.parent / "g2-thoughtworks-structural-v1.json",):
        if candidate.exists():
            structural = json.loads(candidate.read_text())
    lookup = {8192: 0.000087, 32768: 0.000150, 131072: 0.000246, 1048576: 0.000400}
    if structural:
        buckets = structural["summary"]["by_history_bucket"]
        mapping = {"<=8K": 8192, "8K-32K": 32768, "32K-128K": 131072}
        for name, tokens in mapping.items():
            if name in buckets:
                lookup[tokens] = buckets[name]["lookup_ms_p50"] / 1000.0
    active = {int(row["tokens"]): row["times"]["p50_s"]
              for row in warm["active_prefill_rows"]}
    full = {}
    for row in prefill["history_rows"]:
        status = row["full_reprefill"].get("status")
        full[int(row["history_tokens"])] = (
            row["full_reprefill"]["times"]["p50_s"] if status == "measured" else None)
    return Costs(kv_bytes_per_token=kv, bandwidth_gbs=bandwidth, lookup_s=lookup,
                 active_prefill_s=active, full_prefill_s=full)


@dataclass
class Turn:
    session: int
    arrival: float
    history: int
    active: int


@dataclass
class Worker:
    """A worker with bounded warm (HBM) residency and an optional KV tier.

    Capacity is what makes the cold-route cost law matter once the fabric is fast: a
    full-KV mover needs the session's KV to still exist in the tier to move it, and when
    the tier has evicted it the cold route degrades to a full re-prefill (or to nothing,
    if that length is infeasible).  An ephemeral route does not care - it rematerializes
    the active set from the durable transcript.  The transcript and index are global and
    are not modelled per worker.
    """

    index: int
    speed: float = 1.0                       # 1.0 = nominal, <1 = slow worker
    free_at: float = 0.0
    warm: dict[int, None] = field(default_factory=dict)      # LRU, bounded by capacity
    served: int = 0
    migrated: int = 0

    @property
    def resident(self) -> dict[int, None]:
        return self.warm

    def touch(self, session: int, warm_capacity: int) -> None:
        self.warm.pop(session, None)
        self.warm[session] = None
        while len(self.warm) > warm_capacity:
            evicted, _ = next(iter(self.warm.items()))
            del self.warm[evicted]


class KvTier:
    """A cluster-level KV store with bounded capacity.

    The tier is *shared*, so a move is possible to any worker while the session's KV is
    resident in it - the per-worker version of this made every migration a recompute,
    which is not the system being compared against.  Capacity is the knob that matters:
    a KV store that cannot hold every long-lived session's history turns a full-KV cold
    route into a recompute, and past the measured lengths a recompute is infeasible.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.entries: dict[int, None] = {}

    def holds(self, session: int) -> bool:
        return session in self.entries

    def store(self, session: int) -> None:
        if self.capacity <= 0:
            return
        self.entries.pop(session, None)
        self.entries[session] = None
        while len(self.entries) > self.capacity:
            oldest, _ = next(iter(self.entries.items()))
            del self.entries[oldest]

    def drop(self, session: int) -> None:
        self.entries.pop(session, None)


def build_turns(sessions: int, turns_per_session: int, *, seed: int, gap_model: str,
                active_tokens: int,
                history_choices: tuple[int, ...] = HISTORY_CHOICES,
                history_skew: float = 0.0,
                burst: dict | None = None) -> list[Turn]:
    """A declared arrival/gap model: the public corpus has no timestamps.

    `bursty` alternates a short tool gap (the agent is iterating) with a long think gap,
    which is where sticky routing looks best and where mobility is most tempting.

    Two regime knobs change the *workload* rather than the routing policy:

    * `history_skew > 0` draws session sizes from a Zipf-like distribution over the sorted
      choices instead of uniformly.  Real fleets are not uniform: most coding sessions are
      short and a few run for hours, so the KV payload a mover has to shift is dominated by a
      handful of sessions - which is exactly the regime a law of the form `cost ~ history`
      is worst at and a state that does not grow with age is best at.
    * `burst` compresses a share of the sessions into a short arrival window.  A flash crowd
      evicts warm state on every worker at once, so cold routes happen under contention
      rather than one at a time, which is the regime where a cheap cold route is worth most.
    """
    rng = random.Random(seed)
    ordered = sorted(history_choices)
    weights = [1.0 / ((i + 1) ** history_skew) for i in range(len(ordered))] \
        if history_skew > 0 else None
    burst = burst or {}
    burst_at = float(burst.get("at", 0.0))
    burst_window = float(burst.get("window", 0.0))
    burst_share = float(burst.get("share", 0.0))
    burst_sessions = int(round(sessions * burst_share))
    turns: list[Turn] = []
    for session in range(sessions):
        if weights is None:
            target = rng.choice(history_choices)
        else:
            target = rng.choices(ordered, weights=weights, k=1)[0]
        # a session's history grows with its own turns: turn 0 is a short prompt, not the
        # whole eventual transcript.  Charging every first turn a full-history recompute
        # made placement cost dominate every policy that can migrate (measured: p50 57 s
        # for a KV mover against 1.0 s ephemeral, which is an artefact of the workload
        # model, not of the cost law).
        history = 2048
        time = rng.random() * 2.0
        in_burst = session < burst_sessions
        if in_burst:
            time = burst_at + rng.random() * max(burst_window, 1e-6)
        for turn in range(turns_per_session):
            turns.append(Turn(session, time, history, active_tokens))
            if in_burst:
                # a flash crowd arrives together but still iterates at the normal cadence
                gap = rng.expovariate(1 / 2.0)
            elif gap_model == "bursty":
                gap = rng.expovariate(1 / 0.4) if turn % 2 == 0 else rng.expovariate(1 / 6.0)
            else:
                gap = rng.expovariate(1 / 2.0)
            time += gap
            history = min(target, int(history * 1.7) + 512)
        turns.sort(key=lambda t: t.arrival)
    return sorted(turns, key=lambda t: t.arrival)


def tax_of(policy: str, costs: Costs, turn: Turn) -> float | None:
    """The cold-route penalty a policy pays for this turn.

    `strict_sticky` is a full-KV system that additionally refuses to migrate: its cold
    route is the same full re-prefill, but it is only ever taken when the session is not
    resident anywhere (placement), because a session that *is* resident elsewhere is
    served by that worker and by no other.
    """
    if policy == "ephemeral":
        return costs.ephemeral(turn.history, turn.active)
    if policy in ("full_kv_move", "strict_sticky"):
        return costs.full_prefill(turn.history) if policy == "strict_sticky" \
            else costs.move_s(turn.history)
    if policy == "full_reprefill":
        return costs.full_prefill(turn.history)
    if policy == "sticky_saturated":
        # the escape hatch is a *KV-aware* migration: the tier holds the history, so the
        # cold route is a move rather than a recompute (paying a full re-prefill here
        # made this baseline worse than plain sticky, which is not the baseline the gate
        # is supposed to beat)
        return costs.move_s(turn.history)
    raise ValueError(policy)


def route(turns: list[Turn], costs: Costs, *, workers: int, policy: str,
          turn_service_s: float = 0.6, escape_s: float | None = None,
          warm_capacity: int = 4, tier_capacity: int = 0,
          slow_worker: int | None = None, slow_worker_speed: float = 0.35,
          fail_at: float | None = None,
          fail_worker: int | None = None) -> dict:
    fleet = [Worker(i, speed=slow_worker_speed if i == slow_worker else 1.0)
             for i in range(workers)]
    tier = KvTier(tier_capacity)
    finished: list[tuple[float, float]] = []          # (completion, latency)
    unserved = 0
    for turn in turns:
        if fail_at is not None and turn.arrival >= fail_at and fail_worker is not None:
            # the worker is *gone*: its warm state goes with it and the remaining fleet
            # must absorb its sessions.  A failure that only clears warm state while the
            # worker keeps serving (the first version of this) barely changes anything -
            # sessions stay warm, so no cold-route cost law is ever exercised, which is
            # why every policy scored within a few percent of every other.
            if fail_worker in [w.index for w in fleet]:
                worker = fleet[fail_worker]
                worker.warm.clear()
                fleet = [w for w in fleet if w.index != fail_worker]
            continue

        def cold_tax(worker: Worker) -> float | None:
            """The tax of serving this turn on `worker` from cold (cluster-wide tier)."""
            if policy == "ephemeral":
                return costs.ephemeral(turn.history, turn.active)
            if policy == "full_reprefill":
                return costs.full_prefill(turn.history)
            # full-KV movers can fetch the history while the *cluster* tier holds it
            if tier.holds(turn.session):
                return costs.move_s(turn.history)
            # otherwise the tier has evicted it: the cold route recomputes the whole
            # history, which is infeasible past the lengths G3 could prefill at all
            return costs.full_prefill(turn.history)
        best, best_cost = None, None
        # sticky-until-saturated: stay on the warm worker until its *queue* is worse than
        # paying the cold route, then migrate like everyone else.  This is the strongest
        # sticky baseline (production systems ship it) and the one the gate must beat.
        if policy == "sticky_saturated":
            warm_worker = next((w for w in fleet if turn.session in w.resident), None)
            if warm_worker is not None:
                wait = max(warm_worker.free_at, turn.arrival) - turn.arrival
                cold = tax_of(policy, costs, turn)
                threshold = escape_s if escape_s is not None else 2.0
                if wait <= threshold:
                    start = max(warm_worker.free_at, turn.arrival)
                    completion = start + turn_service_s / warm_worker.speed
                    warm_worker.free_at = completion
                    warm_worker.served += 1
                    finished.append((completion, completion - turn.arrival))
                    continue
        for worker in fleet:
            warm = turn.session in worker.warm
            if policy == "strict_sticky" and not warm and any(
                    turn.session in w.warm for w in fleet):
                continue                              # only the warm worker may serve it
            if policy == "strict_sticky" and warm:
                best = worker
                best_cost = max(worker.free_at, turn.arrival) - turn.arrival
                break                                 # the warm worker is always chosen
            start = max(worker.free_at, turn.arrival)
            wait = start - turn.arrival
            tax = 0.0 if warm else cold_tax(worker)
            if tax is None:
                continue                              # infeasible route at this length
            total = wait + (tax + turn_service_s) / worker.speed
            if best_cost is None or total < best_cost:
                best, best_cost = worker, total
        if best is None:
            unserved += 1
            continue                                  # no worker can serve this turn
        start = max(best.free_at, turn.arrival)
        tax = 0.0 if turn.session in best.warm else (cold_tax(best) or 0.0)
        service = (tax + turn_service_s) / best.speed
        completion = start + service
        best.free_at = completion
        if turn.session not in best.warm:
            best.migrated += 1
        best.touch(turn.session, warm_capacity)
        tier.store(turn.session)
        best.served += 1
        finished.append((completion, completion - turn.arrival))
    latencies = sorted(latency for _completion, latency in finished)
    on_time = sum(1 for latency in latencies if latency <= SLO_SECONDS)
    def pct(q):
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, int(q * len(latencies)))]
    return {
        "policy": policy,
        "requests": len(latencies),
        "unserved": unserved,
        "p50_s": pct(0.50), "p95_s": pct(0.95), "p99_s": pct(0.99),
        "slo_goodput": on_time / max(1, len(latencies)),
        "migrations": sum(w.migrated for w in fleet),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--g3", required=True)
    p.add_argument("--g3-warm", required=True)
    p.add_argument("--g3-prefill", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sessions", type=int, default=16,
                   help="size the fleet so the decoded turns fit the worker capacity: "
                        "128 turns of 0.6 s on 4 workers is ~75%% utilisation, while 512 "
                        "turns is 3x oversubscribed and every policy queues for hundreds "
                        "of seconds regardless of its cold route")
    p.add_argument("--turns-per-session", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--active-tokens", type=int, default=16384)
    p.add_argument("--bandwidth-gbs", type=float, default=None,
                   help="override the measured local H2D rate, e.g. to model a remote "
                        "worker link; the local measurement is 23.2-23.4 GB/s")
    p.add_argument("--warm-capacity", type=int, default=4,
                   help="sessions a worker keeps warm in HBM; the rest are evicted to "
                        "the tier, which is where a full-KV cold route starts to cost a "
                        "recompute instead of a move")
    p.add_argument("--tier-capacity", type=int, default=8,
                   help="session histories the *cluster* KV tier can hold; 0 means every "
                        "cold route is a recompute")
    p.add_argument("--escape-s", type=float, default=1.0,
                   help="sticky-until-saturated: the queue wait above which the warm "
                        "worker is abandoned")
    p.add_argument("--turn-service-s", type=float, default=0.6,
                   help="decode + warm-turn service time common to every policy; "
                        "without it the cold-route tax is the only cost and a policy "
                        "with a cheap cold route never queues")
    p.add_argument("--slow-worker-speed", type=float, default=0.35,
                   help="service-rate multiplier for the slow worker (0.35 is the regime the "
                        "earlier receipts used; 0.1 is a hotspot)")
    p.add_argument("--gap-model", choices=["poisson", "bursty"], default="bursty")
    p.add_argument("--regimes", default="balanced,slow_worker,worker_failure",
                   help="comma-separated regimes to replay.  The first three are the "
                        "worker-behaviour regimes; `burst` (a flash crowd) and `size_skew` "
                        "(a heavy-tailed session-size mix) are workload-shaped and opt-in, so "
                        "receipts that predate them are unchanged and the grid gains cells "
                        "rather than repeating them")
    p.add_argument("--burst-share", type=float, default=0.5,
                   help="share of sessions arriving inside the burst window (`burst` regime)")
    p.add_argument("--burst-window", type=float, default=4.0,
                   help="seconds the flash crowd's first turns are spread over")
    p.add_argument("--history-skew", type=float, default=1.2,
                   help="Zipf exponent over the sorted history choices (`size_skew` regime); "
                        "0 is the uniform mix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kv-bytes-per-token", type=int, default=0,
                   help="override the measured KV geometry (12,288 B/token on the G3 model).  "
                        "The payload a full-KV move pays is history x this number, so the model "
                        "size is a first-class axis of the phase diagram: an 8B-class model "
                        "(131,072 B/token) makes a 128K session a 16 GiB transfer and a 70B-class "
                        "one (327,680) makes it 40 GiB, both infeasible at the measured 23.3 GB/s "
                        "while rematerialising 8,192 tokens stays 0.203 s.  0 keeps the measured "
                        "value")
    p.add_argument("--history-choices", default="32768,131072,262144",
                   help="comma-separated session-history sizes the workload samples (the "
                        "measured corpus reaches 32K-262K; `1048576` adds the scale the "
                        "durability argument is about, with an extrapolated lookup row)")
    args = p.parse_args(argv)

    costs = load_costs(Path(args.g3), Path(args.g3_warm), Path(args.g3_prefill))
    if args.bandwidth_gbs:
        costs.bandwidth_gbs = float(args.bandwidth_gbs)
    if args.kv_bytes_per_token:
        costs.kv_bytes_per_token = int(args.kv_bytes_per_token)
    history_choices = tuple(int(x) for x in str(args.history_choices).split(",") if x.strip())
    turns = build_turns(args.sessions, args.turns_per_session, seed=args.seed,
                        gap_model=args.gap_model, active_tokens=args.active_tokens,
                        history_choices=history_choices)
    policies = ["strict_sticky", "sticky_saturated", "full_kv_move",
                "full_reprefill", "ephemeral"]
    rows = {}
    for policy in policies:
        rows[policy] = route(turns, costs, workers=args.workers, policy=policy,
                             turn_service_s=args.turn_service_s, escape_s=args.escape_s,
                             warm_capacity=args.warm_capacity,
                             tier_capacity=args.tier_capacity)
    regimes = {
        "balanced": {},
        "slow_worker": {"slow_worker": 0, "slow_worker_speed": args.slow_worker_speed},
        "worker_failure": {"fail_at": statistics.median(t.arrival for t in turns),
                           "fail_worker": 1},
    }
    # the workload-shaped regimes change the arrival or size distribution rather than a worker's
    # behaviour, so each replays its own turn stream at the same seed.  They are opt-in: a receipt
    # that does not ask for them is byte-identical to the grid measured before they existed, and
    # the phase summary adds their cells instead of duplicating the ones already there.
    regime_turns = {name: turns for name in regimes}
    selected = [r.strip() for r in str(args.regimes).split(",") if r.strip()]
    if "burst" in selected:
        regimes["burst"] = {}
        regime_turns["burst"] = build_turns(
            args.sessions, args.turns_per_session, seed=args.seed, gap_model=args.gap_model,
            active_tokens=args.active_tokens, history_choices=history_choices,
            burst={"at": statistics.median(t.arrival for t in turns),
                   "window": args.burst_window, "share": args.burst_share})
    if "size_skew" in selected:
        regimes["size_skew"] = {}
        regime_turns["size_skew"] = build_turns(
            args.sessions, args.turns_per_session, seed=args.seed, gap_model=args.gap_model,
            active_tokens=args.active_tokens, history_choices=history_choices,
            history_skew=args.history_skew)
    regimes = {name: kwargs for name, kwargs in regimes.items() if name in selected}
    by_regime = {}
    for name, kwargs in regimes.items():
        by_regime[name] = {
            policy: route(regime_turns[name], costs, workers=args.workers, policy=policy,
                          turn_service_s=args.turn_service_s, escape_s=args.escape_s,
                          warm_capacity=args.warm_capacity,
                          tier_capacity=args.tier_capacity, **kwargs)
            for policy in policies
        }
    def gate(table):
        """The plan's G4 Best gate, evaluated instead of asserted in prose.

        `advance` needs ephemeral mobility to beat the strongest baseline (by SLO
        goodput) by >= 1.5x or by >= 30% on p99.  `balanced_regression_ok` records the
        other half of the gate: balanced-load median latency must not regress by more
        than 5%.
        """
        baseline = max(("full_kv_move", "sticky_saturated", "strict_sticky"),
                       key=lambda name: table[name]["slo_goodput"])
        base, cand = table[baseline], table["ephemeral"]
        goodput_ratio = (cand["slo_goodput"] / base["slo_goodput"]
                         if base["slo_goodput"] > 0 else float("inf"))
        p99_gain = ((base["p99_s"] - cand["p99_s"]) / base["p99_s"]
                    if base and base["p99_s"] else None)
        return {
            "strongest_baseline": baseline,
            "goodput_ratio": goodput_ratio,
            "p99_reduction": p99_gain,
            "decision": "advance" if (goodput_ratio >= 1.5 or
                                      (p99_gain is not None and p99_gain >= 0.30))
                        else "not_established",
        }

    verdicts = {name: gate(table) for name, table in
                [("balanced", rows)] + list(by_regime.items())}
    ratio_regression = ((rows["ephemeral"]["p50_s"] - rows["full_kv_move"]["p50_s"])
                        / rows["full_kv_move"]["p50_s"])
    verdicts["balanced"]["median_regression_vs_full_kv_move"] = ratio_regression
    verdicts["balanced"]["median_regression_ok"] = ratio_regression <= 0.05

    payload = {
        "schema": "ephemeral-kv-g4-routing-replay-v1",
        "verdict": verdicts,
        "kind": "replay_simulation",
        "config": {**vars(args), "history_choices": list(history_choices)},
        "costs": {
            "kv_bytes_per_token": costs.kv_bytes_per_token,
            "bandwidth_gbs": costs.bandwidth_gbs,
            "lookup_s": costs.lookup_s,
            "active_prefill_s": costs.active_prefill_s,
            "full_prefill_s": costs.full_prefill_s,
        },
        "declared_workload": (
            "turn arrivals and tool gaps come from an explicit model; the public corpus "
            "has no timestamps, so this replays sessions against measured costs rather "
            "than measured arrivals"),
        "slo_seconds": SLO_SECONDS,
        "balanced": rows,
        "by_regime": by_regime,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print("verdicts:", json.dumps(verdicts, indent=1))
    for name, table in [("balanced", rows)] + list(by_regime.items()):
        print(f"[{name}]")
        for policy, row in table.items():
            print(f"   {policy:<15} p50 {row['p50_s']:.3f}s p99 {row['p99_s']:.3f}s "
                  f"goodput {row['slo_goodput']:.3f} migrations {row['migrations']} "
                  f"unserved {row['unserved']}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
