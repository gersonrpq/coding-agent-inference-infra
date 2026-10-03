"""Placement: which worker serves a request (the `pick` function), and the hop it implies.

LiteLLM's routing strategies (`least-busy`, `simple-shuffle`, ...) know nothing about sessions, and its own session affinity is
pure stickiness (our `aff`, measured worse). The policy that won the routing experiment, `affload`, is therefore code:

  session  hash of the first user message (a conversation keeps it; its cached history lives on the worker that served it)
  load     requests in flight per worker, counted here (exact), divided by the worker's weight
  rule     keep the session on its previous worker unless that worker has `slack` more requests in flight than the other;
           new sessions and batch requests go to the least loaded worker (batch gains little from locality and must not
           crowd interactive sessions)

PLACEMENT_POLICY=litellm leaves the choice to LiteLLM's `routing_strategy` and this module does nothing.

Three more things `pick` does live here:
  queue depth   the engine's waiting queue of each worker breaks ties between equally loaded workers (a scorer, not only an admit input)
  Shed          `pick` returns a worker id or a `Shed`: if no worker answers its metrics, the request is refused (`no_healthy_worker`);
                a worker that is down is never picked
  ramp          a worker that comes back is not slammed to 100 %: its weight starts at RAMP_FLOOR and grows to 1 over RAMP_SECONDS,
                and the clock only runs while the fleet TTFT p99 is below RAMP_HOLD_P99_S ("ramp while p99 holds")
Otherwise `pick` returns a worker id and the admission hook rewrites the model to that worker's alias.

A hop is a call served by a different worker than the session's previous one (src != dst). Nothing is moved by the gateway:
SGLang's hierarchical cache (GPU -> host RAM -> Mooncake) lets the destination read the prefix instead of recomputing it.
`pick` already knows src and dst, so it records the hop here (`orch_hops_total`, `orch_hop_tokens_total` with the estimated
prompt tokens). State is process-wide and imported once (security.place), like fleet_state.
"""
import hashlib
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass

from prometheus_client import Counter, Gauge

WORKERS = tuple(w for w in os.getenv("PLACEMENT_WORKERS", "w0,w1").split(",") if w)
POLICIES = ("litellm", "affload")
SESSION_TTL_S = 1800.0
SESSION_MAX = 4096
HOP_BACKEND = os.getenv("HOP_BACKEND", "mooncake")
RAMP_SECONDS = float(os.getenv("RAMP_SECONDS", "60"))
RAMP_FLOOR = float(os.getenv("RAMP_FLOOR", "0.25"))
RAMP_HOLD_P99_S = float(os.getenv("RAMP_HOLD_P99_S", "10"))   # half of the 20 s SLO: above it the ramp waits


@dataclass(frozen=True)
class PlacementRequest:
    session: str | None      # session_key(...); None when it cannot be told
    priority: int = 5
    is_batch: bool = False
    prompt_tokens: int = 0   # estimate, used only to size the hop that a move implies
    queue_depth: dict | None = None   # {worker: requests waiting in its engine queue}, from the telemetry (stale by seconds)


@dataclass(frozen=True)
class Shed:
    """`pick` answer when no worker can take the request (`Worker | Shed`)."""
    reason: str = "no_healthy_worker"
    code: int = 503
    retry_after_s: int = 2


def session_key(messages) -> str | None:
    """Stable id of a conversation: hash of its first user message (text parts are joined)."""
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        if isinstance(content, str) and content:
            return hashlib.sha1(content[:2000].encode()).hexdigest()[:16]
        return None
    return None


class Placement:
    def __init__(self, workers=WORKERS, weights=None, slack: float = 3.0, rng=None):
        self.workers = tuple(workers)
        self.weights = {w: 1.0 for w in self.workers}
        self.weights.update(weights or {})
        self.slack = slack
        self.rng = rng or random.Random()
        self.inflight = {w: 0 for w in self.workers}
        self.down: set[str] = set()                       # workers whose metrics do not answer
        self._progress: dict[str, float] = {}             # worker -> ramp progress 0..1 (absent = full weight)
        self._last_observe: float | None = None
        self.ramp_seconds, self.ramp_floor, self.ramp_hold_p99_s = RAMP_SECONDS, RAMP_FLOOR, RAMP_HOLD_P99_S
        self._sessions: OrderedDict[str, tuple[str, float]] = OrderedDict()   # session -> (worker, last seen)
        self._open: dict[str, str] = {}                                       # request id -> worker

    # --- load ---------------------------------------------------------------
    def ramp_factor(self, worker: str) -> float:
        """1.0 for a worker at full weight; RAMP_FLOOR..1 while it ramps up after coming back."""
        progress = self._progress.get(worker)
        return 1.0 if progress is None else self.ramp_floor + (1.0 - self.ramp_floor) * progress

    def load(self, worker: str) -> float:
        return self.inflight[worker] / (self.weights[worker] * self.ramp_factor(worker))

    def available(self) -> list[str]:
        return [w for w in self.workers if w not in self.down]

    def observe(self, up: set[str] | None, p99_s: float = 0.0, now: float | None = None) -> None:
        """Feed the telemetry. `up` = workers whose metrics answered the last scrape (None = no scrape yet: all are assumed up).
        A worker that was down and answers again starts its ramp; the ramp advances only while the fleet p99 is below the hold."""
        now = time.monotonic() if now is None else now
        elapsed = 0.0 if self._last_observe is None else min(now - self._last_observe, 30.0)
        self._last_observe = now
        returned: set[str] = set()
        if up is not None:
            for w in self.workers:
                if w not in up:
                    self.down.add(w)
                    self._progress.pop(w, None)
                elif w in self.down:                       # came back: the ramp starts now, not at the previous observation
                    self.down.discard(w)
                    self._progress[w] = 0.0
                    returned.add(w)
                    RAMPS.labels(w).inc()
        if p99_s <= self.ramp_hold_p99_s:
            for w in list(self._progress):
                if w in returned:
                    continue
                self._progress[w] += elapsed / self.ramp_seconds
                if self._progress[w] >= 1.0:
                    del self._progress[w]

    def least_loaded(self, queue_depth: dict | None = None) -> str:
        """The least loaded available worker; equal loads are broken by the shorter engine queue, then at random."""
        candidates = self.available() or list(self.workers)
        best = min(self.load(w) for w in candidates)
        tied = [w for w in candidates if self.load(w) == best]
        if queue_depth:
            shortest = min(queue_depth.get(w, 0) for w in tied)
            tied = [w for w in tied if queue_depth.get(w, 0) == shortest]
        return self.rng.choice(tied)

    def start(self, request_id: str, worker: str) -> None:
        self._open[request_id] = worker
        self.inflight[worker] += 1

    def finish(self, request_id: str | None) -> None:
        """Idempotent: success and failure events may both arrive."""
        worker = self._open.pop(request_id, None) if request_id else None
        if worker is not None:
            self.inflight[worker] -= 1

    # --- sessions -----------------------------------------------------------
    def _previous(self, session: str | None) -> str | None:
        if session is None:
            return None
        entry = self._sessions.get(session)
        if entry is None or time.monotonic() - entry[1] > SESSION_TTL_S:
            return None
        return entry[0]

    def _remember(self, session: str | None, worker: str) -> None:
        if session is None:
            return
        self._sessions[session] = (worker, time.monotonic())
        self._sessions.move_to_end(session)
        while len(self._sessions) > SESSION_MAX:
            self._sessions.popitem(last=False)

    # --- the decision -------------------------------------------------------
    def pick(self, request: PlacementRequest, policy: str) -> "str | Shed | None":
        """Worker id; a Shed when no worker answers; None to leave the choice to LiteLLM's router."""
        if policy == "litellm":
            return None
        if policy not in POLICIES:
            raise ValueError(f"unknown placement policy {policy!r}")
        if self.workers and not self.available():
            return Shed()

        previous = self._previous(request.session)
        if previous in self.down:
            previous = None                    # its worker is gone: the session starts again where the least loaded one is
        other = self.least_loaded(request.queue_depth)

        if request.is_batch or previous is None:
            worker = other
        elif self.load(previous) - self.load(other) >= self.slack:
            worker = other
        else:
            worker = previous

        if request.session is not None and not request.is_batch:
            result = "new" if previous is None else ("kept" if worker == previous else "moved")
            SESSIONS.labels(result).inc()
            if result == "moved":
                HOPS.labels(previous, worker, HOP_BACKEND).inc()
                HOP_TOKENS.labels(previous, worker, HOP_BACKEND).inc(max(request.prompt_tokens, 0))
            self._remember(request.session, worker)

        PLACED.labels(policy, worker, "batch" if request.is_batch else "interactive").inc()
        return worker


def pick(request: PlacementRequest, placement: "Placement", *, policy: str) -> "str | Shed | None":
    """`pick(req, workers, *, policy)`; the workers and their load live in `placement`."""
    return placement.pick(request, policy)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


PLACEMENT = Placement(slack=_env_float("PLACEMENT_SLACK", 3.0))

# --- metrics (exposed on LiteLLM's /metrics as litellm_orch_*) --------
PLACED = Counter("orch_placement_total", "Requests placed by the policy", ["policy", "worker", "priority_class"])
SESSIONS = Counter("orch_placement_session_total", "Interactive requests by what happened to their session", ["result"])
for _r in ("new", "kept", "moved"):
    SESSIONS.labels(_r)
HOPS = Counter("orch_hops_total", "Calls placed on a different worker than their session's previous call (src != dst)", ["src", "dst", "backend"])
HOP_TOKENS = Counter("orch_hop_tokens_total", "Estimated prompt tokens of those calls", ["src", "dst", "backend"])
RAMPS = Counter("orch_worker_ramps_total", "Times a worker came back and started its ramp", ["worker"])
AVAILABLE = Gauge("orch_worker_available", "1 if the worker's metrics answer, 0 if it is down", ["worker"])
RAMP_FACTOR = Gauge("orch_worker_ramp_factor", "Weight of the worker in placement: 1 = full, RAMP_FLOOR..1 while it ramps up", ["worker"])
INFLIGHT = Gauge("orch_placement_inflight", "Requests in flight per worker, counted by placement", ["worker"])
for _w in PLACEMENT.workers:
    INFLIGHT.labels(_w).set_function(lambda w=_w: PLACEMENT.inflight[w])
    AVAILABLE.labels(_w).set_function(lambda w=_w: 0 if w in PLACEMENT.down else 1)
    RAMP_FACTOR.labels(_w).set_function(lambda w=_w: PLACEMENT.ramp_factor(w))
    RAMPS.labels(_w)
