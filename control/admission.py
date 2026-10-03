"""Admission: decide per request whether to run it or shed it, and with which reason; place it; bound its first token.

What this hook does, in order (cheap and stateless first):
  1. timeout_queue    (optional, off by default) engine queue would eat > half of the request's deadline
  2. kv_pressure      KV >= 95% and the shared prefix is new (cached prefixes are free)
  3. batch_pressure   fleet tail latency is very bad: batch is sacrificed first
  4. place            `affload` picks the worker and the model is rewritten to its alias (place.py); no worker answering is a 503

The cap on concurrent requests (12 running + K = 2 waiting in SGLang's priority queue = 14) is NOT here: it is LiteLLM's own
admission middleware (`max_in_flight_requests_per_worker: 14`, `max_queued_requests_per_worker: 0` in litellm/config.yaml), which
answers 503 at once when full. The gateway holds nothing.

Capacity sheds are 503 (the only code allowed to overflow). After a failed call the overflow gate decides stay or leave:
429, 500 and `slice_oom` stay, 503/529 may leave, and anything refused by the guard (`source: inspect`) stays. Forwarding to the
overflow model is off (OVERFLOW_ENABLED=0): the gate only counts what would leave.
Decisions read the cached FleetSnapshot (fleet_state), never Prometheus per request.
"""
import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger #type: ignore
from litellm.proxy._types import UserAPIKeyAuth #type: ignore
from security.fleet_state import ( #type: ignore
    ADMITTED_TOTAL,
    OVERFLOW_DECISIONS,
    REQUEST_LATENCY,
    REQUEST_TTFT,
    SHED_TOTAL,
    ensure_metrics_poller_started,
    global_fleet_state,
    print_fleet_state,
)
from security.place import PLACEMENT, PlacementRequest, Shed, session_key, pick as place_pick #type: ignore

logger = logging.getLogger(__name__)

INTERACTIVE_PRIORITY = 5             # priority <= 5 is interactive, > 5 is batch (lower = more urgent)
DEADLINE_S = {"interactive": 20.0, "batch": 120.0}   # request deadline per class; the router timeout is the ceiling
KV_SHED_THRESHOLD = float(os.getenv("KV_SHED_THRESHOLD", "0.95"))   # lowered only by the live KV probe (metrics/probes/kv_probe.py)
SHED_RETRY_AFTER_S = 2

# --- Switches (the defaults are the production behaviour) -------------------------------------
# Stateless checks that run first. timeout_queue is available but off by default: it reads gauges that are stale by seconds.
ADMISSION_CHECKS = {c for c in os.getenv("ADMISSION_CHECKS", "kv_pressure,batch_pressure").split(",") if c}
# Placement (control/place.py). `litellm` = leave the choice to LiteLLM's router (routing_strategy in config.yaml);
# `affload` picks the worker here and rewrites the model to that worker's alias.
PLACEMENT_POLICY = os.getenv("PLACEMENT_POLICY", "litellm")
WORKER_ALIASES = {"w0": "qwen-coding-w0", "w1": "qwen-coding-w1"}
# Overflow: forwarding to the external model is off; the gate decides and counts.
OVERFLOW_ENABLED = os.getenv("OVERFLOW_ENABLED", "0") == "1"
OVERFLOW_MAY_LEAVE = {503, 529}
# Batch is admitted only while fewer than this many requests (of ANY class) are in flight, so the last places of the cap
# (14 - 10 = 4) are kept for interactive calls (0 = no reservation). LiteLLM's cap does not know classes. This is the rule of
# the old counter (batch limit = 75 % of the places), which the final run V4b showed was needed.
BATCH_ADMIT_BELOW = int(os.getenv("BATCH_ADMIT_BELOW", "10"))
GENERATION_CALL_TYPES = {"completion", "acompletion", "text_completion", "atext_completion"}
PREFIX_TTL_S = 1800.0                # how long a seen prefix is assumed to still be cached
PREFIX_MAX_ENTRIES = 2048


async def litellm_worker_startup():
    """LiteLLM startup hook: start the fleet poller (state lives in fleet_state)."""
    ensure_metrics_poller_started()
    print_fleet_state("after startup")


def _cls(is_batch: bool) -> str:
    return "batch" if is_batch else "interactive"


def _get_priority(data: dict) -> int:
    value = data.get("priority")

    if value is None:
        value = (data.get("metadata") or {}).get("priority")

    try:
        return int(value) if value is not None else INTERACTIVE_PRIORITY
    except (TypeError, ValueError):
        return INTERACTIVE_PRIORITY


def _get_deadline_s(data: dict, is_batch: bool) -> float:
    default = DEADLINE_S[_cls(is_batch)]

    try:
        value = float(data.get("timeout"))
    except (TypeError, ValueError):
        return default

    return value if value > 0 else default


class _SeenPrefixes:
    """Shared prefixes (system prompt + tools) admitted recently, so assumed cached.

    Approximate on purpose: the engine may have evicted a prefix, but the gateway
    cannot ask per request. LRU with TTL, fleet-wide (workers share KV via Mooncake).
    """

    def __init__(self):
        self._seen: OrderedDict[str, float] = OrderedDict()

    def seen(self, key: str | None) -> bool:
        if key is None:
            return False

        at = self._seen.get(key)

        if at is None or time.monotonic() - at > PREFIX_TTL_S:
            return False

        self._seen.move_to_end(key)
        return True

    def add(self, key: str | None) -> None:
        if key is None:
            return

        self._seen[key] = time.monotonic()
        self._seen.move_to_end(key)

        while len(self._seen) > PREFIX_MAX_ENTRIES:
            self._seen.popitem(last=False)


PREFIXES = _SeenPrefixes()


class _InFlight:
    """Ids of the requests this hook admitted and has not seen close yet (any class); released by every closing event.
    Only used to apply BATCH_ADMIT_BELOW: the cap itself is LiteLLM's."""

    def __init__(self):
        self.ids: set[str] = set()


INFLIGHT = _InFlight()


def _prefix_key(data: dict) -> str | None:
    """Hash of what the agent repeats on every request: system prompt and tool definitions."""
    system = [
        m.get("content")
        for m in data.get("messages") or []
        if isinstance(m, dict) and m.get("role") in ("system", "developer")
    ]
    tools = data.get("tools")

    if not system and not tools:
        return None

    raw = json.dumps([system, tools], sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()


def _estimate_prompt_tokens(messages) -> int:
    """~4 characters per token of text; only used to size the hop that a placement move implies."""
    chars = 0
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(len(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return chars // 4


def overflow_decision(status: int | None, reason: str | None = None, source: str | None = None) -> str:
    """'leave' or 'stay' after a failed call. Only a capacity refusal (503/529) may leave; a 429, a 500, `slice_oom` and
    anything refused by the guard (`source: inspect`, even its fail-closed 503) stay."""
    if source == "inspect" or reason == "slice_oom" or status not in OVERFLOW_MAY_LEAVE:
        return "stay"
    return "leave"


def _shed(reason: str, is_batch: bool, detail: str, status: int = 503):
    SHED_TOTAL.labels(reason, str(status), _cls(is_batch)).inc()
    logger.warning("Admission shed reason=%s class=%s: %s", reason, _cls(is_batch), detail)

    raise HTTPException(
        status_code=status,
        detail=f"{detail} Try again later.",
        headers={"source": "admission", "retry-after": "2", "shed-reason": reason},
    )


@dataclass(frozen=True)
class ShedRequest:
    """What `should_shed` needs to know about a request."""
    is_batch: bool = False
    deadline_s: float = 20.0
    cached_prefix: bool = False


def should_shed(req: ShedRequest, snap, *, in_flight: int = 0, checks=None) -> tuple[bool, int | None, str | None, int | None]:
    """The admission decision (`should_shed(req, snap)`): `(shed?, code, reason, retry_after_seconds)`.

    `snap` is the cached FleetSnapshot (fleet_state), never Prometheus per request. Checks, cheap and stateless first:
      timeout_queue   (optional) the engine queue would eat more than half of the deadline
      kv_pressure     KV >= 95 % and the shared prefix is new (cached prefixes cost no new KV)
      batch_pressure  fleet tail latency is bad (p99 > 4x p50 and > 10 s): batch is sacrificed first
      batch_share     batch only enters while fewer than BATCH_ADMIT_BELOW requests of any class are in flight
    The cap on requests in flight is LiteLLM's middleware, before this; a capacity shed is a 503 (the only code that may overflow).
    """
    checks = ADMISSION_CHECKS if checks is None else checks
    if "timeout_queue" in checks and snap.ttft_p50_s > 0 and snap.estimated_queue_wait_s > req.deadline_s / 2:
        return True, 503, "timeout_queue", SHED_RETRY_AFTER_S
    if "kv_pressure" in checks and snap.kv_usage_max >= KV_SHED_THRESHOLD and not req.cached_prefix:
        return True, 503, "kv_pressure", SHED_RETRY_AFTER_S
    if "batch_pressure" in checks and snap.very_bad_tail_latency and req.is_batch:
        return True, 503, "batch_pressure", SHED_RETRY_AFTER_S
    if req.is_batch and BATCH_ADMIT_BELOW and in_flight >= BATCH_ADMIT_BELOW:
        return True, 503, "batch_share", SHED_RETRY_AFTER_S
    return False, None, None, None


def _shed_detail(reason: str, req: ShedRequest, snap, in_flight: int) -> str:
    return {
        "timeout_queue": f"Estimated engine queue wait {snap.estimated_queue_wait_s:.1f}s exceeds half of the {req.deadline_s:.0f}s deadline.",
        "kv_pressure": f"Fleet KV usage is {snap.kv_usage_max:.0%} and the prefix is new.",
        "batch_pressure": "Fleet tail latency is too high for batch requests.",
        "batch_share": f"{in_flight} requests in flight: the last places are kept for interactive calls.",
        "no_healthy_worker": "No worker answers its metrics.",
    }.get(reason, reason)


def _request_metadata(kwargs_or_data: dict) -> dict:
    """Metadata written by the pre-call hook, as seen from a request dict or a log event."""
    in_params = (kwargs_or_data.get("litellm_params") or {}).get("metadata")
    return in_params or kwargs_or_data.get("metadata") or {}


class AdmissionPreCallHandler(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        if call_type not in GENERATION_CALL_TYPES:
            return data

        ensure_metrics_poller_started()   # idempotent: one shared poller
        print_fleet_state(f"before admission [{call_type}]")

        fleet = global_fleet_state
        priority = _get_priority(data)
        is_batch = priority > INTERACTIVE_PRIORITY
        deadline_s = _get_deadline_s(data, is_batch)
        prefix = _prefix_key(data)
        cached_prefix = PREFIXES.seen(prefix)

        shed_req = ShedRequest(is_batch, deadline_s, cached_prefix)
        shed, code, reason, _retry_after = should_shed(shed_req, fleet, in_flight=len(INFLIGHT.ids))

        if shed:
            _shed(reason, is_batch, _shed_detail(reason, shed_req, fleet, len(INFLIGHT.ids)), status=code)

        placement_worker = None
        placement_id = None

        if PLACEMENT_POLICY != "litellm":
            PLACEMENT.observe(fleet.workers_up, fleet.ttft_p99_s)          # who answers, and the ramp of a worker that came back
            placement_worker = place_pick(
                PlacementRequest(session_key(data.get("messages")), priority, is_batch,
                                 _estimate_prompt_tokens(data.get("messages")), fleet.queue_depth_by_worker()),
                PLACEMENT,
                policy=PLACEMENT_POLICY,
            )

            if isinstance(placement_worker, Shed):
                _shed(placement_worker.reason, is_batch, _shed_detail(placement_worker.reason, shed_req, fleet, 0), status=placement_worker.code)

            if placement_worker:
                data["model"] = WORKER_ALIASES[placement_worker]
                placement_id = uuid.uuid4().hex
                PLACEMENT.start(placement_id, placement_worker)

        inflight_id = uuid.uuid4().hex
        INFLIGHT.ids.add(inflight_id)

        PREFIXES.add(prefix)

        metadata = data.setdefault("metadata", {})
        metadata.update(
            fleet_priority=priority,
            fleet_deadline_s=deadline_s,
            fleet_interactive=not is_batch,
            fleet_cached_prefix=cached_prefix,
            fleet_waiting_reqs=fleet.waiting_total,
            fleet_max_kv_usage=fleet.kv_usage_max,
            fleet_estimated_wait_s=fleet.estimated_queue_wait_s,
            placement_worker=placement_worker,
            placement_id=placement_id,
            inflight_id=inflight_id,
        )

        # LiteLLM does not forward `priority` to the engine on its own.
        data["extra_body"] = {**(data.get("extra_body") or {}), "priority": priority}

        ADMITTED_TOTAL.labels(_cls(is_batch)).inc()
        return data

    def _release(self, metadata: dict) -> None:
        PLACEMENT.finish(metadata.get("placement_id"))
        INFLIGHT.ids.discard(metadata.get("inflight_id"))

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Give the placement count back and record per-class latency (interactive vs batch p99 spread)."""
        try:
            metadata = _request_metadata(kwargs)
            self._release(metadata)

            if "fleet_interactive" not in metadata:
                return

            priority_class = "interactive" if metadata["fleet_interactive"] else "batch"
            REQUEST_LATENCY.labels(priority_class).observe((end_time - start_time).total_seconds())

            first_token_at = kwargs.get("completion_start_time")

            if kwargs.get("stream") and first_token_at is not None:
                REQUEST_TTFT.labels(priority_class).observe((first_token_at - start_time).total_seconds())

        except Exception:
            logger.exception("Failed to release placement / record per-class latency")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._release(_request_metadata(kwargs))

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None):
        metadata = _request_metadata(request_data)
        self._release(metadata)
        self._gate_overflow(original_exception)

        return None


    @staticmethod
    def _gate_overflow(exc) -> None:
        """Stay or leave? Counted for every failed call; the request would be forwarded to the overflow model only when
        OVERFLOW_ENABLED=1 (off: the gate measures how much would leave, without calling the external model)."""
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int):
            return
        headers = getattr(exc, "headers", None) or {}
        decision = overflow_decision(status, headers.get("shed-reason"), headers.get("source"))
        OVERFLOW_DECISIONS.labels(decision, str(status), "yes" if decision == "leave" and OVERFLOW_ENABLED else "no").inc()


admission_handler = AdmissionPreCallHandler()
