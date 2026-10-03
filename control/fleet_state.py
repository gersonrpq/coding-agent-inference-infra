import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

# Custom shed counter. Lives here (not in admission.py) because this module is
# imported exactly once, so the Counter is never registered twice no matter how
# LiteLLM imports admission. It is exposed on LiteLLM's /metrics; Prometheus
# prefixes it as litellm_orch_requests_shed_total (see prometheus-config).
SHED_REASONS = ("timeout_queue", "kv_pressure", "batch_pressure", "batch_share", "no_healthy_worker")   # all 503; the cap on in-flight requests is LiteLLM's own middleware
SHED_TOTAL = Counter(
    "orch_requests_shed_total",
    "Requests shed by the admission hook",
    labelnames=("reason", "status_code", "priority_class"),
)
for _reason in SHED_REASONS:
    for _cls in ("interactive", "batch"):
        SHED_TOTAL.labels(_reason, "503", _cls)

# Overflow gate: what happens to a failed call (stay = answer the error, leave = may go to the overflow model).
OVERFLOW_DECISIONS = Counter(
    "orch_overflow_decisions_total", "Overflow gate decisions after a failed call", ["decision", "status", "forwarded"]
)

# Requests that passed the admission hook. With SHED_TOTAL this gives the shed
# ratio per priority class.
ADMITTED_TOTAL = Counter(
    "orch_requests_admitted_total",
    "Requests admitted by the admission hook",
    labelnames=("priority_class",),
)
for _cls in ("interactive", "batch"):
    ADMITTED_TOTAL.labels(_cls)

# Per priority class, measured at the gateway on successful requests. These feed
# the interactive-vs-batch tail comparison (p99 spread).
REQUEST_LATENCY = Histogram(
    "orch_request_latency_seconds",
    "End-to-end latency of successful requests as seen by the gateway",
    labelnames=("priority_class",),
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60, 120, 300),
)
REQUEST_TTFT = Histogram(
    "orch_request_ttft_seconds",
    "Time to first token of successful streaming requests as seen by the gateway",
    labelnames=("priority_class",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
for _cls in ("interactive", "batch"):
    REQUEST_LATENCY.labels(_cls)
    REQUEST_TTFT.labels(_cls)

# The FleetSnapshot the hook decides on, exposed per replica.
REPLICA_QUEUE_DEPTH = Gauge(
    "orch_replica_queue_depth",
    "Engine waiting-queue depth per replica, as last scraped by the control plane",
    labelnames=("replica",),
)
REPLICA_RUNNING = Gauge(
    "orch_replica_running_requests",
    "Requests running on the engine per replica, as last scraped by the control plane",
    labelnames=("replica",),
)
REPLICA_KV_USED = Gauge(
    "orch_replica_kv_used_ratio",
    "KV cache used ratio (0-1) per replica, as last scraped by the control plane",
    labelnames=("replica",),
)


class cfg:
    KV_CAPACITY_TOKENS = 1_400_000   # fallback until the first scrape reports max_total_num_tokens
    TTFT_METRIC = "sglang:time_to_first_token_seconds"
    VERY_BAD_TAIL_MULTIPLIER = 4.0
    # Batch is sacrificed only when the tail is real, not merely wide: in this workload p99 is 7-8x p50 in normal operation
    # (the final F3 run: 373 batch refusals while the interactive p99 was 7.4 s). The floor is half of the 20 s SLO.
    VERY_BAD_TAIL_MIN_P99_S = float(os.getenv("BATCH_PRESSURE_MIN_P99_S", "10"))
    TTFT_WINDOW_S = 60.0     # TTFT quantiles use the last minute
    TTFT_MIN_SAMPLES = 20    # fewer samples than this: no tail/wait decisions


def worker_id(url: str) -> str:
    """'http://sglang-worker-1:30001' -> 'w1' (the placement name of the worker)."""
    host = urlparse(url).hostname or url
    tail = host.rsplit("-", 1)[-1]
    return f"w{tail}" if tail.isdigit() else host


@dataclass
class ReplicaState:
    url: str
    id: str
    waiting: int = 0
    running: int = 0
    kv_usage: float = 0.0
    kv_capacity_tokens: int = cfg.KV_CAPACITY_TOKENS

    # Prometheus histogram buckets:
    # {upper_bound_seconds: cumulative_count}
    ttft_buckets: dict[float, float] = field(
        default_factory=dict
    )


@dataclass
class FleetState:
    replicas: list[ReplicaState] = field(default_factory=list)
    scrape_ok_at: float = 0.0
    queue_wait_s: float = 0.0

    ttft_p50_s: float = 0.0
    ttft_p99_s: float = 0.0
    workers_up: set | None = None    # worker ids ("w0", "w1") whose metrics answered the last scrape; None = no scrape yet

    def queue_depth_by_worker(self) -> dict:
        """{worker id: requests waiting in its engine queue}, from the last scrape."""
        return {worker_id(r.url): r.waiting for r in self.replicas}

    @property
    def kv_usage_max(self) -> float:
        return max(
            (replica.kv_usage for replica in self.replicas),
            default=0.0,
        )

    @property
    def waiting_total(self) -> int:
        return sum(
            replica.waiting
            for replica in self.replicas
        )

    @property
    def estimated_queue_wait_s(self) -> float:
        if self.ttft_p50_s <= 0:
            return 0.0

        return (
            self.waiting_total
            * self.ttft_p50_s
        )

    @property
    def very_bad_tail_latency(self) -> bool:
        if self.ttft_p50_s <= 0:
            return False

        return (
            self.ttft_p99_s > cfg.VERY_BAD_TAIL_MULTIPLIER * self.ttft_p50_s
            and self.ttft_p99_s > cfg.VERY_BAD_TAIL_MIN_P99_S
        )


# Única instancia compartida por todo el proceso.
global_fleet_state = FleetState()

# Gauges evaluated at scrape time from the shared fleet state, so they show
# exactly what the admission hook sees right now.
FLEET_SNAPSHOT_AGE = Gauge(
    "orch_fleet_snapshot_age_seconds",
    "Seconds since the last successful fleet scrape (-1 = never scraped)",
)
FLEET_SNAPSHOT_AGE.set_function(
    lambda: (time.monotonic() - global_fleet_state.scrape_ok_at)
    if global_fleet_state.scrape_ok_at
    else -1.0
)
FLEET_EST_WAIT = Gauge(
    "orch_fleet_estimated_queue_wait_seconds",
    "Estimated queue wait used by the admission hook (waiting x TTFT p50)",
)
FLEET_EST_WAIT.set_function(lambda: global_fleet_state.estimated_queue_wait_s)
FLEET_TTFT = Gauge(
    "orch_fleet_ttft_seconds",
    "Fleet TTFT quantile computed by the control plane from the SGLang histogram",
    labelnames=("quantile",),
)
FLEET_TTFT.labels("0.5").set_function(lambda: global_fleet_state.ttft_p50_s)
FLEET_TTFT.labels("0.99").set_function(lambda: global_fleet_state.ttft_p99_s)

# (timestamp, aggregated cumulative TTFT buckets) snapshots for the sliding window.
_ttft_history: deque[tuple[float, dict[float, float]]] = deque()

# Única referencia al background task.
_metrics_task: asyncio.Task | None = None


def fleet_debug_enabled() -> bool:
    value = os.getenv("FLEET_LOG_DEBUG", "")
    return bool(value.strip())


def print_fleet_state(prefix: str = "Fleet"):
    """Debug dump of the fleet snapshot; silent unless FLEET_LOG_DEBUG is set."""
    if not fleet_debug_enabled():
        return

    fleet = global_fleet_state

    print(
        f"[FLEET] {prefix} | "
        f"replicas={len(fleet.replicas)} "
        f"waiting={fleet.waiting_total} "
        f"max_kv_usage={fleet.kv_usage_max:.1%} "
        f"ttft_p50={fleet.ttft_p50_s:.3f}s "
        f"ttft_p99={fleet.ttft_p99_s:.3f}s "
        f"estimated_wait={fleet.estimated_queue_wait_s:.3f}s "
        f"very_bad_tail={fleet.very_bad_tail_latency} "
        f"scrape_ok_at={fleet.scrape_ok_at:.3f}",
        flush=True,
    )

    for replica in fleet.replicas:
        print(
            f"[FLEET] replica={replica.url} "
            f"waiting={replica.waiting} "
            f"running={replica.running} "
            f"kv_usage={replica.kv_usage:.1%} "
            f"kv_capacity={replica.kv_capacity_tokens}",
            flush=True,
        )


def _get_replica_urls() -> list[str]:
    urls_env = os.getenv(
        "SGLANG_REPLICA_URLS",
        "http://sglang-worker-0:30000,http://sglang-worker-1:30001",
    )

    return [
        url.strip()
        for url in urls_env.split(",")
        if url.strip()
    ]


def _histogram_quantile(
    buckets: dict[float, float],
    quantile: float,
) -> float:
    """
    Aproxima un quantile a partir de buckets acumulativos
    de un histograma Prometheus.
    """
    if not buckets:
        return 0.0

    ordered = sorted(
        buckets.items(),
        key=lambda item: item[0],
    )

    total = ordered[-1][1]

    if total <= 0:
        return 0.0

    rank = quantile * total

    previous_bound = 0.0
    previous_count = 0.0

    for upper_bound, cumulative_count in ordered:
        if cumulative_count >= rank:
            bucket_count = (
                cumulative_count
                - previous_count
            )

            if bucket_count <= 0:
                return upper_bound

            position = (
                rank - previous_count
            ) / bucket_count

            if upper_bound == float("inf"):
                return previous_bound

            return (
                previous_bound
                + (
                    upper_bound
                    - previous_bound
                )
                * position
            )

        previous_bound = upper_bound
        previous_count = cumulative_count

    return ordered[-1][0]


def _aggregate_ttft_buckets(
    replicas: list[ReplicaState],
) -> dict[float, float]:
    """
    Agrega los histogramas acumulativos de todas las replicas.
    """
    aggregated: dict[float, float] = {}

    for replica in replicas:
        for upper_bound, count in replica.ttft_buckets.items():
            aggregated[upper_bound] = (
                aggregated.get(upper_bound, 0.0)
                + count
            )

    return aggregated


def _update_fleet_ttft(
    replicas: list[ReplicaState],
) -> None:
    """Fleet TTFT p50/p99 over the last TTFT_WINDOW_S, not since worker start.

    The scraped histogram is cumulative, so after a restart one cold request would
    dominate p99 for a long time. The window is the difference between the newest
    snapshot and the oldest one still inside the window; with fewer than
    TTFT_MIN_SAMPLES samples the quantiles are 0 (no basis for a decision).
    """
    now = time.monotonic()
    current = _aggregate_ttft_buckets(replicas)

    _ttft_history.append((now, current))

    while (
        len(_ttft_history) > 1
        and now - _ttft_history[1][0] >= cfg.TTFT_WINDOW_S
    ):
        _ttft_history.popleft()

    base = _ttft_history[0][1]
    window = {
        bound: max(0.0, count - base.get(bound, 0.0))   # a restarted worker resets its counters
        for bound, count in current.items()
    }

    samples = max(window.values(), default=0.0)

    if samples < cfg.TTFT_MIN_SAMPLES:
        global_fleet_state.ttft_p50_s = 0.0
        global_fleet_state.ttft_p99_s = 0.0
        return

    global_fleet_state.ttft_p50_s = _histogram_quantile(window, 0.50)
    global_fleet_state.ttft_p99_s = _histogram_quantile(window, 0.99)


async def _scrape_replica(
    client: httpx.AsyncClient,
    url: str,
) -> ReplicaState | None:
    try:
        response = await client.get(
            f"{url}/metrics",
            timeout=2.0,
        )
        response.raise_for_status()

        waiting = 0.0
        running = 0.0
        kv_cap = cfg.KV_CAPACITY_TOKENS
        kv_used = 0.0
        ttft_buckets: dict[float, float] = {}

        # With priority scheduling SGLang reports the total under priority=""
        # and a per-priority breakdown under priority="<n>"; summing every
        # sample would count each request twice, so only the total is used.
        for family in text_string_to_metric_families(
            response.text
        ):
            if family.name == "sglang:num_queue_reqs":
                waiting = sum(
                    sample.value
                    for sample in family.samples
                    if not sample.labels.get("priority")
                )

            elif family.name == "sglang:num_running_reqs":
                running = sum(
                    sample.value
                    for sample in family.samples
                    if not sample.labels.get("priority")
                )

            elif family.name == "sglang:max_total_num_tokens":
                kv_cap = sum(
                    sample.value
                    for sample in family.samples
                )

            elif family.name == "sglang:kv_used_tokens":
                kv_used = sum(
                    sample.value
                    for sample in family.samples
                )

            elif family.name == cfg.TTFT_METRIC:
                for sample in family.samples:
                    if not sample.name.endswith("_bucket"):
                        continue

                    le = sample.labels.get("le")

                    if le is None:
                        continue

                    try:
                        upper_bound = float(le)
                    except (TypeError, ValueError):
                        continue

                    ttft_buckets[upper_bound] = (
                        ttft_buckets.get(
                            upper_bound,
                            0.0,
                        )
                        + sample.value
                    )

        kv_usage = (
            kv_used / kv_cap
            if kv_cap > 0
            else 0.0
        )

        return ReplicaState(
            url=url,
            id=url,
            waiting=int(waiting),
            running=int(running),
            kv_usage=float(kv_usage),
            kv_capacity_tokens=int(kv_cap),
            ttft_buckets=ttft_buckets,
        )

    except Exception as exc:
        logger.warning(
            "Error scraping fleet replica %s: %s",
            url,
            exc,
        )
        return None


async def refresh_sglang_metrics() -> bool:
    replica_urls = _get_replica_urls()

    if not replica_urls:
        logger.warning(
            "No fleet replica URLs configured"
        )
        return False

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(
                _scrape_replica(client, url)
                for url in replica_urls
            )
        )

    replicas = [
        replica
        for replica in results
        if replica is not None
    ]

    global_fleet_state.replicas = replicas
    global_fleet_state.workers_up = {worker_id(r.url) for r in replicas}

    for replica in replicas:
        name = urlparse(replica.url).hostname or replica.url
        REPLICA_QUEUE_DEPTH.labels(name).set(replica.waiting)
        REPLICA_RUNNING.labels(name).set(replica.running)
        REPLICA_KV_USED.labels(name).set(replica.kv_usage)

    if replicas:
        global_fleet_state.scrape_ok_at = (
            time.monotonic()
        )

        _update_fleet_ttft(replicas)
    else:
        global_fleet_state.ttft_p50_s = 0.0
        global_fleet_state.ttft_p99_s = 0.0

    print_fleet_state("Fleet metrics updated")

    return bool(replicas)


async def poll_sglang_metrics(interval: int = 15):
    logger.info("Fleet metrics poller started (every %ss)", interval)

    while True:
        try:
            await refresh_sglang_metrics()

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "Unexpected error in fleet metrics poller"
            )

        await asyncio.sleep(interval)


def ensure_metrics_poller_started() -> asyncio.Task:
    global _metrics_task

    if (
        _metrics_task is None
        or _metrics_task.done()
    ):
        _metrics_task = asyncio.create_task(
            poll_sglang_metrics()
        )

    return _metrics_task


def start_metrics_poller():
    """
    Punto de entrada para el startup hook de LiteLLM.
    """
    return ensure_metrics_poller_started()
