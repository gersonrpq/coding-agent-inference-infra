import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)


class cfg:
    DEFAULT_MAX_NUM_SEQS = 44
    KV_CAPACITY_TOKENS = 1_400_000
    TTFT_METRIC = "sglang:time_to_first_token_seconds"
    VERY_BAD_TAIL_MULTIPLIER = 4.0


@dataclass
class ReplicaState:
    url: str
    id: str
    waiting: int = 0
    running: int = 0
    kv_usage: float = 0.0
    max_num_seqs: int = cfg.DEFAULT_MAX_NUM_SEQS
    kv_capacity_tokens: int = cfg.KV_CAPACITY_TOKENS
    in_flight: int = 0

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
            self.ttft_p99_s
            > cfg.VERY_BAD_TAIL_MULTIPLIER
            * self.ttft_p50_s
        )


# Única instancia compartida por todo el proceso.
global_fleet_state = FleetState()

# Única referencia al background task.
_metrics_task: asyncio.Task | None = None


def fleet_debug_enabled() -> bool:
    value = os.getenv("FLEET_LOG_DEBUG", "")
    return bool(value.strip())


def print_fleet_state(prefix: str = "Fleet"):
    fleet = global_fleet_state

    print(
        f"[FLEET] pid={os.getpid()} "
        f"fleet_id={id(fleet)} "
        f"task_id={id(_metrics_task) if _metrics_task else None} "
        f"{prefix} | "
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

    if fleet_debug_enabled():
        for replica in fleet.replicas:
            print(
                f"[FLEET] pid={os.getpid()} "
                f"replica={replica.url} "
                f"waiting={replica.waiting} "
                f"running={replica.running} "
                f"kv_usage={replica.kv_usage:.1%} "
                f"kv_capacity={replica.kv_capacity_tokens} "
                f"ttft_buckets={len(replica.ttft_buckets)}",
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
    buckets = _aggregate_ttft_buckets(replicas)

    global_fleet_state.ttft_p50_s = (
        _histogram_quantile(buckets, 0.50)
    )

    global_fleet_state.ttft_p99_s = (
        _histogram_quantile(buckets, 0.99)
    )


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

        for family in text_string_to_metric_families(
            response.text
        ):
            if family.name == "sglang:num_queue_reqs":
                waiting = sum(
                    sample.value
                    for sample in family.samples
                )

            elif family.name == "sglang:num_running_reqs":
                running = sum(
                    sample.value
                    for sample in family.samples
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
    print(
        f"🔥 Fleet metrics poller started "
        f"pid={os.getpid()}",
        flush=True,
    )

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
        print(
            f"🚀 Starting fleet metrics background task "
            f"pid={os.getpid()} "
            f"fleet_id={id(global_fleet_state)}",
            flush=True,
        )

        _metrics_task = asyncio.create_task(
            poll_sglang_metrics()
        )

    return _metrics_task


def start_metrics_poller():
    """
    Punto de entrada para el startup hook de LiteLLM.
    """
    return ensure_metrics_poller_started()
