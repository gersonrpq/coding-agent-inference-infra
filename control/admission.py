import logging
import os
from typing import Any, Literal

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

from security.fleet_state import (
    ensure_metrics_poller_started,
    global_fleet_state,
    print_fleet_state,
)

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_S = 10.0
KV_SHED_THRESHOLD = 0.95
INTERACTIVE_PRIORITY = 5


async def litellm_worker_startup():
    """
    Startup hook de LiteLLM.

    El estado y el background task viven en
    security.fleet_state, por lo que no dependen de
    cómo LiteLLM importe este módulo.
    """
    print(
        f"🚀 LiteLLM worker startup pid={os.getpid()}",
        flush=True,
    )

    ensure_metrics_poller_started()

    print_fleet_state("after startup")


def _get_timeout_s(data: dict) -> float:
    value = data.get("timeout")

    if value is None:
        return DEFAULT_TIMEOUT_S

    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S

    if timeout <= 0:
        return DEFAULT_TIMEOUT_S

    return timeout


def _get_priority(data: dict) -> int:
    value = data.get("priority")

    if value is None:
        metadata = data.get("metadata") or {}
        value = metadata.get("priority")

    if value is None:
        return INTERACTIVE_PRIORITY

    try:
        return int(value)
    except (TypeError, ValueError):
        return INTERACTIVE_PRIORITY


def _has_cached_prefix(data: dict) -> bool:
    """
    LiteLLM puede no incluir prompt_tokens_details.

    Si existe y cached_tokens > 0, consideramos que
    el prefix ya está presente en la cache.
    """
    prompt_tokens_details = data.get(
        "prompt_tokens_details"
    )

    if not isinstance(
        prompt_tokens_details,
        dict,
    ):
        return False

    cached_tokens = prompt_tokens_details.get(
        "cached_tokens",
        0,
    )

    try:
        return float(cached_tokens) > 0
    except (TypeError, ValueError):
        return False


class AdmissionPreCallHandler(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> dict:
        print(
            f"🔥 ADMISSION PRE-CALL: "
            f"{call_type} "
            f"pid={os.getpid()}",
            flush=True,
        )

        # Garantiza que exista un único poller compartido.
        ensure_metrics_poller_started()

        fleet = global_fleet_state

        print(fleet)

        print_fleet_state(
            f"before admission [{call_type}]"
        )

        timeout_s = _get_timeout_s(data)
        priority = _get_priority(data)

        is_interactive = (
            priority <= INTERACTIVE_PRIORITY
        )
        is_batch = not is_interactive

        has_cached_prefix = _has_cached_prefix(data)

        estimated_wait_s = (
            fleet.estimated_queue_wait_s
        )

        # ---------------------------------------------------------
        # 1. Request probablemente esperaría más de la mitad
        #    de su timeout.
        # ---------------------------------------------------------
        if (
            fleet.ttft_p50_s > 0
            and estimated_wait_s > timeout_s / 2
        ):
            logger.warning(
                "Admission shed: estimated wait "
                "%.3fs > timeout/2 %.3fs "
                "(waiting=%d, p50_ttft=%.3fs, "
                "timeout=%.3fs)",
                estimated_wait_s,
                timeout_s / 2,
                fleet.waiting_total,
                fleet.ttft_p50_s,
                timeout_s,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Fleet queue wait is too high "
                    f"(estimated: {estimated_wait_s:.2f}s, "
                    f"timeout: {timeout_s:.2f}s). "
                    "Try again later."
                ),
            )

        # ---------------------------------------------------------
        # 2. KV casi lleno.
        #
        #    Si el prefix ya está cacheado, permitimos la request.
        #    Si es un prefix nuevo, hacemos shed.
        # ---------------------------------------------------------
        if (
            fleet.kv_usage_max >= KV_SHED_THRESHOLD
            and not has_cached_prefix
        ):
            logger.warning(
                "Admission shed: fleet KV usage %.1f%% "
                "and no cached prefix",
                fleet.kv_usage_max * 100,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Fleet KV capacity is too high "
                    f"({fleet.kv_usage_max:.1%}) "
                    "for a new prefix. "
                    "Try again later."
                ),
            )

        # ---------------------------------------------------------
        # 3. Tail latency muy degradada.
        #
        #    Interactive: aceptar.
        #    Batch: shed.
        # ---------------------------------------------------------
        if (
            fleet.very_bad_tail_latency
            and is_batch
        ):
            logger.warning(
                "Admission shed: very bad fleet tail latency "
                "for batch request "
                "(p50=%.3fs, p99=%.3fs, "
                "priority=%d)",
                fleet.ttft_p50_s,
                fleet.ttft_p99_s,
                priority,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Fleet tail latency is too high "
                    "for batch requests. "
                    "Try again later."
                ),
            )

        # Metadata para observabilidad.
        metadata = data.setdefault(
            "metadata",
            {},
        )

        metadata["fleet_waiting_reqs"] = (
            fleet.waiting_total
        )
        metadata["fleet_max_kv_usage"] = (
            fleet.kv_usage_max
        )
        metadata["fleet_replicas"] = (
            len(fleet.replicas)
        )
        metadata["fleet_ttft_p50_s"] = (
            fleet.ttft_p50_s
        )
        metadata["fleet_ttft_p99_s"] = (
            fleet.ttft_p99_s
        )
        metadata["fleet_estimated_wait_s"] = (
            estimated_wait_s
        )
        metadata["fleet_very_bad_tail_latency"] = (
            fleet.very_bad_tail_latency
        )
        metadata["fleet_cached_prefix"] = (
            has_cached_prefix
        )
        metadata["fleet_priority"] = priority
        metadata["fleet_interactive"] = (
            is_interactive
        )

        return data


admission_handler = AdmissionPreCallHandler()


print(
    f"### ADMISSION MODULE READY pid={os.getpid()} ###",
    flush=True,
)
