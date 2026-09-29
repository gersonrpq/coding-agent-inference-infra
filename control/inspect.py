from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger


# ---------------------------------------------------------------------------
# MODULE INIT
# ---------------------------------------------------------------------------

print(
    ">>> SECURITY_INSPECT MODULE LOADED <<<",
    flush=True,
)

logger = logging.getLogger("litellm.security.inspect")


# ---------------------------------------------------------------------------
# POLICY
# ---------------------------------------------------------------------------

MODEL = "qwen-coding-local"

CONTEXT_WINDOW = 65_536

# Input máximo permitido.
MAX_INPUT_TOKENS = 60_000

# Output máximo permitido.
MAX_OUTPUT_TOKENS = 4_096

# ---------------------------------------------------------------------------
# REQUEST LIMITS
# ---------------------------------------------------------------------------

MAX_MESSAGES = 128

MAX_MESSAGE_CHARS = 500_000

MAX_TOTAL_CONTENT_CHARS = 2_000_000

MAX_TOOLS = 64

MAX_TOOL_SCHEMA_BYTES = 64 * 1024

MAX_REQUEST_BYTES = 4 * 1024 * 1024

MAX_JSON_DEPTH = 12

# ---------------------------------------------------------------------------
# RATE LIMIT
# ---------------------------------------------------------------------------
#
# Rate limit GLOBAL dentro de este proceso/pod.
#
# Ejemplo:
#   10 requests/second
#   burst máximo de 20 requests
#
# Si tienes N pods, cada pod tiene su propio bucket.
#
# Para un rate limit realmente global de Kubernetes necesitarías Redis
# u otro mecanismo compartido.
# ---------------------------------------------------------------------------

GLOBAL_RATE_LIMIT_RPS = 10.0

GLOBAL_RATE_LIMIT_BURST = 20.0

# No permitimos una cantidad ilimitada de requests ejecutándose
# simultáneamente dentro de esta instancia.
MAX_CONCURRENT_REQUESTS = 22


# ---------------------------------------------------------------------------
# MESSAGE POLICY
# ---------------------------------------------------------------------------

ALLOWED_ROLES = {
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
}


FORBIDDEN_PARAMS = {
    "best_of",
    "num_return_sequences",
    "num_beams",
}


# ---------------------------------------------------------------------------
# GLOBAL RATE LIMIT STATE
# ---------------------------------------------------------------------------

_rate_lock = asyncio.Lock()

_rate_tokens = GLOBAL_RATE_LIMIT_BURST

_rate_last_update = time.monotonic()

_concurrency_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_REQUESTS
)


async def _acquire_rate_limit() -> None:
    """
    Consume un token del bucket global de esta instancia.

    No espera.

    Si no hay capacidad disponible, rechaza inmediatamente
    con HTTP 429.
    """

    global _rate_tokens
    global _rate_last_update

    if GLOBAL_RATE_LIMIT_RPS <= 0:
        return

    now = time.monotonic()

    async with _rate_lock:

        elapsed = now - _rate_last_update

        if elapsed > 0:
            _rate_tokens = min(
                GLOBAL_RATE_LIMIT_BURST,
                _rate_tokens
                + elapsed * GLOBAL_RATE_LIMIT_RPS,
            )

            _rate_last_update = now

        if _rate_tokens < 1.0:

            logger.warning(
                "SECURITY_INSPECT RATE_LIMIT_REJECT "
                "tokens=%.3f rps=%.3f burst=%.3f",
                _rate_tokens,
                GLOBAL_RATE_LIMIT_RPS,
                GLOBAL_RATE_LIMIT_BURST,
            )

            _reject(
                429,
                "rate_limit_exceeded",
                "Too many requests",
            )

        _rate_tokens -= 1.0


# ---------------------------------------------------------------------------
# LOGGING / IDENTITY
# ---------------------------------------------------------------------------

def _caller_id(
    user_api_key_dict: Any,
) -> str:
    """
    Identificador anónimo para logs.

    Nunca registramos la API key.
    """

    if user_api_key_dict is None:
        return "anonymous"

    for attr in (
        "user_id",
        "team_id",
        "token",
    ):
        try:
            value = getattr(
                user_api_key_dict,
                attr,
                None,
            )
        except Exception:
            value = None

        if value:
            return hashlib.sha256(
                str(value).encode()
            ).hexdigest()[:12]

    return "anonymous"


def _reject(
    status: int,
    code: str,
    message: str,
) -> None:

    logger.warning(
        "SECURITY_INSPECT REJECT "
        "status=%s reason=%s message=%s",
        status,
        code,
        message,
    )

    error_type = (
        "api_error"
        if status >= 500
        else "invalid_request_error"
    )

    raise HTTPException(
        status_code=status,
        detail={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


# ---------------------------------------------------------------------------
# SAFE SIZE CALCULATION
# ---------------------------------------------------------------------------

def _estimate_object_bytes(
    value: Any,
    depth: int = 0,
) -> int:
    """
    Estimación segura del tamaño de una estructura.

    IMPORTANTE:

    No usamos json.dumps(value).

    LiteLLM puede añadir objetos internos como UserAPIKeyAuth,
    que no son JSON serializables.

    Esta función mide strings, bytes, números, listas y diccionarios
    sin depender de que los objetos internos sean serializables.
    """

    if depth > MAX_JSON_DEPTH:
        return 0

    if value is None:
        return 4

    if isinstance(value, str):
        return len(value.encode("utf-8"))

    if isinstance(value, bytes):
        return len(value)

    if isinstance(value, bool):
        return 4

    if isinstance(value, (int, float)):
        return 16

    if isinstance(value, dict):

        total = 2

        for key, item in value.items():

            if isinstance(key, str):
                total += len(
                    key.encode("utf-8")
                )
            else:
                total += 32

            total += _estimate_object_bytes(
                item,
                depth + 1,
            )

        return total

    if isinstance(value, list):

        total = 2

        for item in value:
            total += _estimate_object_bytes(
                item,
                depth + 1,
            )

        return total

    if isinstance(value, tuple):

        total = 2

        for item in value:
            total += _estimate_object_bytes(
                item,
                depth + 1,
            )

        return total

    # Objetos internos de LiteLLM.
    #
    # No intentamos serializarlos.
    # Solo contabilizamos una cantidad pequeña y constante.
    return 64


def _request_size(
    data: dict[str, Any],
) -> int:
    """
    Calcula el tamaño de la parte controlable por el cliente.

    NO serializa el objeto completo recibido por LiteLLM.

    Esto evita problemas con objetos internos como:
        UserAPIKeyAuth
    """

    size = 0

    fields = (
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "temperature",
        "top_p",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
    )

    for field in fields:

        if field not in data:
            continue

        size += _estimate_object_bytes(
            data[field]
        )

    return size


def _check_size(
    data: dict[str, Any],
) -> None:

    try:
        request_size = _request_size(data)

    except Exception:

        logger.exception(
            "SECURITY_INSPECT request_size_failed"
        )

        _reject(
            503,
            "security_inspection_failed",
            "Unable to inspect request size",
        )

    logger.info(
        "SECURITY_INSPECT request_size=%d "
        "max_request_bytes=%d",
        request_size,
        MAX_REQUEST_BYTES,
    )

    if request_size > MAX_REQUEST_BYTES:

        _reject(
            413,
            "request_too_large",
            "Request body is too large",
        )


# ---------------------------------------------------------------------------
# JSON DEPTH
# ---------------------------------------------------------------------------

def _json_depth(
    value: Any,
    depth: int = 0,
) -> int:

    if depth > MAX_JSON_DEPTH:
        return depth

    if isinstance(value, dict):

        if not value:
            return depth

        return max(
            _json_depth(
                item,
                depth + 1,
            )
            for item in value.values()
        )

    if isinstance(value, list):

        if not value:
            return depth

        return max(
            _json_depth(
                item,
                depth + 1,
            )
            for item in value
        )

    if isinstance(value, tuple):

        if not value:
            return depth

        return max(
            _json_depth(
                item,
                depth + 1,
            )
            for item in value
        )

    return depth


# ---------------------------------------------------------------------------
# SHAPE VALIDATION
# ---------------------------------------------------------------------------

def _check_shape(
    data: dict[str, Any],
) -> None:

    if not isinstance(data, dict):

        _reject(
            400,
            "invalid_request",
            "Request body must be an object",
        )

    try:
        depth = _json_depth(data)

    except Exception:

        logger.exception(
            "SECURITY_INSPECT depth_check_failed"
        )

        _reject(
            503,
            "security_inspection_failed",
            "Unable to inspect request structure",
        )

    if depth > MAX_JSON_DEPTH:

        _reject(
            413,
            "request_too_deep",
            "Request nesting is too deep",
        )

    messages = data.get("messages")

    if not isinstance(messages, list):

        _reject(
            400,
            "invalid_messages",
            "messages must be an array",
        )

    if not messages:

        _reject(
            400,
            "empty_messages",
            "messages cannot be empty",
        )

    if len(messages) > MAX_MESSAGES:

        _reject(
            413,
            "too_many_messages",
            "Too many messages",
        )


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def _check_model(
    data: dict[str, Any],
) -> None:

    if data.get("model") != MODEL:

        _reject(
            403,
            "model_not_allowed",
            (
                f"Requested model "
                f"'{data.get('model')}' "
                f"is not allowed"
            ),
        )


# ---------------------------------------------------------------------------
# MESSAGE VALIDATION
# ---------------------------------------------------------------------------

def _content_text(
    content: Any,
) -> str:

    if isinstance(content, str):
        return content

    if content is None:
        return ""

    if isinstance(content, list):

        parts: list[str] = []

        for item in content:

            if (
                isinstance(item, dict)
                and isinstance(
                    item.get("text"),
                    str,
                )
            ):
                parts.append(
                    item["text"]
                )

        return "\n".join(parts)

    return str(content)


def _check_messages(
    messages: list[Any],
) -> None:

    total_chars = 0

    for index, message in enumerate(
        messages
    ):

        if not isinstance(
            message,
            dict,
        ):

            _reject(
                400,
                "invalid_message",
                (
                    f"Invalid message "
                    f"at index {index}"
                ),
            )

        role = message.get("role")

        if role not in ALLOWED_ROLES:

            _reject(
                400,
                "invalid_role",
                (
                    "Unsupported message role "
                    f"at index {index}"
                ),
            )

        content = message.get(
            "content"
        )

        if content is None:
            continue

        # Text-only fleet.
        if isinstance(
            content,
            list,
        ):

            _reject(
                400,
                "multimodal_not_allowed",
                "Multimodal content is not enabled",
            )

        if not isinstance(
            content,
            str,
        ):

            _reject(
                400,
                "invalid_content",
                "Message content must be text",
            )

        content_chars = len(content)

        if (
            content_chars
            > MAX_MESSAGE_CHARS
        ):

            _reject(
                413,
                "message_too_large",
                "Message is too large",
            )

        total_chars += content_chars

        if (
            total_chars
            > MAX_TOTAL_CONTENT_CHARS
        ):

            _reject(
                413,
                "request_content_too_large",
                "Total request content is too large",
            )


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

def _check_tools(
    data: dict[str, Any],
) -> None:

    tools = data.get("tools")

    if tools is None:
        return

    if not isinstance(
        tools,
        list,
    ):

        _reject(
            400,
            "invalid_tools",
            "tools must be an array",
        )

    if len(tools) > MAX_TOOLS:

        _reject(
            413,
            "too_many_tools",
            "Too many tools",
        )

    for index, tool in enumerate(
        tools
    ):

        if not isinstance(
            tool,
            dict,
        ):

            _reject(
                400,
                "invalid_tool",
                (
                    "Invalid tool definition "
                    f"at index {index}"
                ),
            )

        try:
            tool_size = (
                _estimate_object_bytes(
                    tool
                )
            )

        except Exception:

            logger.exception(
                "SECURITY_INSPECT "
                "tool_size_failed"
            )

            _reject(
                503,
                "security_inspection_failed",
                "Unable to inspect tool schema",
            )

        if (
            tool_size
            > MAX_TOOL_SCHEMA_BYTES
        ):

            _reject(
                413,
                "tool_schema_too_large",
                "Tool schema is too large",
            )


# ---------------------------------------------------------------------------
# OUTPUT TOKEN LIMIT
# ---------------------------------------------------------------------------

def _max_output_tokens(
    data: dict[str, Any],
) -> int:

    value = data.get(
        "max_completion_tokens"
    )

    if value is None:

        value = data.get(
            "max_tokens"
        )

    # Ausencia = límite seguro.
    if value is None:
        return MAX_OUTPUT_TOKENS

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):

        _reject(
            400,
            "invalid_max_tokens",
            "max_tokens must be an integer",
        )

    return value


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

def _check_parameters(
    data: dict[str, Any],
) -> None:

    for parameter in FORBIDDEN_PARAMS:

        if parameter in data:

            _reject(
                400,
                "parameter_not_allowed",
                (
                    f"Parameter '{parameter}' "
                    "is not allowed"
                ),
            )

    temperature = data.get(
        "temperature"
    )

    if temperature is not None:

        if (
            isinstance(
                temperature,
                bool,
            )
            or not isinstance(
                temperature,
                (int, float),
            )
            or not math.isfinite(
                float(temperature)
            )
            or not 0
            <= float(temperature)
            <= 2
        ):

            _reject(
                400,
                "invalid_temperature",
                "temperature must be between 0 and 2",
            )

    top_p = data.get(
        "top_p"
    )

    if top_p is not None:

        if (
            isinstance(
                top_p,
                bool,
            )
            or not isinstance(
                top_p,
                (int, float),
            )
            or not math.isfinite(
                float(top_p)
            )
            or not 0
            <= float(top_p)
            <= 1
        ):

            _reject(
                400,
                "invalid_top_p",
                "top_p must be between 0 and 1",
            )

    n = data.get("n")

    if n is not None and n != 1:

        _reject(
            400,
            "invalid_n",
            "Only n=1 is allowed",
        )

    output_tokens = (
        _max_output_tokens(data)
    )

    if output_tokens <= 0:

        _reject(
            400,
            "invalid_max_tokens",
            "max_tokens must be greater than zero",
        )

    if (
        output_tokens
        > MAX_OUTPUT_TOKENS
    ):

        _reject(
            400,
            "max_tokens_exceeded",
            (
                f"Maximum output is "
                f"{MAX_OUTPUT_TOKENS} tokens"
            ),
        )


# ---------------------------------------------------------------------------
# TOKEN ESTIMATION
# ---------------------------------------------------------------------------

def _estimate_tokens_sync(
    messages: list[dict[str, Any]],
) -> int:

    """
    Intenta usar el tokenizador de LiteLLM.

    Si no está disponible, sobreestima usando chars / 3.
    """

    try:

        from litellm.utils import (
            token_counter,
        )

        result = token_counter(
            model="Qwen/Qwen3.6-27B-FP8",
            messages=messages,
        )

        if (
            isinstance(result, int)
            and result >= 0
        ):

            return result

    except Exception:

        logger.debug(
            "SECURITY_INSPECT "
            "tokenizer_unavailable",
            exc_info=True,
        )

    chars = 0

    for message in messages:

        content = _content_text(
            message.get("content")
        )

        chars += (
            len(content)
            + 32
        )

    return math.ceil(
        chars / 3
    )


async def _estimate_tokens(
    messages: list[dict[str, Any]],
) -> int:

    return await asyncio.to_thread(
        _estimate_tokens_sync,
        messages,
    )


# ---------------------------------------------------------------------------
# CONTEXT VALIDATION
# ---------------------------------------------------------------------------

async def _check_context(
    data: dict[str, Any],
) -> tuple[int, int, int]:

    messages = data[
        "messages"
    ]

    input_tokens = (
        await _estimate_tokens(
            messages
        )
    )

    output_tokens = (
        _max_output_tokens(data)
    )

    requested_context = (
        input_tokens
        + output_tokens
    )

    logger.info(
        "SECURITY_INSPECT context "
        "input_tokens=%d "
        "output_tokens=%d "
        "requested_context=%d "
        "window=%d",
        input_tokens,
        output_tokens,
        requested_context,
        CONTEXT_WINDOW,
    )

    if (
        input_tokens
        > MAX_INPUT_TOKENS
    ):

        _reject(
            400,
            "input_context_too_large",
            "Input context is too large",
        )

    if (
        requested_context
        > CONTEXT_WINDOW
    ):

        _reject(
            400,
            "context_window_exceeded",
            "Request exceeds model context window",
        )

    return (
        input_tokens,
        output_tokens,
        requested_context,
    )


# ---------------------------------------------------------------------------
# MAIN POLICY
# ---------------------------------------------------------------------------

async def inspect(
    data: dict[str, Any],
    user_api_key_dict: Any = None,
    call_type: str | None = None,
) -> dict[str, Any]:

    identity = _caller_id(
        user_api_key_dict
    )

    logger.info(
        "SECURITY_INSPECT request "
        "caller=%s "
        "model=%s "
        "call_type=%s",
        identity,
        data.get("model"),
        call_type,
    )

    # ---------------------------------------------------------
    # RATE LIMIT
    # ---------------------------------------------------------

    await _acquire_rate_limit()

    # ---------------------------------------------------------
    # REQUEST CONCURRENCY
    # ---------------------------------------------------------

    # El semaphore evita que muchas inspecciones pesadas
    # ejecuten simultáneamente.
    if (
        _concurrency_semaphore.locked()
    ):

        logger.info(
            "SECURITY_INSPECT "
            "concurrency_limit_wait"
        )

    await _concurrency_semaphore.acquire()

    try:

        # -----------------------------------------------------
        # STRUCTURAL VALIDATION
        # -----------------------------------------------------

        _check_size(data)

        _check_shape(data)

        _check_model(data)

        _check_messages(
            data["messages"]
        )

        _check_tools(data)

        _check_parameters(data)

        # -----------------------------------------------------
        # CONTEXT
        # -----------------------------------------------------

        (
            input_tokens,
            output_tokens,
            requested_context,
        ) = await _check_context(
            data
        )

        logger.info(
            "SECURITY_INSPECT ALLOW "
            "caller=%s "
            "model=%s "
            "input_tokens=%d "
            "output_tokens=%d "
            "requested_context=%d",
            identity,
            data.get("model"),
            input_tokens,
            output_tokens,
            requested_context,
        )

        return data

    finally:

        _concurrency_semaphore.release()


# ---------------------------------------------------------------------------
# LITELLM HOOK
# ---------------------------------------------------------------------------

class SecurityInspectHook(
    CustomLogger
):

    def __init__(self):

        super().__init__()

        print(
            "### SECURITY_INSPECT INSTANCE CREATED ###",
            flush=True,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: dict | None = None,
        call_type: str = "completion",
        **kwargs: Any,
    ):

        print(
            "### SECURITY_INSPECT PRE_CALL ###",
            flush=True,
        )

        if data is None:
            data = {}

        # -----------------------------------------------------
        # SOLO GENERACIÓN DE TEXTO
        # -----------------------------------------------------

        if call_type not in {
            "completion",
            "text_completion",
        }:

            return data

        identity = _caller_id(
            user_api_key_dict
        )

        try:

            return await inspect(
                data=data,
                user_api_key_dict=(
                    user_api_key_dict
                ),
                call_type=call_type,
            )

        except HTTPException:

            raise

        except Exception:

            logger.exception(
                "SECURITY_INSPECT "
                "FAIL_CLOSED "
                "caller=%s "
                "model=%s",
                identity,
                data.get("model"),
            )

            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": (
                            "Security inspection "
                            "is temporarily unavailable"
                        ),
                        "type": "api_error",
                        "param": None,
                        "code": (
                            "security_inspection_failed"
                        ),
                    }
                },
            )


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

security_inspect = SecurityInspectHook()


print(
    "### SECURITY_INSPECT MODULE READY ###",
    flush=True,
)
