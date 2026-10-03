"""Guard: platform policy applied before admission (first callback after auth).

Product rules only. LiteLLM keeps the generic work (auth, context-window check,
routing, concurrency limits); admission.py keeps load-based shedding.
Fail-closed: an unexpected error here rejects the request.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger #type: ignore

logger = logging.getLogger("litellm.security.inspect")

# --- Policy ----------------------------------------------------------------
ALLOWED_MODELS = {"qwen-coding-local"}   # only the public alias; not delegated to LiteLLM routing
ALLOWED_ROLES = {"system", "developer", "user", "assistant", "tool"}
# Qwen3.5-9B is multimodal (its vision tower is loaded), so images are accepted, with limits. Audio, video and
# files are not. Text-only content given as a list of parts ([{"type": "text", ...}]) is always accepted.
ALLOW_MULTIMODAL = os.getenv("ALLOW_MULTIMODAL", "1") == "1"     # images in user/tool messages
ALLOW_REMOTE_IMAGE_URLS = False          # an http(s) URL would make the engine fetch from inside the cluster (SSRF); data: URIs only
ALLOWED_IMAGE_TYPES = {"png", "jpeg", "webp", "gif"}
MAX_IMAGES_PER_REQUEST = 8
MAX_IMAGE_BYTES = 5 * 1024 * 1024        # decoded size of one image
MAX_TOTAL_IMAGE_BYTES = 16 * 1024 * 1024
IMAGE_ROLES = {"user", "tool"}
MAX_OUTPUT_TOKENS = 4096                 # product cap, distinct from the 64K context window
# What to do with a request that asks for more than the cap. pi, for example, asks for 16384 by default.
#   clamp (default, decision 54): serve it with max_tokens lowered to the cap    reject: answer 400 (the original behaviour)
MAX_TOKENS_POLICY = os.getenv("MAX_TOKENS_POLICY", "clamp")
# A caller declares a CLASS, not a rank: priority <= 5 is interactive, 6..10 is batch (lower = more urgent). Inside the
# interactive class the order is arrival order, so every interactive value is normalised to 5; otherwise any client could
# send priority=-1000 and jump the engine's queue. Out-of-range values are refused.
PRIORITY_MIN, PRIORITY_MAX, INTERACTIVE_PRIORITY = 1, 10, 5
MAX_CHOICES = 1                          # `n` > 1 would run several sequences while counting as one place
MAX_TOOLS = 64
MAX_TOOL_SCHEMA_BYTES = 64 * 1024        # per tool definition, not the HTTP body
GUARDED_CALL_TYPES = {"completion", "acompletion", "text_completion", "atext_completion"}


def _caller_id(user_api_key_dict: Any) -> str:
    """Hashed caller id for logs; the key itself is never logged."""
    if user_api_key_dict is None:
        return "anonymous"

    for attr in ("user_id", "team_id", "token", "tenant_id"):
        try:
            value = getattr(user_api_key_dict, attr, None)
        except Exception:
            value = None

        if value:
            return hashlib.sha256(str(value).encode()).hexdigest()[:12]

    return "anonymous"


def _reject(status: int, code: str, message: str) -> None:
    """Raise an OpenAI-style error; `source: inspect` marks who refused."""
    logger.warning(
        "SECURITY_INSPECT REJECT status=%s reason=%s message=%s",
        status,
        code,
        message,
    )
    raise HTTPException(
        status_code=status,
        detail={
            "error": {
                "message": message,
                "type": "api_error" if status >= 500 else "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
        headers={"source": "inspect"},
    )


def _check_model(data: dict[str, Any]) -> None:
    model = data.get("model")

    if model not in ALLOWED_MODELS:
        _reject(403, "model_not_allowed", f"Requested model '{model}' is not allowed")


_DATA_URI = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)


def _check_parts(message: dict[str, Any], index: int) -> tuple[int, int]:
    """Validates a list-valued `content`; returns (images, decoded image bytes)."""
    images = 0
    image_bytes = 0

    for part in message["content"]:
        kind = part.get("type") if isinstance(part, dict) else None

        if kind == "text":
            if not isinstance(part.get("text"), str):
                _reject(400, "invalid_content_part", f"Text part without text at message {index}")
            continue

        if kind != "image_url":
            _reject(400, "multimodal_not_allowed", f"Content part type '{kind}' is not supported (text and images only)")

        if not ALLOW_MULTIMODAL:
            _reject(400, "multimodal_not_allowed", "Images are not enabled")

        if message.get("role") not in IMAGE_ROLES:
            _reject(400, "invalid_content_part", f"Images are only accepted in {sorted(IMAGE_ROLES)} messages")

        ref = part.get("image_url")
        url = ref.get("url") if isinstance(ref, dict) else ref

        if not isinstance(url, str):
            _reject(400, "invalid_content_part", f"image_url without a url at message {index}")

        match = _DATA_URI.match(url)

        if match is None:
            if url.startswith(("http://", "https://")) and not ALLOW_REMOTE_IMAGE_URLS:
                _reject(400, "remote_image_not_allowed", "Remote image URLs are not accepted: send the image as a data: URI")
            _reject(400, "invalid_content_part", "Unsupported image reference")

        if match.group(1).lower() not in ALLOWED_IMAGE_TYPES:
            _reject(400, "unsupported_image_type", f"Image type '{match.group(1)}' is not supported")

        size = len(match.group(2)) * 3 // 4

        if size > MAX_IMAGE_BYTES:
            _reject(413, "image_too_large", f"An image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")

        images += 1
        image_bytes += size

    return images, image_bytes


def _check_messages(data: dict[str, Any]) -> None:
    messages = data.get("messages")

    if not isinstance(messages, list):   # generic validation is LiteLLM's job
        return

    total = {"images": 0, "bytes": 0}

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        if message.get("role") not in ALLOWED_ROLES:
            _reject(400, "invalid_role", f"Unsupported message role at index {index}")

        if isinstance(message.get("content"), list):
            images, image_bytes = _check_parts(message, index)
            total["images"] += images
            total["bytes"] += image_bytes

    if total["images"] > MAX_IMAGES_PER_REQUEST:
        _reject(413, "too_many_images", f"At most {MAX_IMAGES_PER_REQUEST} images per request")

    if total["bytes"] > MAX_TOTAL_IMAGE_BYTES:
        _reject(413, "images_too_large", "The images of this request are too large")


def _estimate_object_bytes(value: Any, depth: int = 0) -> int:
    """Bounded size estimate of a tool definition (depth-limited against pathological input)."""
    if depth > 12:
        return 0

    if value is None or isinstance(value, bool):
        return 4

    if isinstance(value, str):
        return len(value.encode("utf-8"))

    if isinstance(value, bytes):
        return len(value)

    if isinstance(value, (int, float)):
        return 16

    if isinstance(value, dict):
        return 2 + sum(
            len(str(key).encode("utf-8")) + _estimate_object_bytes(item, depth + 1)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple)):
        return 2 + sum(_estimate_object_bytes(item, depth + 1) for item in value)

    return 64   # internal LiteLLM objects: not inspected


def _check_tools(data: dict[str, Any]) -> None:
    tools = data.get("tools")

    if tools is None:
        return

    if not isinstance(tools, list):
        _reject(400, "invalid_tools", "tools must be an array")

    if len(tools) > MAX_TOOLS:
        _reject(413, "too_many_tools", "Too many tools")

    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            _reject(400, "invalid_tool", f"Invalid tool definition at index {index}")

        if _estimate_object_bytes(tool) > MAX_TOOL_SCHEMA_BYTES:
            _reject(413, "tool_schema_too_large", "Tool schema is too large")


def _check_generation(data: dict[str, Any]) -> None:
    for key in ("max_completion_tokens", "max_tokens"):
        value = data.get(key)

        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, int):
            _reject(400, "invalid_max_tokens", "max_tokens must be an integer")

        if value <= 0:
            _reject(400, "invalid_max_tokens", "max_tokens must be greater than zero")

        if value > MAX_OUTPUT_TOKENS:
            if MAX_TOKENS_POLICY == "clamp":
                logger.info("SECURITY_INSPECT CLAMP %s %s -> %s", key, value, MAX_OUTPUT_TOKENS)
                data[key] = MAX_OUTPUT_TOKENS
            else:
                _reject(400, "max_tokens_exceeded", f"Maximum output is {MAX_OUTPUT_TOKENS} tokens")


def _check_priority(data: dict[str, Any]) -> None:
    value = data.get("priority")

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, int) or not PRIORITY_MIN <= value <= PRIORITY_MAX:
        _reject(400, "invalid_priority", f"priority must be an integer from {PRIORITY_MIN} to {PRIORITY_MAX}")

    if value <= INTERACTIVE_PRIORITY:
        data["priority"] = INTERACTIVE_PRIORITY


def _check_choices(data: dict[str, Any]) -> None:
    for key in ("n", "best_of"):
        value = data.get(key)

        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value != MAX_CHOICES):
            _reject(400, "multiple_choices_not_supported", f"'{key}' other than {MAX_CHOICES} is not supported: each choice is another sequence")


async def inspect(
    data: dict[str, Any],
    user_api_key_dict: Any = None,
    call_type: str | None = None,
) -> dict[str, Any]:
    identity = _caller_id(user_api_key_dict)

    logger.info(
        "SECURITY_INSPECT request caller=%s model=%s call_type=%s",
        identity,
        data.get("model"),
        call_type,
    )

    _check_model(data)
    _check_messages(data)
    _check_tools(data)
    _check_generation(data)
    _check_priority(data)
    _check_choices(data)

    logger.info(
        "SECURITY_INSPECT ALLOW caller=%s model=%s",
        identity,
        data.get("model"),
    )

    return data


class SecurityInspectHook(CustomLogger):
    """LiteLLM adapter; the exported name below is what config.yaml references."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: dict | None = None,
        call_type: str = "completion",
        **kwargs: Any,
    ):
        if data is None:
            data = {}

        # Only text generation is inspected; other endpoints pass through.
        # The proxy reports async chat calls as "acompletion".
        if call_type not in GUARDED_CALL_TYPES:
            return data

        identity = _caller_id(user_api_key_dict)

        if data.get("priority") is None:
            data["priority"] = 5   # default class: interactive
            data.setdefault("litellm_params", {})["priority"] = data["priority"]

        try:
            return await inspect(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
            )
        except HTTPException:
            raise
        except Exception:
            # Fail-closed: a broken guard must not turn into an allow.
            logger.exception(
                "SECURITY_INSPECT FAIL_CLOSED caller=%s model=%s",
                identity,
                data.get("model"),
            )

            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "Security inspection is temporarily unavailable",
                        "type": "api_error",
                        "param": None,
                        "code": "security_inspection_failed",
                    }
                },
                headers={"source": "inspect"},
            )


security_inspect = SecurityInspectHook()   # referenced as security_inspect.security_inspect
