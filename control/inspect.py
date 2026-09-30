from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger


# ==========================================================================
# LOGGING
# ==========================================================================

# Dedicated logger for this policy layer.
#
# Keeping a dedicated logger makes it possible to filter security policy
# decisions independently from the rest of LiteLLM logs.
logger = logging.getLogger("litellm.security.inspect")


# ==========================================================================
# PLATFORM POLICY
# ==========================================================================

# Models that this security layer is allowed to serve.
#
# This is intentionally NOT delegated to generic LiteLLM model routing.
# LiteLLM answers "can I route to this model?" while this policy answers
# "is this model part of the security policy of this application?"
ALLOWED_MODELS = {
    "qwen-coding-local",
}


# Roles accepted by this coding-agent platform.
#
# This is a platform-level contract rather than a generic syntax check.
#
# If the product later decides to support additional roles, this is the
# explicit policy location where they should be added.
ALLOWED_ROLES = {
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
}


# The current H100/SGLang fleet is configured as text-only.
#
# Therefore image/audio/multimodal message content is rejected here.
#
# This is intentionally kept in the custom policy because it describes
# the capabilities exposed by THIS product, not a generic OpenAI rule.
ALLOW_MULTIMODAL = False


# Maximum output requested by a client.
#
# The model has a larger theoretical context window, but this platform
# deliberately limits generated output to 4096 tokens.
#
# This is an application/product policy and therefore belongs here.
MAX_OUTPUT_TOKENS = 4096


# ==========================================================================
# TOOL POLICY
# ==========================================================================

# Maximum number of tools accepted in a single request.
#
# This is a defensive platform limit rather than a context-window check.
MAX_TOOLS = 64


# Maximum approximate size of one tool definition.
#
# This protects the gateway from unusually large tool schemas.
#
# This is deliberately NOT used to calculate total HTTP request size.
MAX_TOOL_SCHEMA_BYTES = 64 * 1024


# ==========================================================================
# IDENTITY
# ==========================================================================

def _caller_id(user_api_key_dict: Any) -> str:
    """
    Create an anonymous identifier for security logs.

    The actual API key is never written to the logs.

    The identifier is derived from the first available LiteLLM identity
    field. Hashing keeps logs useful for correlation while avoiding
    exposing the original identity/token value.
    """
    if user_api_key_dict is None:
        return "anonymous"

    # Try stable LiteLLM identity fields in priority order.
    #
    # user_id/team_id are preferable when available.
    # token is used only as a final fallback.
    for attr in ("user_id", "team_id", "token","tenant_id"):
        try:
            value = getattr(user_api_key_dict, attr, None)
        except Exception:
            value = None

        if value:
            return hashlib.sha256(str(value).encode()).hexdigest()[:12]

    return "anonymous"


# ==========================================================================
# REJECTION
# ==========================================================================

def _reject(status: int, code: str, message: str) -> None:
    """
    Reject a request using the same general error structure expected by
    the OpenAI-compatible API.

    This function centralizes policy rejection so that every security
    decision has:
      - HTTP status
      - machine-readable error code
      - human-readable message

    The function always raises and therefore never returns normally.
    """
    logger.warning(
        "SECURITY_INSPECT REJECT status=%s reason=%s message=%s",
        status,
        code,
        message,
    )

    # 5xx means the security layer itself failed.
    #
    # 4xx means the client violated an explicit platform policy.
    error_type = "api_error" if status >= 500 else "invalid_request_error"

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


# ==========================================================================
# MODEL POLICY
# ==========================================================================

def _check_model(data: dict[str, Any]) -> None:
    """
    Enforce the set of models exposed by this application.

    LiteLLM can route models, but this check expresses an explicit
    application security boundary.

    A request for an unknown model is rejected instead of allowing the
    custom hook to silently operate on a model outside its policy.
    """
    model = data.get("model")

    if model not in ALLOWED_MODELS:
        _reject(
            403,
            "model_not_allowed",
            f"Requested model '{model}' is not allowed",
        )


# ==========================================================================
# MESSAGE POLICY
# ==========================================================================

def _check_messages(data: dict[str, Any]) -> None:
    """
    Enforce platform-specific message policies.

    This function intentionally does NOT reimplement LiteLLM's generic
    request validation.

    It only checks policies that are specific to this fleet:
      - allowed roles
      - whether multimodal content is enabled
    """
    messages = data.get("messages")

    # If LiteLLM has already normalized/validated the request and messages
    # are absent here, do not create a second generic validation layer.
    if not isinstance(messages, list):
        return

    for index, message in enumerate(messages):
        # Do not duplicate full OpenAI schema validation here.
        #
        # If LiteLLM has accepted the request, this hook focuses only on
        # the security policy fields it actually cares about.
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        # Role restrictions are an application policy.
        if role not in ALLOWED_ROLES:
            _reject(
                400,
                "invalid_role",
                f"Unsupported message role at index {index}",
            )

        content = message.get("content")

        if content is None:
            continue

        # A list-valued content field represents structured/multimodal
        # content in OpenAI-compatible chat requests.
        #
        # The current fleet is text-only, so reject it explicitly.
        if isinstance(content, list):
            if not ALLOW_MULTIMODAL:
                _reject(
                    400,
                    "multimodal_not_allowed",
                    "Multimodal content is not enabled",
                )


# ==========================================================================
# TOOL SIZE ESTIMATION
# ==========================================================================

def _estimate_object_bytes(value: Any, depth: int = 0) -> int:
    """
    Estimate the memory/serialized size of a tool definition.

    This function is intentionally limited in scope.

    It is NOT:
      - an HTTP body-size calculator
      - a JSON serializer
      - a general LiteLLM request validator

    It exists only to enforce the platform's maximum tool-schema size.

    A bounded recursive traversal is used so a malformed structure cannot
    cause an unbounded traversal.
    """
    # Protect this helper from pathological nested structures.
    if depth > 12:
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
        return 2 + sum(
            len(str(key).encode("utf-8")) + _estimate_object_bytes(item, depth + 1)
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple)):
        return 2 + sum(_estimate_object_bytes(item, depth + 1) for item in value)

    # LiteLLM/internal objects are intentionally not recursively inspected.
    #
    # The security hook only needs a conservative bounded estimate.
    return 64


# ==========================================================================
# TOOL POLICY
# ==========================================================================

def _check_tools(data: dict[str, Any]) -> None:
    """
    Apply platform-specific limits to tool calling.

    LiteLLM remains responsible for generic request processing.

    This function only prevents excessive tool counts or excessively large
    individual tool schemas from reaching the model backend.
    """
    tools = data.get("tools")

    # Tool calling is optional.
    if tools is None:
        return

    if not isinstance(tools, list):
        _reject(
            400,
            "invalid_tools",
            "tools must be an array",
        )

    # Explicit application-level tool count limit.
    if len(tools) > MAX_TOOLS:
        _reject(
            413,
            "too_many_tools",
            "Too many tools",
        )

    for index, tool in enumerate(tools):
        # Do not perform a complete JSON Schema validation here.
        # LiteLLM/provider validation should own generic schema handling.
        if not isinstance(tool, dict):
            _reject(
                400,
                "invalid_tool",
                f"Invalid tool definition at index {index}",
            )

        # Only enforce the platform-specific size boundary.
        tool_size = _estimate_object_bytes(tool)

        if tool_size > MAX_TOOL_SCHEMA_BYTES:
            _reject(
                413,
                "tool_schema_too_large",
                "Tool schema is too large",
            )


# ==========================================================================
# GENERATION POLICY
# ==========================================================================

def _check_generation(data: dict[str, Any]) -> None:
    """
    Enforce the platform's output-token policy.

    LiteLLM/model metadata handles context-window capacity.

    This function only answers:

        "Is this client allowed to request more than 4096 output tokens?"

    That distinction is important because the model's 65K context window
    and the product's 4K generation limit are different concepts.
    """
    # OpenAI-compatible clients may use either field depending on the API
    # surface/version they target.
    value = data.get("max_completion_tokens")

    if value is None:
        value = data.get("max_tokens")

    # No explicit value means the normal LiteLLM/model default applies.
    if value is None:
        return

    # This is a type/policy check because we need a numeric value before
    # comparing it against the platform maximum.
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(
            400,
            "invalid_max_tokens",
            "max_tokens must be an integer",
        )

    if value <= 0:
        _reject(
            400,
            "invalid_max_tokens",
            "max_tokens must be greater than zero",
        )

    # This is the actual platform-specific policy.
    if value > MAX_OUTPUT_TOKENS:
        _reject(
            400,
            "max_tokens_exceeded",
            f"Maximum output is {MAX_OUTPUT_TOKENS} tokens",
        )


# ==========================================================================
# MAIN SECURITY INSPECTION
# ==========================================================================

async def inspect(
    data: dict[str, Any],
    user_api_key_dict: Any = None,
    call_type: str | None = None,
) -> dict[str, Any]:
    """
    Execute the platform-specific security policy.

    The function is deliberately lightweight.

    It does NOT:
      - implement rate limiting
      - implement concurrency control
      - tokenize the request
      - estimate the context window
      - implement generic HTTP body limits
      - retry requests
      - perform model routing

    Those responsibilities belong to LiteLLM, the ingress layer, or the
    future admission-control layer.

    The function returns the original data unchanged when the request is
    allowed. This makes the hook composable with future layers such as:

        security_inspect
            ->
        tenant
            ->
        should_shed
            ->
        LiteLLM router
            ->
        SGLang
    """
    identity = _caller_id(user_api_key_dict)

    logger.info(
        "SECURITY_INSPECT request caller=%s model=%s call_type=%s",
        identity,
        data.get("model"),
        call_type,
    )

    # ----------------------------------------------------------------------
    # Model authorization
    # ----------------------------------------------------------------------
    # Security boundary: only models explicitly exposed by this application.
    _check_model(data)

    # ----------------------------------------------------------------------
    # Message policy
    # ----------------------------------------------------------------------
    # Application-specific role and multimodal restrictions.
    _check_messages(data)

    # ----------------------------------------------------------------------
    # Tool policy
    # ----------------------------------------------------------------------
    # Protect the gateway/model from excessive tool definitions.
    _check_tools(data)

    # ----------------------------------------------------------------------
    # Generation policy
    # ----------------------------------------------------------------------
    # Enforce the product's 4096-token output ceiling.
    _check_generation(data)

    logger.info(
        "SECURITY_INSPECT ALLOW caller=%s model=%s",
        identity,
        data.get("model"),
    )

    # No mutation is performed here.
    #
    # The tenant layer is the appropriate place to enrich the request.
    return data


# ==========================================================================
# LITELLM HOOK
# ==========================================================================

class SecurityInspectHook(CustomLogger):
    """
    LiteLLM callback adapter.

    LiteLLM invokes async_pre_call_hook before the model call.

    The class itself contains almost no policy logic. It is intentionally
    just an adapter between LiteLLM's callback lifecycle and inspect().
    """

    def __init__(self):
        super().__init__()
        print("### SECURITY_INSPECT INSTANCE CREATED ###", flush=True)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: dict | None = None,
        call_type: str = "completion",
        **kwargs: Any,
    ):
        """
        Execute security policy before the model call.

        Only completion/text_completion requests are inspected by this
        policy.

        Unsupported call types are returned untouched so this callback
        does not accidentally become a generic blocker for unrelated
        LiteLLM endpoints.
        """
        if data is None:
            data = {}

        # This policy is designed for the coding-agent text-generation path.
        #
        # Other LiteLLM operations should not accidentally inherit these
        # model/message policies.
        if call_type not in {"completion", "text_completion"}:
            return data

        identity = _caller_id(user_api_key_dict)

        try:
            return await inspect(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
            )
        except HTTPException:
            # Policy rejections are already represented by a deliberate
            # HTTPException. Preserve the original status/code.
            raise
        except Exception:
            # Unexpected failure inside a security control is fail-closed.
            #
            # We do NOT allow a broken security layer to accidentally turn
            # into an allow decision.
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
            )


# ==========================================================================
# EXPORT
# ==========================================================================

# LiteLLM loads this object through:
#
#     security_inspect.security_inspect
#
# Therefore the exported variable must have this exact name.
security_inspect = SecurityInspectHook()


print(
    "### SECURITY_INSPECT MODULE READY ###",
    flush=True,
)
