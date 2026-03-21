"""LLM client — model-agnostic facade.

All public functions delegate to the active ``LLMBackend`` (default:
``OllamaBackend``).  To switch backends at runtime call
``set_backend()``.  Existing imports (``call_llm``, ``LLMError``, etc.)
continue to work unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import structlog

from agent.config import get_settings
from agent.llm.backend import LLMBackend
from agent.llm.ollama_backend import LLMError, OllamaBackend

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------

_backend: LLMBackend | None = None


def get_backend() -> LLMBackend:
    """Return the active LLM backend (creates Ollama default on first call)."""
    global _backend
    if _backend is None:
        _backend = OllamaBackend()
    return _backend


def set_backend(backend: LLMBackend) -> None:
    """Replace the active LLM backend."""
    global _backend
    _backend = backend


# Re-export LLMError so existing ``from agent.llm.client import LLMError`` works
__all__ = [
    "LLMError",
    "get_backend",
    "set_backend",
    "call_llm",
    "call_llm_chat",
    "call_llm_json",
    "call_llm_async",
    "call_llm_chat_async",
    "call_llm_json_async",
    "call_llm_stream_async",
    "call_llm_chat_stream_async",
    "call_llm_chat_stream",
    "repair_json",
    "list_models",
    "check_model_exists",
    "check_ollama_health",
]


# =============================================================================
# Synchronous functions
# =============================================================================


def call_llm(prompt: str, force_json: bool = False) -> str:
    """Call the LLM and return raw text."""
    return get_backend().generate(prompt, force_json=force_json)


def call_llm_chat(messages: list[dict[str, str]], model: str | None = None) -> str:
    """Call the LLM with chat messages."""
    return get_backend().chat(messages, model=model)


def call_llm_json(prompt: str) -> dict[str, Any]:
    """
    Call the LLM and guarantee valid JSON output.
    Retries with repair logic if JSON parsing still fails.

    Args:
        prompt: The initial prompt to send

    Returns:
        Parsed JSON as a dictionary

    Raises:
        LLMError: If JSON parsing fails after all retries
    """
    s = get_settings()
    max_retries = s.max_json_retries

    for attempt in range(max_retries + 1):
        try:
            current_prompt = prompt
            if attempt > 0:
                logger.info(f"JSON retry attempt {attempt + 1}/{max_retries + 1}")
                current_prompt = (
                    prompt
                    + "\n\nREMINDER: Output ONLY valid JSON. No markdown, no code blocks, no explanation. Start with { and end with }."
                )

            # Use Ollama's native JSON mode for reliable output
            raw = call_llm(current_prompt, force_json=True)

            # Try direct parse first
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Try repair as fallback
                return repair_json(raw)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            else:
                raise LLMError(
                    f"Failed to get valid JSON after {max_retries + 1} attempts. "
                    "The AI model may be having trouble understanding the request. "
                    "Please try rephrasing."
                )


def repair_json(text: str) -> dict[str, Any]:
    """
    Attempts to fix common LLM JSON errors:
    - Literal newlines within string values
    - Python-style string concatenation
    - Improperly quoted values
    - Missing/extra braces/brackets
    - Markdown code blocks
    """

    # 0. Remove markdown code blocks if present
    if "```json" in text:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    elif "```" in text:
        text = re.sub(r"```\w*\s*", "", text)
        text = re.sub(r"```\s*", "", text)

    # 1. Extract the cleanest JSON-like block
    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("No JSON object found in response")

    # Use a simple stack-based approach to find the end of the object
    stack = 0
    end_idx = -1
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == "{":
                stack += 1
            elif char == "}":
                stack -= 1
                if stack == 0:
                    end_idx = i + 1
                    break

        escape = False

    if end_idx == -1:
        # Try to find a closing brace anyway
        last_brace = text.rfind("}")
        if last_brace > start_idx:
            end_idx = last_brace + 1
        else:
            json_like = text[start_idx:] + "}"
            end_idx = len(json_like) + start_idx

    json_like = text[start_idx:end_idx] if end_idx != -1 else text[start_idx:]

    # 2. Fix literal newlines inside string values
    new_json = ""
    in_string = False
    escape = False
    for char in json_like:
        if char == '"' and not escape:
            in_string = not in_string
            new_json += char
        elif char == "\\" and in_string and not escape:
            escape = True
            new_json += char
        elif char == "\n" and in_string:
            new_json += "\\n"
        elif char == "\t" and in_string:
            new_json += "\\t"
        else:
            new_json += char
            escape = False
    json_like = new_json

    # 3. Fix common syntax errors
    json_like = re.sub(r",\s*}", "}", json_like)  # Trailing comma before }
    json_like = re.sub(r",\s*]", "]", json_like)  # Trailing comma before ]
    json_like = re.sub(r"'\s*:", '":', json_like)  # Single quotes for keys
    json_like = re.sub(
        r":\s*'([^']*)'", r': "\1"', json_like
    )  # Single quotes for values

    try:
        return json.loads(json_like)
    except json.JSONDecodeError:
        pass

    # 4. Try to fix unquoted values
    try:

        def quote_val(m):
            val = m.group(2).strip()
            if not (
                val.startswith('"')
                or val.startswith("'")
                or val.startswith("[")
                or val.startswith("{")
                or val.isdigit()
                or val in ["true", "false", "null"]
            ):
                return f'"{m.group(1)}": "{val}"'
            return m.group(0)

        fixed = re.sub(r'"(\w+)":\s*([^,\}\]\n]+)', quote_val, json_like)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5. Last resort - try to extract just the steps array
    try:
        steps_match = re.search(r'"steps"\s*:\s*\[(.*?)\]', json_like, re.DOTALL)
        if steps_match:
            return {"steps": json.loads(f"[{steps_match.group(1)}]")}
    except (json.JSONDecodeError, ValueError):
        pass

    raise ValueError("Could not parse response as JSON")


def list_models() -> list[str]:
    """List available models."""
    return get_backend().list_models()


def check_model_exists(model_name: str | None = None) -> bool:
    """Check if a model is available."""
    return get_backend().check_model_exists(model_name)


def check_ollama_health() -> tuple[bool, str | None]:
    """Check if the LLM backend is healthy."""
    return get_backend().check_health()


# =============================================================================
# Async Functions
# =============================================================================


async def call_llm_async(prompt: str, force_json: bool = False) -> str:
    """Async version of call_llm."""
    return await get_backend().generate_async(prompt, force_json=force_json)


async def call_llm_chat_async(
    messages: list[dict[str, str]], model: str | None = None
) -> str:
    """Async version of call_llm_chat."""
    return await get_backend().chat_async(messages, model=model)


async def call_llm_json_async(prompt: str) -> dict[str, Any]:
    """Async version of call_llm_json. Guarantees valid JSON output."""
    s = get_settings()
    max_retries = s.max_json_retries

    for attempt in range(max_retries + 1):
        try:
            current_prompt = prompt
            if attempt > 0:
                logger.info(f"Async JSON retry attempt {attempt + 1}/{max_retries + 1}")
                current_prompt = (
                    prompt
                    + "\n\nREMINDER: Output ONLY valid JSON. No markdown, no code blocks, "
                    "no explanation. Start with { and end with }."
                )

            raw = await call_llm_async(current_prompt, force_json=True)

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return repair_json(raw)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Async JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            else:
                raise LLMError(
                    f"Failed to get valid JSON after {max_retries + 1} attempts. "
                    "The AI model may be having trouble understanding the request. "
                    "Please try rephrasing."
                )
    # Unreachable, but keeps mypy happy
    raise LLMError("JSON parsing exhausted all retries")  # pragma: no cover


async def call_llm_stream_async(
    prompt: str, force_json: bool = False
) -> AsyncIterator[str]:
    """Async streaming text generation."""
    async for chunk in get_backend().generate_stream_async(
        prompt, force_json=force_json
    ):
        yield chunk


async def call_llm_chat_stream_async(
    messages: list[dict[str, str]], model: str | None = None
) -> AsyncIterator[str]:
    """Async streaming chat."""
    async for chunk in get_backend().chat_stream_async(messages, model=model):
        yield chunk


def call_llm_chat_stream(messages: list[dict[str, str]], model: str | None = None):
    """Synchronous streaming chat."""
    yield from get_backend().chat_stream(messages, model=model)
