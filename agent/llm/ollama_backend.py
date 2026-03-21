"""Ollama LLM backend implementation.

Wraps the official ``ollama`` Python library.  This is the default
backend used by ``agent.llm.client``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import ollama
import structlog
from ollama import AsyncClient, RequestError, ResponseError

from agent.config import get_settings
from agent.llm.backend import LLMBackend

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton clients (connection-pooled)
# ---------------------------------------------------------------------------

_client: ollama.Client | None = None
_async_client: AsyncClient | None = None
_async_client_loop: asyncio.AbstractEventLoop | None = None


def _get_host() -> str:
    s = get_settings()
    host = s.ollama_url.replace("/api/generate", "").replace("/api/chat", "")
    return host.rstrip("/")


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        s = get_settings()
        _client = ollama.Client(host=_get_host(), timeout=s.ollama_timeout)
    return _client


def _get_async_client() -> AsyncClient:
    global _async_client, _async_client_loop

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _async_client is None or _async_client_loop is not current_loop:
        s = get_settings()
        _async_client = AsyncClient(host=_get_host(), timeout=s.ollama_timeout)
        _async_client_loop = current_loop
        logger.debug("Created new AsyncClient for current event loop")

    return _async_client


class LLMError(Exception):
    """Custom exception for LLM-related errors."""


class OllamaBackend(LLMBackend):
    """Ollama-backed LLM implementation."""

    # -- synchronous ---------------------------------------------------------

    def generate(self, prompt: str, force_json: bool = False) -> str:
        try:
            client = _get_client()
            s = get_settings()
            kwargs: dict[str, Any] = {
                "model": s.ollama_model,
                "prompt": prompt,
                "options": {"num_predict": s.max_tokens, "num_ctx": s.num_ctx},
            }
            if force_json:
                kwargs["format"] = "json"
            response = client.generate(**kwargs)
            return response.response
        except RequestError as e:
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except ResponseError as e:
            raise LLMError(f"Ollama error: {e}")
        except Exception as e:
            raise LLMError(f"LLM request failed: {e}")

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        try:
            client = _get_client()
            s = get_settings()
            active_model = model or s.ollama_model
            response = client.chat(
                model=active_model,
                messages=messages,
                options={"num_predict": s.max_tokens, "num_ctx": s.num_ctx},
            )
            return response.message.content
        except RequestError as e:
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except ResponseError as e:
            raise LLMError(f"Ollama error: {e}")
        except Exception as e:
            raise LLMError(f"LLM chat request failed: {e}") from e

    def chat_stream(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> Any:
        try:
            client = _get_client()
            s = get_settings()
            active_model = model or s.ollama_model
            stream = client.chat(
                model=active_model,
                messages=messages,
                options={"num_predict": s.max_tokens, "num_ctx": s.num_ctx},
                stream=True,
            )
            for chunk in stream:
                if chunk.message and chunk.message.content:
                    yield chunk.message.content
        except RequestError as e:
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except ResponseError as e:
            raise LLMError(f"Ollama error: {e}")
        except Exception as e:
            raise LLMError(f"LLM stream request failed: {e}")

    def list_models(self) -> list[str]:
        try:
            client = _get_client()
            response = client.list()
            return [model.model for model in response.models]
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def check_model_exists(self, model_name: str | None = None) -> bool:
        model = model_name or get_settings().ollama_model
        try:
            models = self.list_models()
            return any(
                m == model or m.split(":")[0] == model.split(":")[0] for m in models
            )
        except Exception:
            return False

    def check_health(self) -> tuple[bool, str | None]:
        try:
            client = _get_client()
            client.list()
            return True, None
        except RequestError as e:
            return False, f"Connection refused. Is Ollama running? ({e})"
        except ResponseError as e:
            return False, f"Ollama error: {e}"
        except Exception as e:
            return False, f"Unknown error: {e}"

    # -- asynchronous --------------------------------------------------------

    async def generate_async(self, prompt: str, force_json: bool = False) -> str:
        try:
            client = _get_async_client()
            s = get_settings()
            kwargs: dict[str, Any] = {
                "model": s.ollama_model,
                "prompt": prompt,
                "options": {"num_predict": s.max_tokens, "num_ctx": s.num_ctx},
            }
            if force_json:
                kwargs["format"] = "json"
            response = await client.generate(**kwargs)
            return response.response
        except RequestError as e:
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except ResponseError as e:
            raise LLMError(f"Ollama error: {e}")
        except TimeoutError:
            s = get_settings()
            raise LLMError(
                f"Request timed out. The model may be slow or overloaded. "
                f"Try increasing LOCALCOWORK_OLLAMA_TIMEOUT (current: {s.ollama_timeout}s)"
            )
        except ConnectionError as e:
            raise LLMError(
                f"Connection lost to Ollama. Check if Ollama is still running. Error: {e}"
            )
        except Exception as e:
            s = get_settings()
            error_str = str(e).lower()
            if "timeout" in error_str:
                raise LLMError(
                    f"Request timed out after {s.ollama_timeout}s. "
                    f"Model '{s.ollama_model}' may be slow. Try a smaller model or increase timeout."
                )
            elif "connection" in error_str or "refused" in error_str:
                raise LLMError(
                    f"Cannot connect to Ollama at {s.ollama_url}. Is Ollama running? "
                    f"Start with: ollama serve"
                )
            elif "memory" in error_str or "oom" in error_str:
                raise LLMError(
                    f"Out of memory loading model '{s.ollama_model}'. "
                    f"Try a smaller model like 'mistral' or 'llama3.2:3b'"
                )
            raise LLMError(f"LLM request failed: {e}")

    async def chat_async(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        try:
            client = _get_async_client()
            s = get_settings()
            active_model = model or s.ollama_model
            response = await client.chat(
                model=active_model,
                messages=messages,
                options={"num_predict": s.max_tokens, "num_ctx": s.num_ctx},
            )
            return response.message.content
        except RequestError as e:
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except ResponseError as e:
            raise LLMError(f"Ollama error: {e}")
        except Exception as e:
            raise LLMError(f"Async LLM chat request failed: {e}") from e

    async def generate_stream_async(
        self, prompt: str, force_json: bool = False
    ) -> AsyncIterator[str]:
        try:
            client = _get_async_client()
            s = get_settings()
            kwargs: dict[str, Any] = {
                "model": s.ollama_model,
                "prompt": prompt,
                "options": {"num_predict": s.max_tokens, "num_ctx": s.num_ctx},
                "stream": True,
            }
            if force_json:
                kwargs["format"] = "json"
            async for chunk in await client.generate(**kwargs):
                if chunk.response:
                    yield chunk.response
        except RequestError as e:
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except ResponseError as e:
            raise LLMError(f"Ollama error: {e}")
        except Exception as e:
            raise LLMError(f"Async stream request failed: {e}")

    async def chat_stream_async(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> AsyncIterator[str]:
        try:
            client = _get_async_client()
            s = get_settings()
            active_model = model or s.ollama_model
            async for chunk in await client.chat(
                model=active_model,
                messages=messages,
                options={"num_predict": s.max_tokens, "num_ctx": s.num_ctx},
                stream=True,
            ):
                if chunk.message and chunk.message.content:
                    yield chunk.message.content
        except RequestError as e:
            raise LLMError(f"Cannot connect to Ollama. Is it running? Error: {e}")
        except ResponseError as e:
            raise LLMError(f"Ollama error: {e}")
        except Exception as e:
            raise LLMError(f"Async chat stream request failed: {e}")
