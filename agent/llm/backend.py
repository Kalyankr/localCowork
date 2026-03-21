"""Model-agnostic LLM backend interface.

Defines the abstract base class that all LLM backends must implement.
The default backend is Ollama (see ``ollama_backend.py``).

To add a new backend:
    1. Subclass ``LLMBackend``
    2. Implement all abstract methods
    3. Call ``agent.llm.client.set_backend(YourBackend())``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMBackend(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, force_json: bool = False) -> str:
        """Synchronous text generation."""
        ...

    @abstractmethod
    async def generate_async(self, prompt: str, force_json: bool = False) -> str:
        """Asynchronous text generation."""
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Synchronous chat completion."""
        ...

    @abstractmethod
    async def chat_async(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Asynchronous chat completion."""
        ...

    @abstractmethod
    def generate_stream_async(
        self, prompt: str, force_json: bool = False
    ) -> AsyncIterator[str]:
        """Async streaming text generation."""
        ...

    @abstractmethod
    def chat_stream_async(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> AsyncIterator[str]:
        """Async streaming chat completion."""
        ...

    @abstractmethod
    def chat_stream(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> Any:
        """Synchronous streaming chat (yields str chunks)."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model names."""
        ...

    @abstractmethod
    def check_model_exists(self, model_name: str | None = None) -> bool:
        """Check whether a model is available."""
        ...

    @abstractmethod
    def check_health(self) -> tuple[bool, str | None]:
        """Health check. Returns (is_healthy, error_message_or_none)."""
        ...
