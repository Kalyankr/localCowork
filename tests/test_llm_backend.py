"""Tests for the model-agnostic LLM backend interface."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.llm.backend import LLMBackend
from agent.llm.client import get_backend, set_backend
from agent.llm.ollama_backend import OllamaBackend


class _FakeBackend(LLMBackend):
    """Minimal backend for testing the backend swap mechanism."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, force_json: bool = False) -> str:
        self.calls.append("generate")
        return "fake-response"

    async def generate_async(self, prompt: str, force_json: bool = False) -> str:
        self.calls.append("generate_async")
        return "fake-async-response"

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        self.calls.append("chat")
        return "fake-chat"

    async def chat_async(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        self.calls.append("chat_async")
        return "fake-chat-async"

    async def generate_stream_async(
        self, prompt: str, force_json: bool = False
    ) -> AsyncIterator[str]:
        self.calls.append("generate_stream_async")
        yield "chunk"

    async def chat_stream_async(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> AsyncIterator[str]:
        self.calls.append("chat_stream_async")
        yield "chunk"

    def chat_stream(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> Any:
        self.calls.append("chat_stream")
        yield "chunk"

    def list_models(self) -> list[str]:
        self.calls.append("list_models")
        return ["fake-model"]

    def check_model_exists(self, model_name: str | None = None) -> bool:
        self.calls.append("check_model_exists")
        return True

    def check_health(self) -> tuple[bool, str | None]:
        self.calls.append("check_health")
        return True, None


class TestBackendSwap:
    """Verify set_backend / get_backend work, and all client functions delegate."""

    def setup_method(self):
        """Save the original backend and install a fake."""
        import agent.llm.client as mod

        self._original = mod._backend
        self.fake = _FakeBackend()
        set_backend(self.fake)

    def teardown_method(self):
        """Restore the original backend."""
        import agent.llm.client as mod

        mod._backend = self._original

    def test_get_backend_returns_fake(self):
        assert get_backend() is self.fake

    def test_call_llm_delegates(self):
        from agent.llm.client import call_llm

        assert call_llm("prompt") == "fake-response"
        assert "generate" in self.fake.calls

    def test_call_llm_chat_delegates(self):
        from agent.llm.client import call_llm_chat

        assert call_llm_chat([{"role": "user", "content": "hi"}]) == "fake-chat"
        assert "chat" in self.fake.calls

    def test_list_models_delegates(self):
        from agent.llm.client import list_models

        assert list_models() == ["fake-model"]
        assert "list_models" in self.fake.calls

    def test_check_model_exists_delegates(self):
        from agent.llm.client import check_model_exists

        assert check_model_exists("x") is True
        assert "check_model_exists" in self.fake.calls

    def test_check_ollama_health_delegates(self):
        from agent.llm.client import check_ollama_health

        ok, err = check_ollama_health()
        assert ok is True and err is None
        assert "check_health" in self.fake.calls

    @pytest.mark.asyncio
    async def test_call_llm_async_delegates(self):
        from agent.llm.client import call_llm_async

        result = await call_llm_async("prompt")
        assert result == "fake-async-response"
        assert "generate_async" in self.fake.calls

    @pytest.mark.asyncio
    async def test_call_llm_chat_async_delegates(self):
        from agent.llm.client import call_llm_chat_async

        result = await call_llm_chat_async([{"role": "user", "content": "hi"}])
        assert result == "fake-chat-async"
        assert "chat_async" in self.fake.calls

    def test_call_llm_chat_stream_delegates(self):
        from agent.llm.client import call_llm_chat_stream

        chunks = list(call_llm_chat_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["chunk"]
        assert "chat_stream" in self.fake.calls


class TestDefaultBackend:
    """get_backend() should return OllamaBackend by default."""

    def test_default_is_ollama(self):
        import agent.llm.client as mod

        original = mod._backend
        mod._backend = None
        try:
            backend = get_backend()
            assert isinstance(backend, OllamaBackend)
        finally:
            mod._backend = original


class TestOllamaBackendHealth:
    """OllamaBackend.check_health delegates to Ollama client."""

    @patch("agent.llm.ollama_backend._get_client")
    def test_check_health_ok(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        mock_get_client.return_value = mock_client
        ok, err = OllamaBackend().check_health()
        assert ok is True and err is None

    @patch("agent.llm.ollama_backend._get_client")
    def test_check_health_fail(self, mock_get_client):
        from ollama import RequestError

        mock_client = MagicMock()
        mock_client.list.side_effect = RequestError("refused")
        mock_get_client.return_value = mock_client
        ok, err = OllamaBackend().check_health()
        assert ok is False and err is not None
