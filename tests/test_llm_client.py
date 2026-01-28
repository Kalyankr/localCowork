"""Tests for the LLM client."""

import pytest
from unittest.mock import patch, MagicMock


class TestCallLLM:
    """Tests for the call_llm function."""

    @patch("agent.llm.client._get_client")
    def test_call_llm_success(self, mock_get_client):
        """call_llm should return response text on success."""
        from agent.llm.client import call_llm

        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(response="Hello, world!")
        mock_get_client.return_value = mock_client

        result = call_llm("Say hello")

        assert result == "Hello, world!"
        mock_client.generate.assert_called_once()

    @patch("agent.llm.client._get_client")
    def test_call_llm_with_json_mode(self, mock_get_client):
        """call_llm with force_json should use json format."""
        from agent.llm.client import call_llm

        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(response='{"key": "value"}')
        mock_get_client.return_value = mock_client

        result = call_llm("Return JSON", force_json=True)

        assert result == '{"key": "value"}'
        call_args = mock_client.generate.call_args
        assert call_args.kwargs.get("format") == "json"

    @patch("agent.llm.client._get_client")
    def test_call_llm_connection_error(self, mock_get_client):
        """call_llm should raise LLMError on connection failure."""
        from agent.llm.client import call_llm, LLMError
        from ollama import RequestError

        mock_client = MagicMock()
        mock_client.generate.side_effect = RequestError("Connection refused")
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMError) as exc_info:
            call_llm("Test prompt")

        assert "Cannot connect to Ollama" in str(exc_info.value)


class TestCallLLMJSON:
    """Tests for the call_llm_json function."""

    @patch("agent.llm.client.call_llm")
    def test_call_llm_json_valid_response(self, mock_call_llm):
        """call_llm_json should parse valid JSON."""
        from agent.llm.client import call_llm_json

        mock_call_llm.return_value = '{"thought": "test", "is_complete": true}'

        result = call_llm_json("Return some JSON")

        assert result["thought"] == "test"
        assert result["is_complete"] is True

    @patch("agent.llm.client.call_llm")
    def test_call_llm_json_retries_on_failure(self, mock_call_llm):
        """call_llm_json should retry on JSON parse failure."""
        from agent.llm.client import call_llm_json

        # First call returns invalid JSON, second returns valid
        mock_call_llm.side_effect = ["This is not JSON", '{"result": "success"}']

        result = call_llm_json("Return JSON please")

        assert result["result"] == "success"
        assert mock_call_llm.call_count == 2

    @patch("agent.llm.client.call_llm")
    def test_call_llm_json_raises_after_max_retries(self, mock_call_llm):
        """call_llm_json should raise LLMError after max retries."""
        from agent.llm.client import call_llm_json, LLMError

        # Always return invalid JSON
        mock_call_llm.return_value = "Not valid JSON ever"

        with pytest.raises(LLMError) as exc_info:
            call_llm_json("Return JSON")

        assert "Failed to get valid JSON" in str(exc_info.value)


class TestRepairJSON:
    """Tests for the repair_json function."""

    def test_repair_json_valid_json(self):
        """repair_json should handle already valid JSON."""
        from agent.llm.client import repair_json

        result = repair_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_repair_json_with_markdown(self):
        """repair_json should strip markdown code blocks."""
        from agent.llm.client import repair_json

        text = '```json\n{"key": "value"}\n```'
        result = repair_json(text)
        assert result == {"key": "value"}

    def test_repair_json_with_preamble(self):
        """repair_json should extract JSON from surrounding text."""
        from agent.llm.client import repair_json

        text = 'Here is the JSON: {"key": "value"}'
        result = repair_json(text)
        assert result == {"key": "value"}

    def test_repair_json_no_json_found(self):
        """repair_json should raise ValueError when no JSON found."""
        from agent.llm.client import repair_json

        with pytest.raises(ValueError) as exc_info:
            repair_json("This has no JSON at all")

        assert "No JSON object found" in str(exc_info.value)


class TestCallLLMChat:
    """Tests for the call_llm_chat function."""

    @patch("agent.llm.client._get_client")
    def test_call_llm_chat_success(self, mock_get_client):
        """call_llm_chat should return assistant message content."""
        from agent.llm.client import call_llm_chat

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "I'm doing well, thanks!"
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Hello, how are you?"}]

        result = call_llm_chat(messages)

        assert result == "I'm doing well, thanks!"

    @patch("agent.llm.client._get_client")
    def test_call_llm_chat_with_model_override(self, mock_get_client):
        """call_llm_chat should use provided model override."""
        from agent.llm.client import call_llm_chat

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_client.chat.return_value = mock_response
        mock_get_client.return_value = mock_client

        messages = [{"role": "user", "content": "Test"}]
        call_llm_chat(messages, model="llama3")

        call_args = mock_client.chat.call_args
        assert call_args.kwargs.get("model") == "llama3"
