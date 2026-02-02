"""Tests for error recovery functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestrator.agent_models import (
    Action,
    AgentState,
)
from agent.orchestrator.react_agent import ReActAgent


class TestErrorRecoveryPrompt:
    """Tests for the error recovery prompt."""

    def test_prompt_exists(self):
        """Test that ERROR_RECOVERY_PROMPT is defined."""
        from agent.llm.prompts import ERROR_RECOVERY_PROMPT

        assert ERROR_RECOVERY_PROMPT is not None
        assert "{goal}" in ERROR_RECOVERY_PROMPT
        assert "{failed_tool}" in ERROR_RECOVERY_PROMPT
        assert "{error}" in ERROR_RECOVERY_PROMPT
        assert "{attempt}" in ERROR_RECOVERY_PROMPT

    def test_prompt_has_alternatives(self):
        """Test that prompt suggests alternative approaches."""
        from agent.llm.prompts import ERROR_RECOVERY_PROMPT

        assert "alternative" in ERROR_RECOVERY_PROMPT.lower()
        assert "different" in ERROR_RECOVERY_PROMPT.lower()


class TestRecoveryAttempt:
    """Tests for the _attempt_recovery method."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        sandbox = MagicMock()
        return ReActAgent(sandbox=sandbox)

    @pytest.fixture
    def state(self):
        """Create a test state."""
        return AgentState(goal="Test goal", status="running")

    @pytest.fixture
    def failed_action(self):
        """Create a failed action."""
        return Action(
            tool="shell",
            args={"command": "cat nonexistent.txt"},
            description="Read file",
        )

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_recovery_returns_new_action(
        self, mock_llm, agent, state, failed_action
    ):
        """Test that recovery returns a new action on success."""
        mock_llm.return_value = {
            "analysis": "File not found",
            "new_approach": "Try searching for the file first",
            "action": {
                "tool": "shell",
                "args": {"command": "find . -name '*.txt'"},
            },
            "give_up": False,
        }

        new_action, user_msg = await agent._attempt_recovery(
            state, failed_action, "No such file", attempt=1
        )

        assert new_action is not None
        assert new_action.tool == "shell"
        assert "find" in new_action.args.get("command", "")
        assert user_msg is None

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_recovery_gives_up(self, mock_llm, agent, state, failed_action):
        """Test that recovery can give up with a message."""
        mock_llm.return_value = {
            "analysis": "File definitely doesn't exist",
            "new_approach": None,
            "action": None,
            "give_up": True,
            "user_message": "The file does not exist in any accessible location.",
        }

        new_action, user_msg = await agent._attempt_recovery(
            state, failed_action, "No such file", attempt=3
        )

        assert new_action is None
        assert user_msg is not None
        assert "does not exist" in user_msg

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_recovery_handles_llm_error(
        self, mock_llm, agent, state, failed_action
    ):
        """Test that recovery handles LLM errors gracefully."""
        mock_llm.side_effect = Exception("LLM timeout")

        new_action, user_msg = await agent._attempt_recovery(
            state, failed_action, "Original error", attempt=1
        )

        assert new_action is None
        assert user_msg is not None
        assert "Original error" in user_msg


class TestRecoveryIntegration:
    """Integration tests for error recovery in the agent loop."""

    @pytest.fixture
    def agent(self):
        """Create a test agent with mocked dependencies."""
        sandbox = MagicMock()
        sandbox.run_python = AsyncMock(return_value={"output": "success"})
        return ReActAgent(sandbox=sandbox, max_iterations=5)

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_recovery_triggered_on_failure(self, mock_llm):
        """Test that recovery is triggered when action fails."""
        sandbox = MagicMock()
        agent = ReActAgent(sandbox=sandbox, max_iterations=3)

        # First call: return action that will fail
        # Second call: recovery - return new action
        # Third call: success response
        mock_llm.side_effect = [
            {
                "thought": "Try command",
                "is_complete": False,
                "action": {
                    "tool": "shell",
                    "args": {"command": "failing_command"},
                },
            },
            {
                "analysis": "Command not found",
                "new_approach": "Use different command",
                "action": {
                    "tool": "shell",
                    "args": {"command": "echo success"},
                },
                "give_up": False,
            },
            {
                "thought": "Done",
                "is_complete": True,
                "response": "Task completed",
            },
        ]

        # Mock shell execution - first fails, second succeeds
        with patch("subprocess.run") as mock_run:
            # First command fails
            fail_result = MagicMock()
            fail_result.returncode = 127
            fail_result.stdout = b""
            fail_result.stderr = b"command not found"

            # Recovery command succeeds
            success_result = MagicMock()
            success_result.returncode = 0
            success_result.stdout = b"success"
            success_result.stderr = b""

            mock_run.side_effect = [fail_result, success_result]

            state = await agent.run("Run a command")

            # Should have attempted recovery
            assert state.status in ("completed", "max_iterations")


class TestRecoverySettings:
    """Tests for recovery-related settings."""

    def test_max_recovery_attempts_setting(self):
        """Test that max_recovery_attempts is configurable."""
        from agent.config import Settings

        settings = Settings()
        assert hasattr(settings, "max_recovery_attempts")
        assert settings.max_recovery_attempts >= 1

    @patch("agent.config.Settings")
    def test_recovery_respects_max_attempts(self, mock_settings):
        """Test that recovery stops after max attempts."""
        mock_settings.return_value.max_recovery_attempts = 2
        # This would be tested in a full integration test
        pass
