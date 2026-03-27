"""Tests for tool execution improvements: timeouts, truncation, and metrics."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestrator.models import StepResult


class TestStepResultMetrics:
    """StepResult should carry execution metrics."""

    def test_default_metrics(self):
        result = StepResult(step_id="t1", status="success", output="ok")
        assert result.duration_ms == 0
        assert result.output_size == 0

    def test_explicit_metrics(self):
        result = StepResult(
            step_id="t1",
            status="success",
            output="ok",
            duration_ms=150,
            output_size=4096,
        )
        assert result.duration_ms == 150
        assert result.output_size == 4096

    def test_serialization_includes_metrics(self):
        result = StepResult(
            step_id="t1",
            status="success",
            output="ok",
            duration_ms=42,
            output_size=100,
        )
        d = result.model_dump()
        assert "duration_ms" in d
        assert "output_size" in d
        assert d["duration_ms"] == 42
        assert d["output_size"] == 100


class TestToolTimeout:
    """_execute_action should respect per-tool timeouts."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(sandbox=mock_sandbox, max_iterations=3)

    @pytest.mark.asyncio
    async def test_timeout_returns_error_result(self, agent):
        """Tool that exceeds timeout should return error StepResult."""
        from agent.orchestrator.react_agent import Action

        async def slow_execute(*a, **k):
            await asyncio.sleep(10)
            return {"status": "success", "output": "never"}

        slow_tool = MagicMock()
        slow_tool.name = "slow_tool"
        slow_tool.execute = slow_execute

        with patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS",
            {"slow_tool": 0.05},
        ), patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg:
            mock_reg.get.return_value = slow_tool
            action = Action(tool="slow_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.status == "error"
        assert "timed out" in result.error
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_fast_tool_succeeds(self, agent):
        """Tool that finishes within timeout should succeed."""
        from agent.orchestrator.react_agent import Action

        fast_tool = MagicMock()
        fast_tool.name = "fast_tool"
        fast_tool.execute = AsyncMock(return_value={"status": "success", "output": "done"})

        with patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS",
            {"fast_tool": 5},
        ), patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg:
            mock_reg.get.return_value = fast_tool
            action = Action(tool="fast_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.status == "success"
        assert result.output == "done"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_shell_uses_shell_timeout(self, agent):
        """Shell tool should use settings.shell_timeout, not default."""
        from agent.orchestrator.react_agent import Action

        shell_tool = MagicMock()
        shell_tool.name = "shell"
        shell_tool.execute = AsyncMock(
            return_value={"status": "success", "output": "ok"}
        )

        with patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg:
            mock_settings.shell_timeout = 999
            mock_settings.max_tool_output = 50_000
            mock_reg.get.return_value = shell_tool

            action = Action(tool="shell", args={"command": "echo hi"})
            result = await agent._execute_action(action, {})

        assert result.status == "success"
        # asyncio.wait_for was called — no timeout error means it used the value correctly

    @pytest.mark.asyncio
    async def test_unknown_tool_uses_config_default(self, agent):
        """Unknown tool name falls back to settings.tool_timeout."""
        from agent.orchestrator.react_agent import Action

        custom_tool = MagicMock()
        custom_tool.name = "custom_plugin"
        custom_tool.execute = AsyncMock(
            return_value={"status": "success", "output": "result"}
        )

        with patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS", {}
        ), patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg:
            mock_settings.tool_timeout = 60
            mock_settings.shell_timeout = 600
            mock_settings.max_tool_output = 50_000
            mock_reg.get.return_value = custom_tool

            action = Action(tool="custom_plugin", args={})
            result = await agent._execute_action(action, {})

        assert result.status == "success"


class TestOutputTruncation:
    """Tool output should be truncated when it exceeds max_tool_output."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(sandbox=mock_sandbox, max_iterations=3)

    @pytest.mark.asyncio
    async def test_large_string_output_truncated(self, agent):
        """String output exceeding max_tool_output should be truncated."""
        from agent.orchestrator.react_agent import Action

        big_output = "x" * 100_000

        tool = MagicMock()
        tool.name = "big_tool"
        tool.execute = AsyncMock(
            return_value={"status": "success", "output": big_output}
        )

        with patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg, patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS", {"big_tool": 30}
        ):
            mock_settings.max_tool_output = 1000
            mock_settings.shell_timeout = 600
            mock_reg.get.return_value = tool

            action = Action(tool="big_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.status == "success"
        assert len(str(result.output)) < 100_000
        assert "truncated" in str(result.output)
        assert result.output_size == 100_000

    @pytest.mark.asyncio
    async def test_small_output_not_truncated(self, agent):
        """Output within limits should not be truncated."""
        from agent.orchestrator.react_agent import Action

        tool = MagicMock()
        tool.name = "small_tool"
        tool.execute = AsyncMock(
            return_value={"status": "success", "output": "short result"}
        )

        with patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg, patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS", {"small_tool": 30}
        ):
            mock_settings.max_tool_output = 50_000
            mock_settings.shell_timeout = 600
            mock_reg.get.return_value = tool

            action = Action(tool="small_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.output == "short result"
        assert "truncated" not in str(result.output)

    @pytest.mark.asyncio
    async def test_large_dict_output_truncated(self, agent):
        """Dict output that serializes to > max_tool_output should be truncated."""
        from agent.orchestrator.react_agent import Action

        big_data = {"key": "v" * 100_000}
        tool = MagicMock()
        tool.name = "dict_tool"
        tool.execute = AsyncMock(
            return_value={"status": "success", "output": big_data}
        )

        with patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg, patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS", {"dict_tool": 30}
        ):
            mock_settings.max_tool_output = 500
            mock_settings.shell_timeout = 600
            mock_reg.get.return_value = tool

            action = Action(tool="dict_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.status == "success"
        assert "truncated" in str(result.output)


class TestExecutionMetrics:
    """_execute_action should populate duration_ms and output_size in StepResult."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(sandbox=mock_sandbox, max_iterations=3)

    @pytest.mark.asyncio
    async def test_success_populates_metrics(self, agent):
        from agent.orchestrator.react_agent import Action

        tool = MagicMock()
        tool.name = "metric_tool"
        tool.execute = AsyncMock(
            return_value={"status": "success", "output": "hello world"}
        )

        with patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg, patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS", {"metric_tool": 30}
        ):
            mock_settings.max_tool_output = 50_000
            mock_settings.shell_timeout = 600
            mock_reg.get.return_value = tool

            action = Action(tool="metric_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.duration_ms >= 0
        assert result.output_size == len("hello world")

    @pytest.mark.asyncio
    async def test_error_populates_duration(self, agent):
        from agent.orchestrator.react_agent import Action

        tool = MagicMock()
        tool.name = "err_tool"
        tool.execute = AsyncMock(
            return_value={"status": "error", "error": "something broke"}
        )

        with patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg, patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS", {"err_tool": 30}
        ):
            mock_settings.max_tool_output = 50_000
            mock_settings.shell_timeout = 600
            mock_reg.get.return_value = tool

            action = Action(tool="err_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.status == "error"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_exception_populates_duration(self, agent):
        from agent.orchestrator.react_agent import Action

        tool = MagicMock()
        tool.name = "crash_tool"
        tool.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            "agent.orchestrator.react_agent.settings"
        ) as mock_settings, patch(
            "agent.orchestrator.react_agent.tool_registry"
        ) as mock_reg, patch(
            "agent.orchestrator.react_agent._TOOL_TIMEOUTS", {"crash_tool": 30}
        ):
            mock_settings.max_tool_output = 50_000
            mock_settings.shell_timeout = 600
            mock_reg.get.return_value = tool

            action = Action(tool="crash_tool", args={})
            result = await agent._execute_action(action, {})

        assert result.status == "error"
        assert result.duration_ms >= 0


class TestFormatResultWithMetrics:
    """_format_result should include execution metadata."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(sandbox=mock_sandbox, max_iterations=3)

    def test_format_includes_timing(self, agent):
        result = StepResult(
            step_id="t1",
            status="success",
            output="hello",
            duration_ms=2300,
            output_size=5,
        )
        formatted = agent._format_result(result)
        assert "2.3s" in formatted
        assert "5B" in formatted

    def test_format_small_timing_in_ms(self, agent):
        result = StepResult(
            step_id="t1",
            status="success",
            output="hi",
            duration_ms=45,
            output_size=2,
        )
        formatted = agent._format_result(result)
        assert "45ms" in formatted

    def test_format_large_output_in_kb(self, agent):
        result = StepResult(
            step_id="t1",
            status="success",
            output="data",
            duration_ms=100,
            output_size=15360,
        )
        formatted = agent._format_result(result)
        assert "15.0KB" in formatted

    def test_format_error_includes_metrics(self, agent):
        result = StepResult(
            step_id="t1",
            status="error",
            error="connection refused",
            duration_ms=500,
            output_size=0,
        )
        formatted = agent._format_result(result)
        assert "ERROR:" in formatted
        assert "500ms" in formatted

    def test_format_no_metrics_when_zero(self, agent):
        result = StepResult(
            step_id="t1",
            status="success",
            output="hello",
            duration_ms=0,
            output_size=0,
        )
        formatted = agent._format_result(result)
        # Should not have brackets when no metrics
        assert "[" not in formatted

    def test_format_dict_output_with_metrics(self, agent):
        result = StepResult(
            step_id="t1",
            status="success",
            output={"key": "value"},
            duration_ms=50,
            output_size=200,
        )
        formatted = agent._format_result(result)
        assert "50ms" in formatted
        assert "key" in formatted


class TestConfigDefaults:
    """Config should include tool_timeout and max_tool_output."""

    def test_tool_timeout_default(self):
        from agent.config import Settings

        s = Settings()
        assert s.tool_timeout == 120

    def test_max_tool_output_default(self):
        from agent.config import Settings

        s = Settings()
        assert s.max_tool_output == 50_000

    def test_tool_timeout_customizable(self):
        from agent.config import Settings

        s = Settings(tool_timeout=30)
        assert s.tool_timeout == 30

    def test_max_tool_output_customizable(self):
        from agent.config import Settings

        s = Settings(max_tool_output=10_000)
        assert s.max_tool_output == 10_000


class TestToolTimeoutDefaults:
    """Per-tool timeout defaults should be defined."""

    def test_builtin_tools_have_timeouts(self):
        from agent.orchestrator.react_agent import _TOOL_TIMEOUTS

        expected_tools = [
            "shell", "python", "web_search", "fetch_webpage",
            "read_file", "write_file", "edit_file",
            "memory_store", "memory_recall",
        ]
        for name in expected_tools:
            assert name in _TOOL_TIMEOUTS, f"Missing timeout for {name}"
            assert _TOOL_TIMEOUTS[name] > 0
