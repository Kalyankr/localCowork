"""Tests for tool result caching within a task run."""

from unittest.mock import MagicMock, patch

import pytest


class TestToolResultCaching:
    """Idempotent tool results should be cached within a task run."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(sandbox=mock_sandbox, max_iterations=3)

    @pytest.mark.asyncio
    async def test_read_file_cached_on_repeat(self, agent):
        """Second read_file with same args should come from cache."""
        from agent.orchestrator.react_agent import Action

        call_count = 0

        async def counting_execute(args, context):
            nonlocal call_count
            call_count += 1
            return {"status": "success", "output": "file contents"}

        tool = MagicMock()
        tool.name = "read_file"
        tool.execute = counting_execute

        context: dict = {}
        with (
            patch("agent.orchestrator.react_agent.tool_registry") as mock_reg,
            patch("agent.orchestrator.react_agent.settings") as mock_settings,
        ):
            mock_settings.shell_timeout = 600
            mock_settings.tool_timeout = 120
            mock_settings.max_tool_output = 50_000
            mock_reg.get.return_value = tool

            action = Action(tool="read_file", args={"path": "/tmp/test.txt"})

            r1 = await agent._execute_action(action, context)
            r2 = await agent._execute_action(action, context)

        assert r1.status == "success"
        assert r2.status == "success"
        assert r2.output == r1.output
        assert call_count == 1  # Only executed once
        assert r2.duration_ms == 0  # Cache hit has no duration

    @pytest.mark.asyncio
    async def test_different_args_not_cached(self, agent):
        """Different args should bypass cache."""
        from agent.orchestrator.react_agent import Action

        call_count = 0

        async def counting_execute(args, context):
            nonlocal call_count
            call_count += 1
            return {"status": "success", "output": f"content of {args.get('path')}"}

        tool = MagicMock()
        tool.name = "read_file"
        tool.execute = counting_execute

        context: dict = {}
        with (
            patch("agent.orchestrator.react_agent.tool_registry") as mock_reg,
            patch("agent.orchestrator.react_agent.settings") as mock_settings,
        ):
            mock_settings.shell_timeout = 600
            mock_settings.tool_timeout = 120
            mock_settings.max_tool_output = 50_000
            mock_reg.get.return_value = tool

            a1 = Action(tool="read_file", args={"path": "/tmp/a.txt"})
            a2 = Action(tool="read_file", args={"path": "/tmp/b.txt"})
            await agent._execute_action(a1, context)
            await agent._execute_action(a2, context)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_cacheable_tool_not_cached(self, agent):
        """Shell tool should never be cached."""
        from agent.orchestrator.react_agent import Action

        call_count = 0

        async def counting_execute(args, context):
            nonlocal call_count
            call_count += 1
            return {"status": "success", "output": "hello"}

        tool = MagicMock()
        tool.name = "shell"
        tool.execute = counting_execute

        context: dict = {}
        with (
            patch("agent.orchestrator.react_agent.tool_registry") as mock_reg,
            patch("agent.orchestrator.react_agent.settings") as mock_settings,
        ):
            mock_settings.shell_timeout = 600
            mock_settings.tool_timeout = 120
            mock_settings.max_tool_output = 50_000
            mock_reg.get.return_value = tool

            action = Action(tool="shell", args={"command": "echo hi"})
            await agent._execute_action(action, context)
            await agent._execute_action(action, context)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_list_dir_is_cacheable(self, agent):
        """list_dir should be cached on repeat."""
        from agent.orchestrator.react_agent import Action

        call_count = 0

        async def counting_execute(args, context):
            nonlocal call_count
            call_count += 1
            return {"status": "success", "output": {"entries": [], "count": 0}}

        tool = MagicMock()
        tool.name = "list_dir"
        tool.execute = counting_execute

        context: dict = {}
        with (
            patch("agent.orchestrator.react_agent.tool_registry") as mock_reg,
            patch("agent.orchestrator.react_agent.settings") as mock_settings,
        ):
            mock_settings.shell_timeout = 600
            mock_settings.tool_timeout = 120
            mock_settings.max_tool_output = 50_000
            mock_reg.get.return_value = tool

            action = Action(tool="list_dir", args={"path": "/tmp"})
            await agent._execute_action(action, context)
            await agent._execute_action(action, context)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cache_isolated_per_context(self, agent):
        """Different context dicts should have independent caches."""
        from agent.orchestrator.react_agent import Action

        call_count = 0

        async def counting_execute(args, context):
            nonlocal call_count
            call_count += 1
            return {"status": "success", "output": "data"}

        tool = MagicMock()
        tool.name = "read_file"
        tool.execute = counting_execute

        ctx1: dict = {}
        ctx2: dict = {}
        with (
            patch("agent.orchestrator.react_agent.tool_registry") as mock_reg,
            patch("agent.orchestrator.react_agent.settings") as mock_settings,
        ):
            mock_settings.shell_timeout = 600
            mock_settings.tool_timeout = 120
            mock_settings.max_tool_output = 50_000
            mock_reg.get.return_value = tool

            action = Action(tool="read_file", args={"path": "/tmp/x.txt"})
            await agent._execute_action(action, ctx1)
            await agent._execute_action(action, ctx2)

        assert call_count == 2  # Different contexts, no sharing

    def test_cacheable_tools_set(self):
        """Verify _CACHEABLE_TOOLS contains expected tools."""
        from agent.orchestrator.react_agent import _CACHEABLE_TOOLS

        assert "read_file" in _CACHEABLE_TOOLS
        assert "list_dir" in _CACHEABLE_TOOLS
        assert "shell" not in _CACHEABLE_TOOLS
        assert "write_file" not in _CACHEABLE_TOOLS
