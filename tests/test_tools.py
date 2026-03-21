"""Tests for the tool plugin registry (agent/tools/)."""

from typing import Any

import pytest

from agent.tools.registry import ToolPlugin, ToolRegistry, tool_registry


class _DummyTool:
    """Minimal tool for testing."""

    name = "dummy"
    description = "A dummy tool"
    args_schema = {"arg1": "description of arg1"}

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        return {"status": "success", "output": f"got {args}"}


class _AnotherTool:
    name = "another"
    description = "Another tool"
    args_schema = {}

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        return {"status": "success", "output": "ok"}


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = _DummyTool()
        reg.register(tool)
        assert reg.get("dummy") is tool

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(_DummyTool())
        reg.unregister("dummy")
        assert reg.get("dummy") is None

    def test_unregister_missing_is_noop(self):
        reg = ToolRegistry()
        reg.unregister("nonexistent")  # should not raise

    def test_list_tools(self):
        reg = ToolRegistry()
        t1 = _DummyTool()
        t2 = _AnotherTool()
        reg.register(t1)
        reg.register(t2)
        assert reg.list_tools() == [t1, t2]

    def test_get_tool_names(self):
        reg = ToolRegistry()
        reg.register(_DummyTool())
        reg.register(_AnotherTool())
        assert reg.get_tool_names() == ["dummy", "another"]

    def test_get_tool_descriptions(self):
        reg = ToolRegistry()
        reg.register(_DummyTool())
        reg.register(_AnotherTool())
        desc = reg.get_tool_descriptions()
        assert "**dummy**" in desc
        assert "**another**" in desc
        assert "A dummy tool" in desc
        assert "arg1" in desc

    def test_register_replaces_existing(self):
        reg = ToolRegistry()
        reg.register(_DummyTool())
        new_tool = _DummyTool()
        new_tool.description = "Replaced"
        reg.register(new_tool)
        assert reg.get("dummy") is new_tool


class TestToolPluginProtocol:
    """Verify protocol compliance."""

    def test_dummy_tool_is_tool_plugin(self):
        assert isinstance(_DummyTool(), ToolPlugin)

    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        tool = _DummyTool()
        result = await tool.execute({"arg1": "hello"}, {})
        assert result["status"] == "success"
        assert "hello" in result["output"]


class TestBuiltinTools:
    """Verify built-in tools register correctly."""

    def test_register_builtin_tools(self):
        from unittest.mock import MagicMock

        from agent.tools.builtin import register_builtin_tools

        reg = ToolRegistry()
        # Temporarily swap global registry
        import agent.tools.registry as reg_mod

        original = reg_mod.tool_registry
        reg_mod.tool_registry = reg
        try:
            sandbox = MagicMock()
            register_builtin_tools(sandbox)
            names = reg.get_tool_names()
            assert "shell" in names
            assert "python" in names
            assert "web_search" in names
            assert "fetch_webpage" in names
        finally:
            reg_mod.tool_registry = original

    @pytest.mark.asyncio
    async def test_shell_tool_runs_echo(self):
        from agent.tools.builtin import ShellTool

        tool = ShellTool()
        result = await tool.execute({"command": "echo hello"}, {})
        assert result["status"] == "success"
        assert "hello" in result["output"]

    @pytest.mark.asyncio
    async def test_shell_tool_error(self):
        from agent.tools.builtin import ShellTool

        tool = ShellTool()
        result = await tool.execute({"command": "false"}, {})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_web_search_tool_missing_query(self):
        from agent.tools.builtin import WebSearchTool

        tool = WebSearchTool()
        result = await tool.execute({"query": ""}, {})
        # Empty query may return error or empty results depending on implementation
        assert "status" in result

    @pytest.mark.asyncio
    async def test_fetch_webpage_tool_bad_url(self):
        from agent.tools.builtin import FetchWebpageTool

        tool = FetchWebpageTool()
        result = await tool.execute({"url": "not-a-url"}, {})
        assert result["status"] == "error"


class TestGlobalRegistry:
    """Global tool_registry singleton."""

    def test_global_registry_exists(self):
        assert isinstance(tool_registry, ToolRegistry)
