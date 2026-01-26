"""
Shared dependencies: ToolRegistry + Sandbox instances.
Import from here to avoid duplicate setup across CLI and server.

This module provides singleton instances of the tool registry and sandbox,
ensuring consistent state across the application.
"""

from functools import lru_cache
from typing import Tuple

from agent.orchestrator.tool_registry import ToolRegistry
from agent.sandbox.sandbox_runner import Sandbox


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    """
    Get the singleton tool registry with all tools registered.
    
    Uses lru_cache to ensure only one instance is created.
    """
    from agent.tools import (
        file_tools,
        markdown_tools,
        data_tools,
        pdf_tools,
        text_tools,
        web_tools,
        shell_tools,
        json_tools,
        archive_tools,
        chat_tools,
    )
    
    registry = ToolRegistry()
    registry.register("file_op", file_tools.dispatch)
    registry.register("markdown_op", markdown_tools.dispatch)
    registry.register("data_op", data_tools.dispatch)
    registry.register("pdf_op", pdf_tools.dispatch)
    registry.register("text_op", text_tools.dispatch)
    registry.register("web_op", web_tools.dispatch)
    registry.register("shell_op", shell_tools.dispatch)
    registry.register("json_op", json_tools.dispatch)
    registry.register("archive_op", archive_tools.dispatch)
    registry.register("chat_op", chat_tools.dispatch)
    
    return registry


@lru_cache(maxsize=1)
def get_sandbox() -> Sandbox:
    """
    Get the singleton sandbox instance.
    
    Uses lru_cache to ensure only one instance is created.
    """
    return Sandbox()


def get_dependencies() -> Tuple[ToolRegistry, Sandbox]:
    """
    Get both tool registry and sandbox as a tuple.
    
    Convenience function for components that need both.
    """
    return get_tool_registry(), get_sandbox()


# For backwards compatibility - lazy loaded singletons
# Prefer using the getter functions above
def __getattr__(name: str):
    if name == "tool_registry":
        return get_tool_registry()
    if name == "sandbox":
        return get_sandbox()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
