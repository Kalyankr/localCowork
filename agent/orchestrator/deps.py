"""
Shared dependencies: Sandbox instance for Python execution.

The ReAct agent uses shell + python directly, not registered tools.
Tool modules (file_tools, web_tools, etc.) are available for direct import
if needed, but the agent doesn't require a registry.
"""

from functools import lru_cache
from typing import Tuple

from agent.orchestrator.tool_registry import ToolRegistry
from agent.sandbox.sandbox_runner import Sandbox


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    """
    Get the tool registry (minimal - for backward compatibility only).

    The ReAct agent uses shell and python directly, so the registry
    is mostly empty. Tools can still be registered for special cases.
    """
    registry = ToolRegistry()
    # No tools registered by default - agent uses shell + python directly
    # Tools can be added here for special integrations if needed
    return registry


@lru_cache(maxsize=1)
def get_sandbox() -> Sandbox:
    """
    Get the singleton sandbox instance for Python code execution.

    Uses lru_cache to ensure only one instance is created.
    """
    return Sandbox()


@lru_cache(maxsize=1)
def get_task_manager():
    """
    Get the singleton task manager instance.

    Uses lru_cache to ensure only one instance is created.
    """
    from agent.orchestrator.task_manager import TaskManager

    return TaskManager()


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
    if name == "task_manager":
        return get_task_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
