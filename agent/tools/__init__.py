"""Tool plugin registry for extensible tool dispatch.

Provides a ToolPlugin protocol and ToolRegistry for registering /
discovering tools at runtime.  Built-in tools (shell, python,
web_search, fetch_webpage) are registered on import.
"""

from agent.tools.registry import ToolPlugin, ToolRegistry, tool_registry

__all__ = ["ToolPlugin", "ToolRegistry", "tool_registry"]
