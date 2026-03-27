"""Tool plugin protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


@runtime_checkable
class ToolPlugin(Protocol):
    """Protocol that all tool plugins must satisfy.

    Attributes:
        name: Unique tool identifier (e.g. "shell").
        description: One-line description shown to the LLM.
        args_schema: Mapping of argument name → short description,
            used to generate the prompt's tool docs.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def args_schema(self) -> dict[str, str]: ...

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the tool.

        Returns a dict with at least:
            status: "success" | "error"
            output: Any (on success)
            error: str  (on error)
        """
        ...


class ToolRegistry:
    """Central registry for tool plugins."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolPlugin] = {}

    def register(self, tool: ToolPlugin) -> None:
        """Register a tool plugin (replaces any existing tool with same name)."""
        self._tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if not found)."""
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolPlugin | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolPlugin]:
        """Return all registered tools in registration order."""
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def get_tool_descriptions(self, tool_names: list[str] | None = None) -> str:
        """Build the tool-documentation block for LLM prompts.

        Args:
            tool_names: If provided, only include these tools.
                        If *None*, include all registered tools.
        """
        include = set(tool_names) if tool_names is not None else None
        lines: list[str] = []
        for tool in self._tools.values():
            if include is not None and tool.name not in include:
                continue
            if tool.args_schema:
                args_parts = ", ".join(
                    f'"{k}": "{v}"' for k, v in tool.args_schema.items()
                )
                lines.append(
                    f"**{tool.name}** - {tool.description}. Args: {{{{{args_parts}}}}}"
                )
            else:
                lines.append(f"**{tool.name}** - {tool.description}")
        return "\n".join(lines)


# Global singleton
tool_registry = ToolRegistry()
