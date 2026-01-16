"""Tool modules for localCowork."""

__all__ = [
    "file_tools", 
    "markdown_tools", 
    "data_tools", 
    "pdf_tools", 
    "text_tools",
    "web_tools",
    "shell_tools",
    "json_tools",
    "archive_tools",
    "create_default_registry",
]


def create_default_registry():
    """Create a ToolRegistry with all default tools registered."""
    from agent.orchestrator.tool_registry import ToolRegistry
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
    return registry