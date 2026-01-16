"""Tool modules for localCowork."""

__all__ = ["file_tools", "markdown_tools", "data_tools", "pdf_tools", "text_tools", "create_default_registry"]


def create_default_registry():
    """Create a ToolRegistry with all default tools registered."""
    from agent.orchestrator.tool_registry import ToolRegistry
    from agent.tools import file_tools, markdown_tools, data_tools, pdf_tools, text_tools
    
    registry = ToolRegistry()
    registry.register("file_op", file_tools.dispatch)
    registry.register("markdown_op", markdown_tools.dispatch)
    registry.register("data_op", data_tools.dispatch)
    registry.register("pdf_op", pdf_tools.dispatch)
    registry.register("text_op", text_tools.dispatch)
    return registry