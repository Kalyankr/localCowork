"""
Shared dependencies: ToolRegistry + Sandbox instances.
Import from here to avoid duplicate setup across CLI and server.
"""

from agent.orchestrator.tool_registry import ToolRegistry
from agent.tools import file_tools, markdown_tools, data_tools, pdf_tools, text_tools
from agent.sandbox.sandbox_runner import Sandbox

# Singleton tool registry
tool_registry = ToolRegistry()
tool_registry.register("file_op", file_tools.dispatch)
tool_registry.register("markdown_op", markdown_tools.dispatch)
tool_registry.register("data_op", data_tools.dispatch)
tool_registry.register("pdf_op", pdf_tools.dispatch)
tool_registry.register("text_op", text_tools.dispatch)

# Singleton sandbox
sandbox = Sandbox()
