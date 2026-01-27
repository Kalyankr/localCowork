"""Orchestrator module for localCowork - Pure Agentic."""

__all__ = [
    "ReActAgent",
    "StepResult",
    "TaskRequest",
    "ToolRegistry",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports and missing deps errors."""
    if name == "ReActAgent":
        from agent.orchestrator.react_agent import ReActAgent
        return ReActAgent
    if name in ("StepResult", "TaskRequest", "TaskResponse"):
        from agent.orchestrator import models
        return getattr(models, name)
    if name == "ToolRegistry":
        from agent.orchestrator.tool_registry import ToolRegistry
        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")