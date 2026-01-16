"""Orchestrator module for localCowork."""

__all__ = [
    "generate_plan",
    "summarize_results",
    "Executor",
    "Plan",
    "Step",
    "StepResult",
    "TaskRequest",
    "TaskResponse",
    "ToolRegistry",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports and missing deps errors."""
    if name in ("generate_plan", "summarize_results"):
        from agent.orchestrator.planner import generate_plan, summarize_results
        return generate_plan if name == "generate_plan" else summarize_results
    if name == "Executor":
        from agent.orchestrator.executor import Executor
        return Executor
    if name in ("Plan", "Step", "StepResult", "TaskRequest", "TaskResponse"):
        from agent.orchestrator import models
        return getattr(models, name)
    if name == "ToolRegistry":
        from agent.orchestrator.tool_registry import ToolRegistry
        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")