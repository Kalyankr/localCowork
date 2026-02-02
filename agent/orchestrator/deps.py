"""
Shared dependencies: Sandbox instance for Python execution.

The ReAct agent uses shell + python directly. No tool registry is needed
since the agent figures out what commands and code to run itself.
"""

from functools import lru_cache

from agent.config import get_settings
from agent.sandbox.sandbox_runner import Sandbox


@lru_cache(maxsize=1)
def get_sandbox(permissive: bool | None = None) -> Sandbox:
    """
    Get the singleton sandbox instance for Python code execution.

    Args:
        permissive: If True, allows full system access (needed for agentic tasks).
                   If False, runs in Docker sandbox (safer but limited).
                   If None (default), uses config setting (use_docker).

    Note: For the agentic workflow, permissive=True is required so the agent
    can actually manipulate files and execute commands. Safety is enforced
    via the safety.py module with user confirmation for dangerous operations.
    """
    if permissive is None:
        # Use config: use_docker=True means permissive=False
        permissive = not get_settings().use_docker
    return Sandbox(permissive=permissive)


@lru_cache(maxsize=1)
def get_task_manager():
    """
    Get the singleton task manager instance.

    Uses lru_cache to ensure only one instance is created.
    """
    from agent.orchestrator.task_manager import TaskManager

    return TaskManager()


# For backwards compatibility - lazy loaded singletons
def __getattr__(name: str):
    if name == "sandbox":
        return get_sandbox()
    if name == "task_manager":
        return get_task_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
