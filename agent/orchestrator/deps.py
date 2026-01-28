"""
Shared dependencies: Sandbox instance for Python execution.

The ReAct agent uses shell + python directly. No tool registry is needed
since the agent figures out what commands and code to run itself.
"""

from functools import lru_cache

from agent.sandbox.sandbox_runner import Sandbox


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


# For backwards compatibility - lazy loaded singletons
def __getattr__(name: str):
    if name == "sandbox":
        return get_sandbox()
    if name == "task_manager":
        return get_task_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
