"""Sandbox module for secure code execution."""

__all__ = ["Sandbox"]


def __getattr__(name: str):
    """Lazy import."""
    if name == "Sandbox":
        from agent.sandbox.sandbox_runner import Sandbox

        return Sandbox
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
