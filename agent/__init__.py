"""LocalCowork - Your Local AI Assistant.

A privacy-first AI assistant that runs entirely on your machine.
Transforms natural language requests into executable multi-step plans.
"""

from agent.version import __version__

__all__ = [
    "__version__",
    "cli",
    "config",
    "llm",
    "orchestrator",
    "sandbox",
]
