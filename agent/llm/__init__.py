"""LLM client module for localCowork."""

__all__ = ["call_llm", "call_llm_json", "LLMError"]


def __getattr__(name: str):
    """Lazy import."""
    if name in ("call_llm", "call_llm_json", "LLMError"):
        from agent.llm import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
