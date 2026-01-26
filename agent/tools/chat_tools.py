"""Chat operations: handle conversational messages that don't require task execution."""


def dispatch(**kwargs) -> str:
    """Return the response directly (planner already generated it)."""
    return kwargs.get("response", "")
