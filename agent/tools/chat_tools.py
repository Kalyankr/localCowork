"""Chat operations: handle conversational messages that don't require task execution."""

from agent.llm.client import call_llm


def chat_response(message: str) -> str:
    """
    Generate a conversational response for non-task messages.
    """
    prompt = f"""You are LocalCowork, a friendly AI assistant that helps with file management and local tasks.

The user said: "{message}"

This is a conversational message, not a task request. Respond naturally and briefly.
If they're greeting you, greet them back and offer to help.
If they're asking what you can do, briefly explain your capabilities:
- Organize and manage files
- Search the web
- Work with JSON and archives
- Run safe shell commands
- Summarize and transform text

Keep your response to 1-3 sentences. Be friendly and helpful.

Response:"""
    
    return call_llm(prompt)


def dispatch(op: str, **kwargs) -> str:
    """Dispatch chat operations."""
    if op == "respond":
        return chat_response(kwargs.get("message", ""))
    raise ValueError(f"Unsupported chat op: {op}")
