"""Session management for conversations.

This module handles:
- Conversation history storage
- Session cleanup
- Session-related utilities
"""

import time
from collections import defaultdict

from agent.config import settings
from agent.orchestrator.models import ConversationMessage

# Session storage
conversation_history: dict[str, list[ConversationMessage]] = defaultdict(list)
conversation_timestamps: dict[str, float] = {}

SESSION_TIMEOUT = settings.session_timeout
MAX_HISTORY = settings.max_history_messages


def cleanup_sessions():
    """Remove expired sessions."""
    now = time.time()
    expired = [
        s for s, t in conversation_timestamps.items() if now - t > SESSION_TIMEOUT
    ]
    for s in expired:
        conversation_history.pop(s, None)
        conversation_timestamps.pop(s, None)


def get_history(session_id: str) -> list[ConversationMessage]:
    """Get conversation history for a session."""
    cleanup_sessions()
    return conversation_history.get(session_id, [])


def add_message(session_id: str, role: str, content: str):
    """Add a message to session history."""
    conversation_history[session_id].append(
        ConversationMessage(role=role, content=content)
    )
    conversation_timestamps[session_id] = time.time()
    if len(conversation_history[session_id]) > MAX_HISTORY:
        conversation_history[session_id] = conversation_history[session_id][
            -MAX_HISTORY:
        ]
