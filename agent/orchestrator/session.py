"""Session management for conversations.

This module handles:
- Conversation history storage (SQLite-backed with in-memory cache)
- Session cleanup
- Session-related utilities
"""

import asyncio
import sqlite3
import time
from collections import defaultdict

from agent.config import settings
from agent.orchestrator.models import ConversationMessage

# In-memory cache (populated from SQLite on first access)
conversation_history: dict[str, list[ConversationMessage]] = defaultdict(list)
conversation_timestamps: dict[str, float] = {}

# Per-session locks to prevent race conditions on concurrent access
_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_global_lock = asyncio.Lock()  # Protects _session_locks dict and cleanup

SESSION_TIMEOUT = settings.session_timeout
MAX_HISTORY = settings.max_history_messages

# Track whether cache is loaded from DB
_cache_loaded = False


def _get_sync_db() -> sqlite3.Connection:
    """Get a synchronous SQLite connection."""
    from agent.orchestrator.database import get_sync_connection

    return get_sync_connection()


def _load_cache_if_needed():
    """Load conversation cache from SQLite on first access (sync)."""
    global _cache_loaded
    if _cache_loaded:
        return
    try:
        conn = _get_sync_db()
        cursor = conn.execute(
            """SELECT session_id, role, content, timestamp
               FROM conversations ORDER BY id ASC"""
        )
        for row in cursor.fetchall():
            sid = row[0]
            conversation_history[sid].append(
                ConversationMessage(role=row[1], content=row[2])
            )
            conversation_timestamps[sid] = row[3]
        conn.close()
        _cache_loaded = True
    except Exception:
        _cache_loaded = True  # Don't retry on failure


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create a per-session lock."""
    async with _global_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


async def cleanup_sessions():
    """Remove expired sessions."""
    _load_cache_if_needed()
    async with _global_lock:
        now = time.time()
        expired = [
            s for s, t in conversation_timestamps.items() if now - t > SESSION_TIMEOUT
        ]
        for s in expired:
            conversation_history.pop(s, None)
            conversation_timestamps.pop(s, None)
            _session_locks.pop(s, None)
        # Clean up in DB too
        if expired:
            try:
                conn = _get_sync_db()
                placeholders = ",".join("?" * len(expired))
                conn.execute(
                    f"DELETE FROM conversations WHERE session_id IN ({placeholders})",
                    expired,
                )
                conn.commit()
                conn.close()
            except Exception:
                pass


def cleanup_sessions_sync():
    """Synchronous cleanup for backward compatibility."""
    _load_cache_if_needed()
    now = time.time()
    expired = [
        s for s, t in conversation_timestamps.items() if now - t > SESSION_TIMEOUT
    ]
    for s in expired:
        conversation_history.pop(s, None)
        conversation_timestamps.pop(s, None)
        _session_locks.pop(s, None)
    if expired:
        try:
            conn = _get_sync_db()
            placeholders = ",".join("?" * len(expired))
            conn.execute(
                f"DELETE FROM conversations WHERE session_id IN ({placeholders})",
                expired,
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


async def get_history(session_id: str) -> list[ConversationMessage]:
    """Get conversation history for a session (async, thread-safe)."""
    _load_cache_if_needed()
    await cleanup_sessions()
    lock = await _get_session_lock(session_id)
    async with lock:
        return list(conversation_history.get(session_id, []))


def get_history_sync(session_id: str) -> list[ConversationMessage]:
    """Get conversation history synchronously (for non-async callers)."""
    _load_cache_if_needed()
    cleanup_sessions_sync()
    return list(conversation_history.get(session_id, []))


async def add_message(session_id: str, role: str, content: str):
    """Add a message to session history (async, thread-safe)."""
    _load_cache_if_needed()
    lock = await _get_session_lock(session_id)
    async with lock:
        msg = ConversationMessage(role=role, content=content)
        now = time.time()
        conversation_history[session_id].append(msg)
        conversation_timestamps[session_id] = now

        # Trim in memory
        if len(conversation_history[session_id]) > MAX_HISTORY:
            conversation_history[session_id] = conversation_history[session_id][
                -MAX_HISTORY:
            ]

        # Persist to SQLite
        try:
            conn = _get_sync_db()
            conn.execute(
                """INSERT INTO conversations (session_id, role, content, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (session_id, role, content, now),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # In-memory cache is still valid


def add_message_sync(session_id: str, role: str, content: str):
    """Add a message synchronously (for non-async callers)."""
    _load_cache_if_needed()
    msg = ConversationMessage(role=role, content=content)
    now = time.time()
    conversation_history[session_id].append(msg)
    conversation_timestamps[session_id] = now

    if len(conversation_history[session_id]) > MAX_HISTORY:
        conversation_history[session_id] = conversation_history[session_id][
            -MAX_HISTORY:
        ]

    # Persist to SQLite
    try:
        conn = _get_sync_db()
        conn.execute(
            """INSERT INTO conversations (session_id, role, content, timestamp)
               VALUES (?, ?, ?, ?)""",
            (session_id, role, content, now),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
