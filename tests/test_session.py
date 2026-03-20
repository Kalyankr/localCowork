"""Tests for session management: concurrency, cleanup, persistence."""

import asyncio
import time

import pytest

# =============================================================================
# Core session operations
# =============================================================================


class TestSessionOperations:
    """Tests for add_message / get_history basic operations."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_message(self):
        from agent.orchestrator.session import add_message, get_history

        sid = f"test-session-{time.time()}"
        await add_message(sid, "user", "hello")
        history = await get_history(sid)
        assert len(history) >= 1
        assert history[-1].role == "user"
        assert history[-1].content == "hello"

    @pytest.mark.asyncio
    async def test_history_returns_copy(self):
        """Modifying returned list should not affect internal state."""
        from agent.orchestrator.session import add_message, get_history

        sid = f"test-session-copy-{time.time()}"
        await add_message(sid, "user", "msg1")
        h1 = await get_history(sid)
        h1.clear()  # Mutate the returned list
        h2 = await get_history(sid)
        assert len(h2) >= 1  # Internal state unaffected

    @pytest.mark.asyncio
    async def test_history_trimming(self):
        """History should be trimmed to MAX_HISTORY."""
        from agent.orchestrator.session import MAX_HISTORY, add_message, get_history

        sid = f"test-session-trim-{time.time()}"
        for i in range(MAX_HISTORY + 5):
            await add_message(sid, "user", f"msg-{i}")
        history = await get_history(sid)
        assert len(history) <= MAX_HISTORY


# =============================================================================
# Concurrent access (race condition tests)
# =============================================================================


class TestSessionConcurrency:
    """Verify that per-session locking prevents data corruption."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_same_session(self):
        """Multiple concurrent add_message calls should not lose messages."""
        from agent.orchestrator.session import add_message, get_history

        sid = f"test-concurrent-{time.time()}"
        n = 20

        async def write(i: int):
            await add_message(sid, "user", f"concurrent-{i}")

        await asyncio.gather(*(write(i) for i in range(n)))
        history = await get_history(sid)

        # All messages should be present (order may vary)
        contents = {m.content for m in history}
        for i in range(n):
            assert f"concurrent-{i}" in contents

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self):
        """Reading and writing concurrently should not raise."""
        from agent.orchestrator.session import add_message, get_history

        sid = f"test-rw-{time.time()}"
        await add_message(sid, "user", "seed")

        async def reader():
            for _ in range(10):
                await get_history(sid)

        async def writer():
            for i in range(10):
                await add_message(sid, "user", f"w-{i}")

        await asyncio.gather(reader(), writer())
        # Just verify no exceptions; data integrity checked above


# =============================================================================
# Session cleanup / expiration
# =============================================================================


class TestSessionCleanup:
    """Tests for session timeout and cleanup."""

    @pytest.mark.asyncio
    async def test_expired_session_is_cleaned(self):
        from agent.orchestrator.session import (
            add_message,
            cleanup_sessions,
            conversation_history,
            conversation_timestamps,
        )

        sid = f"test-expire-{time.time()}"
        await add_message(sid, "user", "old message")

        # Artificially expire the session
        conversation_timestamps[sid] = time.time() - 999_999
        await cleanup_sessions()

        assert sid not in conversation_history

    def test_sync_cleanup_expired_sessions(self):
        from agent.orchestrator.session import (
            add_message_sync,
            cleanup_sessions_sync,
            conversation_history,
            conversation_timestamps,
        )

        sid = f"test-sync-expire-{time.time()}"
        add_message_sync(sid, "user", "old")
        conversation_timestamps[sid] = time.time() - 999_999
        cleanup_sessions_sync()

        assert sid not in conversation_history

    @pytest.mark.asyncio
    async def test_non_expired_session_survives_cleanup(self):
        from agent.orchestrator.session import (
            add_message,
            cleanup_sessions,
            conversation_history,
        )

        sid = f"test-alive-{time.time()}"
        await add_message(sid, "user", "still here")
        await cleanup_sessions()

        assert sid in conversation_history
