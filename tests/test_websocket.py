"""Tests for WebSocket endpoints and ConnectionManager."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for the FastAPI app."""
    from agent.orchestrator.server import app

    return TestClient(app)


# =============================================================================
# ConnectionManager unit tests
# =============================================================================


class TestConnectionManager:
    """Tests for the WebSocket ConnectionManager."""

    def test_disconnect_removes_connection(self):
        from agent.orchestrator.websocket import ConnectionManager

        mgr = ConnectionManager()
        ws = MagicMock()
        mgr.connections.add(ws)
        mgr.task_subs["task-1"].add(ws)

        mgr.disconnect(ws)

        assert ws not in mgr.connections
        assert ws not in mgr.task_subs["task-1"]

    def test_disconnect_unknown_connection_is_noop(self):
        from agent.orchestrator.websocket import ConnectionManager

        mgr = ConnectionManager()
        mgr.disconnect(MagicMock())  # Should not raise

    def test_subscribe_adds_to_task_subs(self):
        from agent.orchestrator.websocket import ConnectionManager

        mgr = ConnectionManager()
        ws = MagicMock()
        mgr.subscribe(ws, "task-abc")

        assert ws in mgr.task_subs["task-abc"]

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_subscribers(self):
        from agent.orchestrator.websocket import ConnectionManager

        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        mgr.task_subs["t1"].add(ws1)
        mgr.task_subs["t1"].add(ws2)

        await mgr.broadcast("t1", {"type": "test"})

        ws1.send_json.assert_called_once_with({"type": "test"})
        ws2.send_json.assert_called_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self):
        from agent.orchestrator.websocket import ConnectionManager

        mgr = ConnectionManager()
        alive = AsyncMock()
        dead = AsyncMock()
        dead.send_json.side_effect = RuntimeError("closed")
        mgr.connections.add(dead)
        mgr.task_subs["t1"].update({alive, dead})

        await mgr.broadcast("t1", {"type": "test"})

        assert dead not in mgr.task_subs["t1"]
        assert dead not in mgr.connections

    @pytest.mark.asyncio
    async def test_broadcast_all_sends_to_all_connections(self):
        from agent.orchestrator.models import WebSocketMessage, WSMessageType
        from agent.orchestrator.websocket import ConnectionManager

        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        mgr.connections.update({ws1, ws2})

        msg = WebSocketMessage(type=WSMessageType.PONG)
        await mgr.broadcast_all(msg)

        assert ws1.send_json.called
        assert ws2.send_json.called

    @pytest.mark.asyncio
    async def test_broadcast_to_nonexistent_task_is_noop(self):
        from agent.orchestrator.websocket import ConnectionManager

        mgr = ConnectionManager()
        await mgr.broadcast("no-such-task", {"type": "test"})  # Should not raise


# =============================================================================
# Sanitization helpers
# =============================================================================


class TestSanitizeHelpers:
    """Tests for WebSocket input sanitization utilities."""

    def test_sanitize_ws_string_truncates_long_input(self):
        from agent.orchestrator.websocket import _sanitize_ws_string

        long_text = "a" * 10_000
        result = _sanitize_ws_string(long_text, max_length=100)
        assert len(result) == 100

    def test_sanitize_ws_string_strips_whitespace(self):
        from agent.orchestrator.websocket import _sanitize_ws_string

        assert _sanitize_ws_string("  hello  ") == "hello"

    def test_sanitize_ws_string_rejects_non_string(self):
        from agent.orchestrator.websocket import _sanitize_ws_string

        assert _sanitize_ws_string(12345) == ""
        assert _sanitize_ws_string(None) == ""

    def test_validate_task_id_accepts_uuid(self):
        from agent.orchestrator.websocket import _validate_task_id

        assert _validate_task_id("abc-123-def") == "abc-123-def"

    def test_validate_task_id_rejects_special_chars(self):
        from agent.orchestrator.websocket import _validate_task_id

        assert _validate_task_id("task/../etc") is None
        assert _validate_task_id("task;drop table") is None

    def test_validate_task_id_rejects_non_string(self):
        from agent.orchestrator.websocket import _validate_task_id

        assert _validate_task_id(999) is None
        assert _validate_task_id(None) is None

    def test_validate_task_id_rejects_empty(self):
        from agent.orchestrator.websocket import _validate_task_id

        assert _validate_task_id("") is None
        assert _validate_task_id("   ") is None


# =============================================================================
# WebSocket endpoint integration tests
# =============================================================================


class TestWebSocketEndpoint:
    """Integration tests for the /ws endpoint."""

    def test_ping_pong(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_subscribe_and_unsubscribe(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "subscribe", "task_id": "t-1"})
            data = ws.receive_json()
            assert data["type"] == "subscribed"
            assert data["task_id"] == "t-1"

            ws.send_json({"type": "unsubscribe", "task_id": "t-1"})
            data = ws.receive_json()
            assert data["type"] == "unsubscribed"

    def test_subscribe_without_task_id_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "subscribe"})
            data = ws.receive_json()
            assert data["type"] == "error"

    def test_invalid_json_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not json at all")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON" in data["data"]["message"]

    def test_non_object_json_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text("[1, 2, 3]")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "JSON object" in data["data"]["message"]

    def test_oversized_message_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            # Send message exceeding MAX_WS_MESSAGE_SIZE (64 KB)
            ws.send_text("{" + " " * 70_000 + "}")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "too large" in data["data"]["message"]

    def test_invalid_message_type_returns_validation_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "totally_bogus_type"})
            data = ws.receive_json()
            assert data["type"] == "error"

    def test_subscribe_with_injection_in_task_id(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "subscribe", "task_id": "../../etc/passwd"})
            data = ws.receive_json()
            assert data["type"] == "error"
