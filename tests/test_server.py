"""Tests for the API server endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def client():
    """Create test client for the FastAPI app."""
    from agent.orchestrator.server import app

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_html(self, client):
        """Root endpoint should serve the web UI."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestTasksEndpoint:
    """Tests for the /tasks endpoints."""

    def test_list_tasks_empty(self, client):
        """List tasks should return empty list initially."""
        response = client.get("/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_tasks_with_invalid_state(self, client):
        """List tasks with invalid state should return 400."""
        response = client.get("/tasks?state=invalid_state")
        assert response.status_code == 400

    def test_get_nonexistent_task(self, client):
        """Getting non-existent task should return 404."""
        response = client.get("/tasks/nonexistent-id")
        assert response.status_code == 404

    def test_cancel_nonexistent_task(self, client):
        """Cancelling non-existent task should return 404."""
        response = client.post("/tasks/nonexistent-id/cancel")
        assert response.status_code == 404


class TestRunEndpoint:
    """Tests for the /run endpoint."""

    def test_run_missing_request(self, client):
        """Run without request body should return 422."""
        response = client.post("/run", json={})
        assert response.status_code == 422

    @patch("agent.orchestrator.react_agent.ReActAgent")
    def test_run_with_llm_error(self, mock_agent_class, client):
        """Run should handle LLM errors gracefully."""
        from agent.llm.client import LLMError

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=LLMError("Connection failed"))
        mock_agent_class.return_value = mock_agent

        response = client.post("/run", json={"request": "test"})
        assert response.status_code == 503


class TestWebSocketEndpoint:
    """Tests for WebSocket functionality."""

    def test_websocket_connect(self, client):
        """WebSocket should accept connections."""
        with client.websocket_connect("/ws") as websocket:
            # Send ping
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_subscribe(self, client):
        """WebSocket should handle subscribe messages."""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "subscribe", "task_id": "test-123"})
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["task_id"] == "test-123"

    def test_websocket_subscribe_without_task_id(self, client):
        """WebSocket subscribe without task_id should return error."""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "subscribe"})
            data = websocket.receive_json()
            assert data["type"] == "error"

    def test_websocket_invalid_message(self, client):
        """WebSocket should handle invalid message format."""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "invalid_type"})
            # Should not crash, may receive error or be ignored
