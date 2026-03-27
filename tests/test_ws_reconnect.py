"""Tests for WebSocket reconnection with state recovery."""

from agent.orchestrator.models import WebSocketMessage, WSMessageType

# =============================================================================
# ConnectionManager state tracking (websocket.py)
# =============================================================================


class TestConnectionManagerStateTracking:
    """Tests for the step/confirm tracking added to ConnectionManager."""

    def _make_mgr(self):
        from agent.orchestrator.websocket import ConnectionManager

        return ConnectionManager()

    def test_record_step_stores_entries(self):
        mgr = self._make_mgr()
        mgr.record_step("t1", {"iteration": 1, "status": "thinking"})
        mgr.record_step("t1", {"iteration": 2, "status": "success"})

        steps, _ = mgr.get_state_sync_data("t1")
        assert len(steps) == 2
        assert steps[0]["iteration"] == 1
        assert steps[1]["iteration"] == 2

    def test_record_step_per_task(self):
        mgr = self._make_mgr()
        mgr.record_step("t1", {"iteration": 1})
        mgr.record_step("t2", {"iteration": 1})

        s1, _ = mgr.get_state_sync_data("t1")
        s2, _ = mgr.get_state_sync_data("t2")
        assert len(s1) == 1
        assert len(s2) == 1

    def test_record_pending_confirm(self):
        mgr = self._make_mgr()
        confirm = {"confirm_id": "c1", "command": "rm -rf /"}
        mgr.record_pending_confirm("t1", confirm)

        _, pending = mgr.get_state_sync_data("t1")
        assert pending is not None
        assert pending["confirm_id"] == "c1"

    def test_clear_pending_confirm(self):
        mgr = self._make_mgr()
        mgr.record_pending_confirm("t1", {"confirm_id": "c1"})
        mgr.record_pending_confirm("t1", None)

        _, pending = mgr.get_state_sync_data("t1")
        assert pending is None

    def test_clear_task_state(self):
        mgr = self._make_mgr()
        mgr.record_step("t1", {"iteration": 1})
        mgr.record_pending_confirm("t1", {"confirm_id": "c1"})

        mgr.clear_task_state("t1")

        steps, pending = mgr.get_state_sync_data("t1")
        assert steps == []
        assert pending is None

    def test_clear_task_state_noop_for_unknown(self):
        mgr = self._make_mgr()
        mgr.clear_task_state("no-such-task")  # Should not raise

    def test_get_state_sync_data_empty(self):
        mgr = self._make_mgr()
        steps, pending = mgr.get_state_sync_data("no-such-task")
        assert steps == []
        assert pending is None

    def test_get_state_sync_data_returns_copies(self):
        """Returned lists should be independent copies."""
        mgr = self._make_mgr()
        mgr.record_step("t1", {"iteration": 1})

        steps1, _ = mgr.get_state_sync_data("t1")
        steps1.append({"iteration": 99})

        steps2, _ = mgr.get_state_sync_data("t1")
        assert len(steps2) == 1  # original unaffected


# =============================================================================
# WebSocketMessage.state_sync factory
# =============================================================================


class TestStateSyncMessage:
    """Tests for the state_sync message type."""

    def test_state_sync_type(self):
        msg = WebSocketMessage.state_sync(
            task_id="t1",
            task_state="executing",
            request="do stuff",
            steps=[],
        )
        assert msg.type == WSMessageType.STATE_SYNC
        assert msg.task_id == "t1"

    def test_state_sync_data_fields(self):
        steps = [{"iteration": 1, "status": "thinking"}]
        msg = WebSocketMessage.state_sync(
            task_id="t1",
            task_state="executing",
            request="do stuff",
            steps=steps,
        )
        assert msg.data["task_state"] == "executing"
        assert msg.data["request"] == "do stuff"
        assert msg.data["steps"] == steps
        assert "pending_confirm" not in msg.data

    def test_state_sync_with_pending_confirm(self):
        confirm = {"confirm_id": "c1", "command": "rm file"}
        msg = WebSocketMessage.state_sync(
            task_id="t1",
            task_state="executing",
            request="clean up",
            steps=[],
            pending_confirm=confirm,
        )
        assert msg.data["pending_confirm"] == confirm

    def test_state_sync_serializes(self):
        msg = WebSocketMessage.state_sync(
            task_id="t1",
            task_state="executing",
            request="test",
            steps=[{"iteration": 1}],
        )
        dumped = msg.model_dump()
        assert dumped["type"] == "state_sync"
        assert dumped["task_id"] == "t1"
        assert dumped["data"]["steps"] == [{"iteration": 1}]


# =============================================================================
# Server.py ConnectionManager state tracking
# =============================================================================


class TestServerConnectionManagerStateTracking:
    """Verify the server.py ConnectionManager also has state tracking."""

    def _make_mgr(self):
        from agent.orchestrator.server import ConnectionManager

        return ConnectionManager()

    def test_record_and_get_steps(self):
        mgr = self._make_mgr()
        mgr.record_step("t1", {"iteration": 1})
        steps, _ = mgr.get_state_sync_data("t1")
        assert len(steps) == 1

    def test_record_and_clear_confirm(self):
        mgr = self._make_mgr()
        mgr.record_pending_confirm("t1", {"confirm_id": "c1"})
        _, pending = mgr.get_state_sync_data("t1")
        assert pending is not None

        mgr.record_pending_confirm("t1", None)
        _, pending = mgr.get_state_sync_data("t1")
        assert pending is None

    def test_clear_task_state(self):
        mgr = self._make_mgr()
        mgr.record_step("t1", {"iteration": 1})
        mgr.record_pending_confirm("t1", {"confirm_id": "c1"})
        mgr.clear_task_state("t1")

        steps, pending = mgr.get_state_sync_data("t1")
        assert steps == []
        assert pending is None


# =============================================================================
# Integration: subscribe sends state_sync for in-flight tasks
# =============================================================================


class TestSubscribeStateSync:
    """Test that subscribing to an in-flight task sends state_sync."""

    def test_subscribe_to_executing_task_sends_state_sync(self):
        """When subscribing to an executing task, client gets state_sync."""
        from fastapi.testclient import TestClient

        from agent.orchestrator.server import app, task_manager, ws
        from agent.orchestrator.task_manager import TaskState as TMState

        # Create a task and mark it executing
        task = task_manager.create_task("test goal", "session-1")
        task_manager.update_state(task.id, TMState.EXECUTING)

        # Simulate some recorded steps
        ws.record_step(
            task.id,
            {"iteration": 1, "status": "thinking", "thought": "Hmm", "action": None},
        )
        ws.record_step(
            task.id,
            {
                "iteration": 2,
                "status": "success",
                "thought": "Got it",
                "action": "shell: ls",
            },
        )

        client = TestClient(app)
        try:
            with client.websocket_connect("/ws") as websocket:
                # Subscribe to the in-flight task
                websocket.send_json({"type": "subscribe", "task_id": task.id})

                # First message: subscribed confirmation
                msg1 = websocket.receive_json()
                assert msg1["type"] == "subscribed"
                assert msg1["task_id"] == task.id

                # Second message: state_sync with step history
                msg2 = websocket.receive_json()
                assert msg2["type"] == "state_sync"
                assert msg2["task_id"] == task.id
                assert msg2["data"]["task_state"] == "executing"
                assert msg2["data"]["request"] == "test goal"
                assert len(msg2["data"]["steps"]) == 2
        finally:
            # Cleanup
            ws.clear_task_state(task.id)

    def test_subscribe_to_completed_task_no_state_sync(self):
        """Subscribing to a completed task should NOT send state_sync."""
        from fastapi.testclient import TestClient

        from agent.orchestrator.server import app, task_manager
        from agent.orchestrator.task_manager import TaskState as TMState

        task = task_manager.create_task("done goal", "session-2")
        task_manager.update_state(task.id, TMState.COMPLETED)

        client = TestClient(app)
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "subscribe", "task_id": task.id})

            msg = websocket.receive_json()
            assert msg["type"] == "subscribed"

            # Send a ping to verify no state_sync is pending
            websocket.send_json({"type": "ping"})
            msg2 = websocket.receive_json()
            assert msg2["type"] == "pong"  # No state_sync was sent

    def test_subscribe_with_pending_confirm(self):
        """State sync should include pending confirmation if one exists."""
        from fastapi.testclient import TestClient

        from agent.orchestrator.server import app, task_manager, ws
        from agent.orchestrator.task_manager import TaskState as TMState

        task = task_manager.create_task("risky goal", "session-3")
        task_manager.update_state(task.id, TMState.EXECUTING)

        confirm_data = {
            "type": "confirm_request",
            "confirm_id": "c-abc",
            "task_id": task.id,
            "command": "rm -rf /tmp/old",
            "reason": "destructive",
            "message": "Delete old files?",
        }
        ws.record_pending_confirm(task.id, confirm_data)

        client = TestClient(app)
        try:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({"type": "subscribe", "task_id": task.id})

                _ = websocket.receive_json()  # subscribed
                msg = websocket.receive_json()  # state_sync

                assert msg["type"] == "state_sync"
                assert msg["data"]["pending_confirm"]["confirm_id"] == "c-abc"
        finally:
            ws.clear_task_state(task.id)
