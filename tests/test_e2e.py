"""End-to-end integration tests for the full application.

These tests verify complete flows through the system:
- Sandbox Python execution
- Server HTTP endpoints
- WebSocket real-time updates
- Real file operations
"""

import pytest
from fastapi.testclient import TestClient

from agent.orchestrator.server import app
from agent.sandbox.sandbox_runner import Sandbox

# =============================================================================
# Sandbox Integration Tests - Real Python Execution
# =============================================================================


class TestSandboxIntegration:
    """Test real sandbox Python execution (permissive mode, no Docker required)."""

    @pytest.fixture
    def sandbox(self):
        """Create a permissive sandbox for testing."""
        return Sandbox(timeout=10, permissive=True)

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace with test files."""
        (tmp_path / "test.txt").write_text("Hello, World!")
        (tmp_path / "data.json").write_text('{"name": "test"}')
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content")
        return tmp_path

    @pytest.mark.asyncio
    async def test_python_code_execution(self, sandbox):
        """Test executing Python code in sandbox."""
        result = await sandbox.run_python("print('Hello from sandbox!')")
        assert "output" in result
        assert "Hello from sandbox!" in result["output"]

    @pytest.mark.asyncio
    async def test_python_file_operations(self, sandbox, temp_workspace):
        """Test Python code can read/write files in workspace."""
        code = f"""
import os
files = os.listdir('{temp_workspace}')
print(f"Found {{len(files)}} files")
for f in files:
    print(f"  - {{f}}")
"""
        result = await sandbox.run_python(code, working_dir=str(temp_workspace))
        assert "output" in result
        assert "Found" in result["output"]
        assert "test.txt" in result["output"]

    @pytest.mark.asyncio
    async def test_python_creates_file(self, sandbox, temp_workspace):
        """Test Python code can create new files."""
        new_file = temp_workspace / "created.txt"
        code = f"""
with open('{new_file}', 'w') as f:
    f.write('Created by agent!')
print('File created successfully')
"""
        result = await sandbox.run_python(code, working_dir=str(temp_workspace))
        assert "output" in result
        assert "created successfully" in result["output"]
        assert new_file.exists()
        assert new_file.read_text() == "Created by agent!"

    @pytest.mark.asyncio
    async def test_python_handles_errors(self, sandbox):
        """Test sandbox captures Python errors."""
        result = await sandbox.run_python("raise ValueError('Test error')")
        assert "error" in result
        assert "ValueError" in result["error"]

    @pytest.mark.asyncio
    async def test_python_timeout(self):
        """Test sandbox enforces timeout."""
        sandbox = Sandbox(timeout=1, permissive=True)
        result = await sandbox.run_python("import time; time.sleep(10)")
        assert "error" in result
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_python_imports_work(self, sandbox):
        """Test Python can import standard library."""
        code = """
import json
import os
import sys
data = {"test": True, "python_version": sys.version_info.major}
print(json.dumps(data))
"""
        result = await sandbox.run_python(code)
        assert "output" in result
        assert '"test": true' in result["output"]


# =============================================================================
# Server HTTP Integration Tests
# =============================================================================


class TestServerHTTPIntegration:
    """Test server HTTP endpoints with full integration."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_root_returns_ui(self, client):
        """Test root returns the web UI."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_static_css(self, client):
        """Test CSS file is served."""
        response = client.get("/static/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    def test_static_js(self, client):
        """Test JavaScript file is served."""
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]

    def test_list_tasks_returns_array(self, client):
        """Test listing tasks returns an array."""
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        # The response is directly an array of tasks
        assert isinstance(data, list)

    def test_get_nonexistent_task(self, client):
        """Test getting a task that doesn't exist."""
        response = client.get("/tasks/nonexistent-id")
        assert response.status_code == 404

    def test_cancel_nonexistent_task(self, client):
        """Test canceling a task that doesn't exist."""
        response = client.post("/tasks/nonexistent-id/cancel")
        assert response.status_code == 404

    def test_run_endpoint_validation(self, client):
        """Test run endpoint validates request body."""
        response = client.post("/run", json={})
        # Should fail validation - missing 'request' field
        assert response.status_code == 422


# =============================================================================
# WebSocket Integration Tests
# =============================================================================


class TestWebSocketIntegration:
    """Test WebSocket real-time communication."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_websocket_connection(self, client):
        """Test WebSocket can connect."""
        with client.websocket_connect("/ws") as websocket:
            # Connection should succeed
            assert websocket is not None

    def test_websocket_subscribe(self, client):
        """Test subscribing to task updates."""
        with client.websocket_connect("/ws") as websocket:
            # Subscribe to a task
            websocket.send_json({"type": "subscribe", "task_id": "test-task-123"})
            # Should handle without crashing

    def test_websocket_invalid_message(self, client):
        """Test WebSocket handles invalid messages."""
        with client.websocket_connect("/ws") as websocket:
            # Send invalid JSON structure
            websocket.send_json({"invalid": "message"})
            # Should not crash - connection stays open


# =============================================================================
# Agent Model Tests
# =============================================================================


class TestAgentModels:
    """Test agent models and state management."""

    def test_agent_state_creation(self):
        """Test AgentState can be created."""
        from agent.orchestrator.react_agent import AgentState

        state = AgentState(goal="Test goal")
        assert state.goal == "Test goal"
        assert state.status == "running"
        assert state.steps == []

    def test_agent_step_creation(self):
        """Test AgentStep can be created."""
        from agent.orchestrator.react_agent import AgentStep, Observation, Thought

        obs = Observation(source="test", content="Test observation")
        thought = Thought(reasoning="Test reasoning")
        step = AgentStep(iteration=1, observation=obs, thought=thought)

        assert step.iteration == 1
        assert step.observation.content == "Test observation"
        assert step.thought.reasoning == "Test reasoning"

    def test_action_creation(self):
        """Test Action can be created."""
        from agent.orchestrator.react_agent import Action

        action = Action(tool="shell", args={"command": "ls -la"})
        assert action.tool == "shell"
        assert action.args["command"] == "ls -la"

    def test_step_result_creation(self):
        """Test StepResult can be created."""
        from agent.orchestrator.react_agent import StepResult

        result = StepResult(step_id="test", status="success", output="Test output")
        assert result.step_id == "test"
        assert result.status == "success"
        assert result.output == "Test output"


# =============================================================================
# Real E2E Tests (require actual shell)
# =============================================================================


@pytest.mark.e2e
class TestRealE2E:
    """End-to-end tests with real execution.

    These tests actually execute commands on the system.
    Run with: pytest -m e2e
    """

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a real temporary workspace."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_real_python_file_creation(self, workspace):
        """Test Python can actually create files."""
        sandbox = Sandbox(timeout=30, permissive=True)
        output_file = workspace / "output.txt"

        code = f"""
with open('{output_file}', 'w') as f:
    f.write('Created by Python agent')
print('File created!')
"""
        result = await sandbox.run_python(code, working_dir=str(workspace))

        assert "output" in result
        assert output_file.exists()
        assert "Created by Python agent" in output_file.read_text()

    @pytest.mark.asyncio
    async def test_real_python_reads_file(self, workspace):
        """Test Python can read existing files."""
        sandbox = Sandbox(timeout=30, permissive=True)

        # Create a file first
        test_file = workspace / "test.txt"
        test_file.write_text("Hello from test!")

        code = f"""
with open('{test_file}', 'r') as f:
    content = f.read()
print(f"Read: {{content}}")
"""
        result = await sandbox.run_python(code, working_dir=str(workspace))

        assert "output" in result
        assert "Hello from test!" in result["output"]

    @pytest.mark.asyncio
    async def test_real_python_directory_listing(self, workspace):
        """Test Python can list directory contents."""
        sandbox = Sandbox(timeout=30, permissive=True)

        # Create some files
        (workspace / "file1.txt").write_text("content1")
        (workspace / "file2.py").write_text("content2")
        subdir = workspace / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")

        code = f"""
import os
for item in os.listdir('{workspace}'):
    print(item)
"""
        result = await sandbox.run_python(code, working_dir=str(workspace))

        assert "output" in result
        assert "file1.txt" in result["output"]
        assert "file2.py" in result["output"]
        assert "subdir" in result["output"]
