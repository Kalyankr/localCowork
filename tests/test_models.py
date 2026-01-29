"""Unit tests for models module."""

from agent.orchestrator.models import (
    ConversationMessage,
    Plan,
    Step,
    StepResult,
    TaskRequest,
    TaskResponse,
)


class TestStep:
    """Tests for Step model."""

    def test_step_minimal(self):
        """Should create step with minimal required fields."""
        step = Step(id="test_step", action="file_op")

        assert step.id == "test_step"
        assert step.action == "file_op"
        assert step.args == {}
        assert step.depends_on == []

    def test_step_full(self):
        """Should create step with all fields."""
        step = Step(
            id="move_files",
            description="Move files to destination",
            action="file_op",
            args={"op": "move", "src": "~/Downloads", "dest": "~/Documents"},
            depends_on=["list_files"],
        )

        assert step.description == "Move files to destination"
        assert step.args["op"] == "move"
        assert "list_files" in step.depends_on


class TestPlan:
    """Tests for Plan model."""

    def test_plan_empty(self):
        """Should allow empty steps list."""
        plan = Plan(steps=[])
        assert plan.steps == []

    def test_plan_with_steps(self):
        """Should create plan with multiple steps."""
        plan = Plan(
            steps=[
                Step(id="step1", action="file_op"),
                Step(id="step2", action="python", depends_on=["step1"]),
            ]
        )

        assert len(plan.steps) == 2
        assert plan.steps[1].depends_on == ["step1"]


class TestConversationMessage:
    """Tests for ConversationMessage model."""

    def test_user_message(self):
        """Should create user message."""
        msg = ConversationMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        """Should create assistant message."""
        msg = ConversationMessage(role="assistant", content="Hi there!")

        assert msg.role == "assistant"


class TestTaskRequest:
    """Tests for TaskRequest model."""

    def test_request_minimal(self):
        """Should create request with just the request string."""
        req = TaskRequest(request="Organize my files")

        assert req.request == "Organize my files"
        assert req.session_id is None

    def test_request_with_session(self):
        """Should create request with session ID."""
        req = TaskRequest(request="Continue", session_id="abc-123")

        assert req.session_id == "abc-123"


class TestStepResult:
    """Tests for StepResult model."""

    def test_success_result(self):
        """Should create successful result."""
        result = StepResult(step_id="test", status="success", output={"moved": 5})

        assert result.status == "success"
        assert result.output == {"moved": 5}
        assert result.error is None

    def test_error_result(self):
        """Should create error result."""
        result = StepResult(step_id="test", status="error", error="File not found")

        assert result.status == "error"
        assert result.error == "File not found"


class TestTaskResponse:
    """Tests for TaskResponse model."""

    def test_response_full(self):
        """Should create complete response."""
        response = TaskResponse(
            task_id="task-123",
            plan=Plan(steps=[Step(id="s1", action="file_op")]),
            results={"s1": StepResult(step_id="s1", status="success")},
        )

        assert response.task_id == "task-123"
        assert len(response.plan.steps) == 1
        assert "s1" in response.results
