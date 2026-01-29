from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class TaskState(str, Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Step(BaseModel):
    id: str
    description: str | None = None
    action: str
    args: dict[str, Any] = {}
    depends_on: list[str] = []


class Plan(BaseModel):
    steps: list[Step]

    @property
    def is_chat(self) -> bool:
        """Check if this plan is just a chat response (no tools needed)."""
        return len(self.steps) == 1 and self.steps[0].action == "chat_op"

    @property
    def chat_response(self) -> str:
        """Get the chat response if this is a chat plan."""
        if self.is_chat:
            return self.steps[0].args.get("response", "")
        return ""


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class TaskRequest(BaseModel):
    request: str
    session_id: str | None = None  # For multi-turn conversations
    auto_approve: bool = False  # Skip approval step if True


class StepResult(BaseModel):
    step_id: str
    status: str
    output: Any | None = None
    error: str | None = None


class TaskSummary(BaseModel):
    """Lightweight task info for listing."""

    id: str
    request: str
    state: TaskState
    created_at: datetime
    step_count: int = 0
    completed_steps: int = 0
    error: str | None = None


class TaskDetail(BaseModel):
    """Full task information."""

    id: str
    request: str
    session_id: str | None = None
    state: TaskState
    created_at: datetime
    updated_at: datetime
    plan: Plan | None = None
    step_results: dict[str, StepResult] = {}
    current_step: str | None = None
    summary: str | None = None
    error: str | None = None
    workspace_path: str | None = None
    workspace_files: dict[str, list[str]] | None = None


class TaskResponse(BaseModel):
    task_id: str
    plan: Plan
    results: dict[str, StepResult]


class WSMessageType(str, Enum):
    """WebSocket message types."""

    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBED = "subscribed"
    TASK_UPDATE = "task_update"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    STEP_OUTPUT = "step_output"
    ERROR = "error"
    # Streaming message types
    STREAM_START = "stream_start"
    STREAM_TOKEN = "stream_token"
    STREAM_END = "stream_end"
    STREAM_THOUGHT = "stream_thought"
    STREAM_ACTION = "stream_action"


class WebSocketMessage(BaseModel):
    """Message format for WebSocket communication."""

    type: WSMessageType
    task_id: str | None = None
    data: dict[str, Any] = {}

    @classmethod
    def subscribe(cls, task_id: str) -> "WebSocketMessage":
        return cls(type=WSMessageType.SUBSCRIBE, task_id=task_id)

    @classmethod
    def pong(cls) -> "WebSocketMessage":
        return cls(type=WSMessageType.PONG)

    @classmethod
    def subscribed(cls, task_id: str) -> "WebSocketMessage":
        return cls(type=WSMessageType.SUBSCRIBED, task_id=task_id)

    @classmethod
    def task_update(cls, task_id: str, data: dict[str, Any]) -> "WebSocketMessage":
        return cls(type=WSMessageType.TASK_UPDATE, task_id=task_id, data=data)

    @classmethod
    def step_output(cls, task_id: str, step: str, output: Any) -> "WebSocketMessage":
        return cls(
            type=WSMessageType.STEP_OUTPUT,
            task_id=task_id,
            data={"step": step, "output": output},
        )

    @classmethod
    def task_complete(cls, task_id: str, summary: str) -> "WebSocketMessage":
        return cls(
            type=WSMessageType.TASK_COMPLETE, task_id=task_id, data={"summary": summary}
        )

    @classmethod
    def task_error(cls, task_id: str, error: str) -> "WebSocketMessage":
        return cls(
            type=WSMessageType.TASK_ERROR, task_id=task_id, data={"error": error}
        )

    @classmethod
    def error(cls, message: str) -> "WebSocketMessage":
        return cls(type=WSMessageType.ERROR, data={"message": message})

    @classmethod
    def stream_start(
        cls, task_id: str, stream_type: str = "response"
    ) -> "WebSocketMessage":
        """Signal the start of a streaming response."""
        return cls(
            type=WSMessageType.STREAM_START,
            task_id=task_id,
            data={"stream_type": stream_type},
        )

    @classmethod
    def stream_token(cls, task_id: str, token: str) -> "WebSocketMessage":
        """Send a single token in a stream."""
        return cls(
            type=WSMessageType.STREAM_TOKEN,
            task_id=task_id,
            data={"token": token},
        )

    @classmethod
    def stream_end(
        cls, task_id: str, full_response: str | None = None
    ) -> "WebSocketMessage":
        """Signal the end of a streaming response."""
        return cls(
            type=WSMessageType.STREAM_END,
            task_id=task_id,
            data={"full_response": full_response} if full_response else {},
        )

    @classmethod
    def stream_thought(
        cls, task_id: str, thought: str, iteration: int
    ) -> "WebSocketMessage":
        """Stream agent's thinking/reasoning."""
        return cls(
            type=WSMessageType.STREAM_THOUGHT,
            task_id=task_id,
            data={"thought": thought, "iteration": iteration},
        )

    @classmethod
    def stream_action(
        cls, task_id: str, tool: str, args: dict[str, Any]
    ) -> "WebSocketMessage":
        """Stream agent's action decision."""
        return cls(
            type=WSMessageType.STREAM_ACTION,
            task_id=task_id,
            data={"tool": tool, "args": args},
        )
