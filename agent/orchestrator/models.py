from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


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
    description: Optional[str] = None
    action: str
    args: Dict[str, Any] = {}
    depends_on: List[str] = []


class Plan(BaseModel):
    steps: List[Step]

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
    session_id: Optional[str] = None  # For multi-turn conversations
    auto_approve: bool = False  # Skip approval step if True


class StepResult(BaseModel):
    step_id: str
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None


class TaskSummary(BaseModel):
    """Lightweight task info for listing."""

    id: str
    request: str
    state: TaskState
    created_at: datetime
    step_count: int = 0
    completed_steps: int = 0
    error: Optional[str] = None


class TaskDetail(BaseModel):
    """Full task information."""

    id: str
    request: str
    session_id: Optional[str] = None
    state: TaskState
    created_at: datetime
    updated_at: datetime
    plan: Optional[Plan] = None
    step_results: Dict[str, StepResult] = {}
    current_step: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    workspace_path: Optional[str] = None
    workspace_files: Optional[Dict[str, List[str]]] = None


class TaskResponse(BaseModel):
    task_id: str
    plan: Plan
    results: Dict[str, StepResult]


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


class WebSocketMessage(BaseModel):
    """Message format for WebSocket communication."""

    type: WSMessageType
    task_id: Optional[str] = None
    data: Dict[str, Any] = {}

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
    def task_update(cls, task_id: str, data: Dict[str, Any]) -> "WebSocketMessage":
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
