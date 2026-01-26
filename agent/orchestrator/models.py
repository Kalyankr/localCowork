from pydantic import BaseModel, Field
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


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class TaskRequest(BaseModel):
    request: str
    session_id: Optional[str] = None  # For multi-turn conversations
    auto_approve: bool = False  # Skip approval step if True


class TaskApproval(BaseModel):
    """Request to approve or reject a pending task."""
    task_id: str
    approved: bool
    feedback: Optional[str] = None  # Optional feedback if rejected


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


class WebSocketMessage(BaseModel):
    """Message format for WebSocket communication."""
    type: str  # subscribe, unsubscribe, approve, reject, cancel
    task_id: Optional[str] = None
    data: Dict[str, Any] = {}
