from pydantic import BaseModel
from typing import List, Dict, Any, Optional


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


class StepResult(BaseModel):
    step_id: str
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    plan: Plan
    results: Dict[str, StepResult]
