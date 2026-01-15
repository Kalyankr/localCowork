from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Step(BaseModel):
    id: str
    description: str
    action: str
    args: Dict[str, Any] = {}
    depends_on: List[str] = []


class Plan(BaseModel):
    steps: List[Step]


class TaskRequest(BaseModel):
    request: str


class StepResult(BaseModel):
    step_id: str
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    plan: Plan
    results: Dict[str, StepResult]
