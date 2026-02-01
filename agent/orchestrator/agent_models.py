"""Agent state models for the ReAct agent.

This module contains the Pydantic models that represent the agent's
internal state during execution.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent.orchestrator.models import StepResult


class Observation(BaseModel):
    """What the agent observes from the environment."""

    source: str  # "tool", "error", "initial", "reflection"
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)


class Thought(BaseModel):
    """The agent's reasoning about what to do."""

    reasoning: str  # Why the agent is taking this action
    confidence: float = 1.0  # 0-1 confidence score
    is_goal_complete: bool = False  # Whether the agent thinks the goal is done


class Action(BaseModel):
    """An action the agent decides to take."""

    tool: str  # Tool name (e.g., "file_op", "python", "done")
    args: dict[str, Any] = {}
    description: str = ""  # Human-readable description


class AgentStep(BaseModel):
    """A single step in the agent's execution."""

    iteration: int
    observation: Observation
    thought: Thought
    action: Action | None = None
    result: StepResult | None = None


class AgentState(BaseModel):
    """The full state of an agent execution."""

    goal: str
    status: str = "running"  # running, completed, failed, max_iterations
    steps: list[AgentStep] = []
    context: dict[str, Any] = {}  # Variables from tool outputs
    final_answer: str | None = None
    error: str | None = None
    # Sub-agent tracking
    is_sub_agent: bool = False
    parent_goal: str | None = None
    sub_agent_results: list[dict[str, Any]] = []


class SubTask(BaseModel):
    """A subtask that can be executed by a sub-agent."""

    id: str
    description: str
    dependencies: list[str] = []  # IDs of tasks this depends on
    status: str = "pending"  # pending, running, completed, failed
    result: str | None = None
    error: str | None = None


class SubAgentState(BaseModel):
    """State for coordinating multiple sub-agents."""

    main_goal: str
    subtasks: list[SubTask] = []
    completed_subtasks: dict[str, Any] = {}  # id -> result
    status: str = "planning"  # planning, executing, merging, completed, failed


# Type for progress callback
ProgressCallback = Callable[
    [int, str, str, str | None], None
]  # iteration, status, thought, action

# Type for confirmation callback (for dangerous operations)
# Returns True if user confirms, False to cancel
ConfirmCallback = Callable[
    [str, str, str], Awaitable[bool]
]  # command, reason, message -> confirmed
