"""Task Manager for tracking task lifecycle, history, and persistence."""

import contextlib
import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.config import settings

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    """Task lifecycle states."""

    PENDING = "pending"  # Just created
    PLANNING = "planning"  # LLM is generating plan
    AWAITING_APPROVAL = "awaiting_approval"  # Plan ready, waiting for user approval
    APPROVED = "approved"  # User approved, ready to execute
    REJECTED = "rejected"  # User rejected the plan
    EXECUTING = "executing"  # Steps are being executed
    COMPLETED = "completed"  # All steps finished successfully
    FAILED = "failed"  # One or more steps failed
    CANCELLED = "cancelled"  # User cancelled mid-execution


class TaskEvent(BaseModel):
    """Event emitted during task lifecycle for real-time streaming."""

    task_id: str
    type: str  # state_change, step_progress, step_complete, error, log
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = {}


class Task(BaseModel):
    """Represents a task with full lifecycle tracking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: str
    session_id: str | None = None
    state: TaskState = TaskState.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Plan info (populated after planning)
    plan: dict[str, Any] | None = None

    # Execution info (populated during/after execution)
    step_results: dict[str, dict[str, Any]] = {}
    current_step: str | None = None

    # Final output
    summary: str | None = None
    error: str | None = None

    # Workspace
    workspace_path: str | None = None

    def touch(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


# Type alias for event callbacks
EventCallback = Callable[[TaskEvent], None]


class TaskManager:
    """Manages task lifecycle, persistence, and event broadcasting."""

    def __init__(
        self,
        history_file: Path | None = None,
        workspace_root: Path | None = None,
    ):
        self.history_file = history_file or Path(settings.history_path)
        self.workspace_root = workspace_root or Path(settings.workspace_path)

        # In-memory task storage
        self._tasks: dict[str, Task] = {}

        # Event subscribers
        self._subscribers: dict[str, list[EventCallback]] = {}  # task_id -> callbacks
        self._global_subscribers: list[EventCallback] = []

        # Ensure directories exist
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Load history on startup
        self._load_history()

    def _load_history(self):
        """Load task history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    for task_data in data.get("tasks", []):
                        task = Task(**task_data)
                        self._tasks[task.id] = task
                logger.info(f"Loaded {len(self._tasks)} tasks from history")
            except Exception as e:
                logger.warning(f"Failed to load task history: {e}")

    def _save_history(self):
        """Save task history to disk."""
        try:
            # Only save completed/failed tasks (not transient ones)
            completed_states = {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.REJECTED,
                TaskState.CANCELLED,
            }
            tasks_to_save = [
                t.model_dump(mode="json")
                for t in self._tasks.values()
                if t.state in completed_states
            ]

            with open(self.history_file, "w") as f:
                json.dump({"tasks": tasks_to_save}, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save task history: {e}")

    def create_task(self, request: str, session_id: str | None = None) -> Task:
        """Create a new task and set up its workspace."""
        task = Task(request=request, session_id=session_id)

        # Create workspace directory
        workspace = self.workspace_root / task.id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "input").mkdir(exist_ok=True)
        (workspace / "output").mkdir(exist_ok=True)
        task.workspace_path = str(workspace)

        self._tasks[task.id] = task
        self._emit_event(task, "task_created")

        logger.info(f"Created task {task.id}: {request[:50]}...")
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_tasks(
        self,
        session_id: str | None = None,
        states: list[TaskState] | None = None,
        limit: int = 50,
    ) -> list[Task]:
        """Get tasks, optionally filtered by session or state."""
        tasks = list(self._tasks.values())

        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]

        if states:
            tasks = [t for t in tasks if t.state in states]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def update_state(
        self,
        task_id: str,
        new_state: TaskState,
        error: str | None = None,
    ):
        """Update task state and emit event."""
        task = self._tasks.get(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found")
            return

        old_state = task.state
        task.state = new_state
        task.touch()

        if error:
            task.error = error

        self._emit_event(
            task,
            "state_change",
            {
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )

        # Persist on terminal states
        if new_state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.REJECTED,
            TaskState.CANCELLED,
        }:
            self._save_history()

        logger.info(f"Task {task_id}: {old_state.value} -> {new_state.value}")

    def set_plan(self, task_id: str, plan: dict[str, Any]):
        """Set the task's plan after generation."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.plan = plan
        task.touch()

        self._emit_event(task, "plan_ready", {"plan": plan})

    def update_step_progress(
        self,
        task_id: str,
        step_id: str,
        status: str,
        current: int,
        total: int,
    ):
        """Update progress on a specific step."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.current_step = step_id
        task.touch()

        self._emit_event(
            task,
            "step_progress",
            {
                "step_id": step_id,
                "status": status,
                "current": current,
                "total": total,
            },
        )

    def set_step_result(
        self,
        task_id: str,
        step_id: str,
        result: dict[str, Any],
    ):
        """Record a step's result."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.step_results[step_id] = result
        task.touch()

        self._emit_event(
            task,
            "step_complete",
            {
                "step_id": step_id,
                "result": result,
            },
        )

    def set_summary(self, task_id: str, summary: str):
        """Set the final summary for a task."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.summary = summary
        task.touch()

        self._emit_event(task, "summary", {"summary": summary})

    def subscribe(
        self,
        callback: EventCallback,
        task_id: str | None = None,
    ):
        """Subscribe to task events."""
        if task_id:
            if task_id not in self._subscribers:
                self._subscribers[task_id] = []
            self._subscribers[task_id].append(callback)
        else:
            self._global_subscribers.append(callback)

    def unsubscribe(
        self,
        callback: EventCallback,
        task_id: str | None = None,
    ):
        """Unsubscribe from task events."""
        if task_id and task_id in self._subscribers:
            with contextlib.suppress(ValueError):
                self._subscribers[task_id].remove(callback)
        else:
            with contextlib.suppress(ValueError):
                self._global_subscribers.remove(callback)

    def _emit_event(
        self,
        task: Task,
        event_type: str,
        data: dict[str, Any] | None = None,
    ):
        """Emit an event to all subscribers."""
        event = TaskEvent(
            task_id=task.id,
            type=event_type,
            data=data or {},
        )

        # Notify task-specific subscribers
        for callback in self._subscribers.get(task.id, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        # Notify global subscribers
        for callback in self._global_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Global event callback error: {e}")

    def get_workspace_path(self, task_id: str) -> Path | None:
        """Get the workspace path for a task."""
        task = self._tasks.get(task_id)
        if task and task.workspace_path:
            return Path(task.workspace_path)
        return None

    def list_workspace_files(self, task_id: str) -> dict[str, list[str]]:
        """List files in a task's workspace."""
        workspace = self.get_workspace_path(task_id)
        if not workspace or not workspace.exists():
            return {"input": [], "output": []}

        def list_files(path: Path) -> list[str]:
            if not path.exists():
                return []
            return [str(f.relative_to(path)) for f in path.rglob("*") if f.is_file()]

        return {
            "input": list_files(workspace / "input"),
            "output": list_files(workspace / "output"),
        }

    def cleanup_old_workspaces(self, max_age_days: int = 7):
        """Clean up old task workspaces."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=max_age_days)

        for task in list(self._tasks.values()):
            if task.created_at < cutoff:
                workspace = self.get_workspace_path(task.id)
                if workspace and workspace.exists():
                    try:
                        import shutil

                        shutil.rmtree(workspace)
                        logger.info(f"Cleaned up workspace for task {task.id}")
                    except Exception as e:
                        logger.error(f"Failed to clean workspace {task.id}: {e}")


# Singleton instance
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get the singleton TaskManager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
