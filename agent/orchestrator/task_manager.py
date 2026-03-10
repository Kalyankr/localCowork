"""Task Manager for tracking task lifecycle, history, and persistence."""

import contextlib
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from agent.config import settings

logger = structlog.get_logger(__name__)


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


def _now_utc() -> datetime:
    """Get current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class TaskEvent(BaseModel):
    """Event emitted during task lifecycle for real-time streaming."""

    task_id: str
    type: str  # state_change, step_progress, step_complete, error, log
    timestamp: datetime = Field(default_factory=_now_utc)
    data: dict[str, Any] = {}


class Task(BaseModel):
    """Represents a task with full lifecycle tracking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: str
    session_id: str | None = None
    state: TaskState = TaskState.PENDING
    created_at: datetime = Field(default_factory=_now_utc)
    updated_at: datetime = Field(default_factory=_now_utc)

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
        self.updated_at = _now_utc()


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
        self._db_path = settings.db_path

        # In-memory task cache (loaded from SQLite on init)
        self._tasks: dict[str, Task] = {}

        # Event subscribers
        self._subscribers: dict[str, list[EventCallback]] = {}  # task_id -> callbacks
        self._global_subscribers: list[EventCallback] = []

        # Ensure directories exist
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Load from SQLite synchronously (startup)
        self._load_from_db()

    def _load_from_db(self):
        """Load task history from SQLite database."""
        from agent.orchestrator.database import get_sync_connection

        try:
            conn = get_sync_connection(self._db_path)
            cursor = conn.execute("SELECT * FROM tasks")
            rows = cursor.fetchall()
            for row in rows:
                data = dict(row)
                # Deserialize JSON fields
                if data.get("plan"):
                    data["plan"] = json.loads(data["plan"])
                if data.get("step_results"):
                    data["step_results"] = json.loads(data["step_results"])
                else:
                    data["step_results"] = {}
                task = Task(**data)
                self._tasks[task.id] = task
            conn.close()
            logger.info("task_history_loaded", count=len(self._tasks), source="sqlite")
        except Exception as e:
            logger.warning("task_history_load_failed", error=str(e))
            # Fall back to JSON if DB fails (migration path)
            self._load_history_json()

    def _load_history_json(self):
        """Legacy: load task history from JSON file (migration fallback)."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    for task_data in data.get("tasks", []):
                        task = Task(**task_data)
                        self._tasks[task.id] = task
                logger.info(
                    "task_history_loaded",
                    count=len(self._tasks),
                    source="json_fallback",
                )
            except Exception as e:
                logger.warning("json_history_load_failed", error=str(e))

    def _persist_task(self, task: Task):
        """Persist a single task to SQLite synchronously."""
        from agent.orchestrator.database import get_sync_connection

        try:
            data = task.model_dump(mode="json")
            conn = get_sync_connection(self._db_path)
            conn.execute(
                """INSERT OR REPLACE INTO tasks
                   (id, request, session_id, state, created_at, updated_at,
                    plan, step_results, current_step, summary, error, workspace_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["id"],
                    data["request"],
                    data.get("session_id"),
                    data["state"],
                    data["created_at"],
                    data["updated_at"],
                    json.dumps(data.get("plan")) if data.get("plan") else None,
                    json.dumps(data.get("step_results", {})),
                    data.get("current_step"),
                    data.get("summary"),
                    data.get("error"),
                    data.get("workspace_path"),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("task_persist_failed", task_id=task.id, error=str(e))

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
        self._persist_task(task)
        self._emit_event(task, "task_created")

        logger.info("task_created", task_id=task.id, request=request[:50])
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
        # Handle both timezone-aware and naive datetimes for backward compatibility
        def sort_key(t: Task) -> datetime:
            if t.created_at.tzinfo is None:
                # Convert naive datetime to UTC for comparison
                return t.created_at.replace(tzinfo=UTC)
            return t.created_at

        tasks.sort(key=sort_key, reverse=True)

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
            logger.warning("task_not_found", task_id=task_id)
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

        # Persist state change
        self._persist_task(task)

        logger.info(
            "task_state_change",
            task_id=task_id,
            old_state=old_state.value,
            new_state=new_state.value,
        )

    def set_plan(self, task_id: str, plan: dict[str, Any]):
        """Set the task's plan after generation."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.plan = plan
        task.touch()
        self._persist_task(task)

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
        self._persist_task(task)

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
        self._persist_task(task)

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
        self._persist_task(task)

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
                logger.error("event_callback_error", error=str(e))

        # Notify global subscribers
        for callback in self._global_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error("global_event_callback_error", error=str(e))

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

        cutoff = _now_utc() - timedelta(days=max_age_days)

        for task in list(self._tasks.values()):
            if task.created_at < cutoff:
                workspace = self.get_workspace_path(task.id)
                if workspace and workspace.exists():
                    try:
                        import shutil

                        shutil.rmtree(workspace)
                        logger.info("workspace_cleaned", task_id=task.id)
                    except Exception as e:
                        logger.error(
                            "workspace_cleanup_failed", task_id=task.id, error=str(e)
                        )


# Singleton instance
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get the singleton TaskManager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
