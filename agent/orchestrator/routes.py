"""REST API routes for the LocalCowork server.

This module contains all HTTP endpoints for task management and execution.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse

from agent.config import settings
from agent.llm.client import LLMError
from agent.orchestrator.deps import get_sandbox, get_task_manager
from agent.orchestrator.middleware import check_rate_limit, verify_api_key
from agent.orchestrator.models import TaskDetail, TaskRequest, TaskState, TaskSummary
from agent.orchestrator.session import add_message, get_history
from agent.orchestrator.task_manager import TaskState as TMState

logger = logging.getLogger(__name__)

# Shared resources
sandbox = get_sandbox()
task_manager = get_task_manager()

# Static files
STATIC_DIR = Path(__file__).parent / "static"


async def root():
    """Serve web UI."""
    return FileResponse(STATIC_DIR / "index.html")


async def run_task(
    request: TaskRequest,
    req: Request,
    ws_manager,  # Passed from server.py
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
):
    """
    Run a task using the ReAct agent.

    The agent uses shell commands and Python to accomplish tasks,
    adapting step-by-step based on results.
    """
    from agent.orchestrator.react_agent import ReActAgent

    session_id = request.session_id or str(uuid.uuid4())
    add_message(session_id, "user", request.request)

    # Create task
    task = task_manager.create_task(request.request, session_id)
    task_manager.update_state(task.id, TMState.EXECUTING)

    async def on_progress(iteration: int, status: str, thought: str, action: str):
        await ws_manager.broadcast(
            task.id,
            {
                "type": "progress",
                "task_id": task.id,
                "iteration": iteration,
                "status": status,
                "thought": thought,
                "action": action,
            },
        )

    try:
        history = get_history(session_id)
        conv = [{"role": m.role, "content": m.content} for m in history]

        # For server mode, confirmations are handled via WebSocket
        pending_confirmations: dict[str, asyncio.Event] = {}
        confirmation_results: dict[str, bool] = {}

        async def on_confirm(command: str, reason: str, message: str) -> bool:
            """Request confirmation via WebSocket and wait for response."""
            confirm_id = str(uuid.uuid4())
            event = asyncio.Event()
            pending_confirmations[confirm_id] = event

            # Send confirmation request to connected clients
            await ws_manager.broadcast(
                task.id,
                {
                    "type": "confirm_request",
                    "confirm_id": confirm_id,
                    "task_id": task.id,
                    "command": command[:200],
                    "reason": reason,
                    "message": message,
                },
            )

            # Wait for response (timeout after 60 seconds)
            try:
                await asyncio.wait_for(event.wait(), timeout=60.0)
                return confirmation_results.get(confirm_id, False)
            except TimeoutError:
                logger.warning(f"Confirmation timeout for {confirm_id}")
                return False  # Default to deny on timeout
            finally:
                pending_confirmations.pop(confirm_id, None)
                confirmation_results.pop(confirm_id, None)

        agent = ReActAgent(
            sandbox=sandbox,
            on_progress=on_progress,
            on_confirm=on_confirm,
            max_iterations=settings.max_agent_iterations,
            conversation_history=conv,
            require_confirmation=True,
        )

        state = await agent.run(request.request)

        if state.status == "completed":
            task_manager.update_state(task.id, TMState.COMPLETED)
            task_manager.set_summary(task.id, state.final_answer or "Done")
        else:
            task_manager.update_state(task.id, TMState.FAILED, state.error)

        if state.final_answer:
            add_message(session_id, "assistant", state.final_answer)

        await ws_manager.broadcast(
            task.id,
            {
                "type": "complete",
                "task_id": task.id,
                "status": state.status,
                "response": state.final_answer,
            },
        )

        return {
            "task_id": task.id,
            "session_id": session_id,
            "status": state.status,
            "response": state.final_answer,
            "steps": len(state.steps),
        }

    except LLMError as e:
        task_manager.update_state(task.id, TMState.FAILED, str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Agent failed")
        task_manager.update_state(task.id, TMState.FAILED, str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def list_tasks(
    session_id: str | None = None,
    state: str | None = None,
    limit: int = 50,
    _auth: bool = Depends(verify_api_key),
) -> list[TaskSummary]:
    """List tasks."""
    states = None
    if state:
        try:
            states = [TMState(state)]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    tasks = task_manager.get_tasks(session_id=session_id, states=states, limit=limit)
    return [_to_summary(t) for t in tasks]


async def get_task(task_id: str, _auth: bool = Depends(verify_api_key)) -> TaskDetail:
    """Get task details."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _to_detail(task)


async def cancel_task(task_id: str, ws_manager, _auth: bool = Depends(verify_api_key)):
    """Cancel a task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.state in {TMState.COMPLETED, TMState.FAILED, TMState.CANCELLED}:
        raise HTTPException(status_code=400, detail=f"Task already {task.state.value}")

    task_manager.update_state(task_id, TMState.CANCELLED)
    await ws_manager.broadcast(task_id, {"type": "cancelled", "task_id": task_id})
    return {"status": "cancelled", "task_id": task_id}


async def health():
    """Health check."""
    return {"status": "ok", "version": "0.3.0"}


# =============================================================================
# Helpers
# =============================================================================


def _to_summary(task) -> TaskSummary:
    return TaskSummary(
        id=task.id,
        request=task.request,
        state=TaskState(task.state.value),
        created_at=task.created_at,
        step_count=0,
        completed_steps=0,
        error=task.error,
    )


def _to_detail(task) -> TaskDetail:
    return TaskDetail(
        id=task.id,
        request=task.request,
        session_id=task.session_id,
        state=TaskState(task.state.value),
        created_at=task.created_at,
        updated_at=task.updated_at,
        plan=None,
        step_results={},
        current_step=task.current_step,
        summary=task.summary,
        error=task.error,
        workspace_path=task.workspace_path,
        workspace_files=task_manager.list_workspace_files(task.id),
    )
