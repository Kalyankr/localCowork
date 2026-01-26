"""LocalCowork API Server with WebSocket support, approval flow, and task history."""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from agent.orchestrator.models import (
    TaskRequest, TaskResponse, TaskApproval, TaskSummary, TaskDetail,
    ConversationMessage, Plan, StepResult, TaskState,
)
from agent.orchestrator.planner import generate_plan, summarize_results
from agent.orchestrator.executor import Executor
from agent.orchestrator.deps import get_tool_registry, get_sandbox, get_task_manager
from agent.orchestrator.task_manager import TaskState as TMState, TaskEvent
from agent.config import settings
from agent.llm.client import LLMError
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LocalCowork API",
    description="AI-powered local task automation with approval workflows",
    version="0.2.0",
)

# Use shared dependencies
tool_registry = get_tool_registry()
sandbox = get_sandbox()
task_manager = get_task_manager()

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Conversation history storage (in-memory, keyed by session_id)
conversation_history: Dict[str, List[ConversationMessage]] = defaultdict(list)
conversation_timestamps: Dict[str, float] = {}

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # All active connections
        self.active_connections: Set[WebSocket] = set()
        # Task-specific subscriptions: task_id -> set of websockets
        self.task_subscriptions: Dict[str, Set[WebSocket]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        # Remove from all task subscriptions
        for subscribers in self.task_subscriptions.values():
            subscribers.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    def subscribe_to_task(self, websocket: WebSocket, task_id: str):
        self.task_subscriptions[task_id].add(websocket)
    
    def unsubscribe_from_task(self, websocket: WebSocket, task_id: str):
        self.task_subscriptions[task_id].discard(websocket)
    
    async def broadcast_to_task(self, task_id: str, message: dict):
        """Send message to all subscribers of a task."""
        subscribers = self.task_subscriptions.get(task_id, set())
        dead_connections = set()
        
        for websocket in subscribers:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.add(websocket)
        
        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(ws)
    
    async def broadcast_all(self, message: dict):
        """Broadcast to all connected clients."""
        dead_connections = set()
        
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.add(websocket)
        
        for ws in dead_connections:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# Session timeout from config
SESSION_TIMEOUT = settings.session_timeout
MAX_HISTORY_MESSAGES = settings.max_history_messages


def cleanup_old_sessions():
    """Remove sessions older than SESSION_TIMEOUT."""
    current_time = time.time()
    expired = [
        sid for sid, ts in conversation_timestamps.items()
        if current_time - ts > SESSION_TIMEOUT
    ]
    for sid in expired:
        conversation_history.pop(sid, None)
        conversation_timestamps.pop(sid, None)


def get_session_history(session_id: str) -> List[ConversationMessage]:
    """Get conversation history for a session."""
    cleanup_old_sessions()
    return conversation_history.get(session_id, [])


def add_to_history(session_id: str, role: str, content: str):
    """Add a message to conversation history."""
    conversation_history[session_id].append(
        ConversationMessage(role=role, content=content)
    )
    conversation_timestamps[session_id] = time.time()
    
    if len(conversation_history[session_id]) > MAX_HISTORY_MESSAGES:
        conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY_MESSAGES:]


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates."""
    await ws_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            task_id = data.get("task_id")
            
            if msg_type == "subscribe" and task_id:
                ws_manager.subscribe_to_task(websocket, task_id)
                await websocket.send_json({
                    "type": "subscribed",
                    "task_id": task_id,
                })
            
            elif msg_type == "unsubscribe" and task_id:
                ws_manager.unsubscribe_from_task(websocket, task_id)
                await websocket.send_json({
                    "type": "unsubscribed", 
                    "task_id": task_id,
                })
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/tasks/create")
async def create_task(request: TaskRequest):
    """Create a new task and generate a plan (awaits approval by default)."""
    session_id = request.session_id or str(uuid.uuid4())
    
    # Create task in manager
    task = task_manager.create_task(request.request, session_id)
    
    # Update state to planning
    task_manager.update_state(task.id, TMState.PLANNING)
    
    # Add user message to history
    add_to_history(session_id, "user", request.request)
    
    try:
        # Get conversation history for context
        history = get_session_history(session_id)
        
        # Generate plan
        plan = generate_plan(request.request, history=history if history else None)
        
        # Save plan
        task_manager.set_plan(task.id, plan.model_dump())
        
        # Determine next state based on approval mode
        if request.auto_approve or not settings.require_approval:
            task_manager.update_state(task.id, TMState.APPROVED)
            # Auto-execute
            asyncio.create_task(_execute_task(task.id, plan, session_id))
        else:
            task_manager.update_state(task.id, TMState.AWAITING_APPROVAL)
        
        # Broadcast update to WebSocket clients
        await ws_manager.broadcast_all({
            "type": "task_created",
            "task": _task_to_summary(task_manager.get_task(task.id)).model_dump(mode="json"),
        })
        
        return {
            "task_id": task.id,
            "session_id": session_id,
            "state": task_manager.get_task(task.id).state.value,
            "plan": plan.model_dump(),
            "requires_approval": settings.require_approval and not request.auto_approve,
        }
    
    except LLMError as e:
        task_manager.update_state(task.id, TMState.FAILED, str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Plan generation failed")
        task_manager.update_state(task.id, TMState.FAILED, str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate plan: {e}")


@app.post("/tasks/approve")
async def approve_task(approval: TaskApproval):
    """Approve or reject a pending task."""
    task = task_manager.get_task(approval.task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.state != TMState.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=400, 
            detail=f"Task is not awaiting approval (current state: {task.state.value})"
        )
    
    if approval.approved:
        task_manager.update_state(task.id, TMState.APPROVED)
        
        # Start execution
        plan = Plan(**task.plan)
        asyncio.create_task(_execute_task(task.id, plan, task.session_id))
        
        # Broadcast update
        await ws_manager.broadcast_to_task(task.id, {
            "type": "task_approved",
            "task_id": task.id,
        })
        
        return {"status": "approved", "task_id": task.id}
    else:
        task_manager.update_state(task.id, TMState.REJECTED, approval.feedback)
        
        # Broadcast update
        await ws_manager.broadcast_to_task(task.id, {
            "type": "task_rejected",
            "task_id": task.id,
            "feedback": approval.feedback,
        })
        
        return {"status": "rejected", "task_id": task.id}


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running or pending task."""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.state in {TMState.COMPLETED, TMState.FAILED, TMState.CANCELLED}:
        raise HTTPException(
            status_code=400,
            detail=f"Task is already in terminal state: {task.state.value}"
        )
    
    task_manager.update_state(task_id, TMState.CANCELLED)
    
    # Broadcast update
    await ws_manager.broadcast_to_task(task_id, {
        "type": "task_cancelled",
        "task_id": task_id,
    })
    
    return {"status": "cancelled", "task_id": task_id}


@app.get("/tasks")
async def list_tasks(
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
) -> List[TaskSummary]:
    """List tasks, optionally filtered by session or state."""
    states = None
    if state:
        try:
            states = [TMState(state)]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid state: {state}")
    
    tasks = task_manager.get_tasks(session_id=session_id, states=states, limit=limit)
    return [_task_to_summary(t) for t in tasks]


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> TaskDetail:
    """Get detailed information about a specific task."""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return _task_to_detail(task)


@app.get("/tasks/{task_id}/workspace")
async def get_workspace_files(task_id: str):
    """List files in a task's workspace."""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    files = task_manager.list_workspace_files(task_id)
    
    return {
        "task_id": task_id,
        "workspace_path": task.workspace_path,
        "files": files,
    }


# ============================================================================
# Streaming Endpoint (for backward compatibility)
# ============================================================================

@app.post("/tasks/stream")
async def stream_task(task: TaskRequest, parallel: bool = True):
    """Stream task execution progress via SSE (legacy endpoint)."""
    session_id = task.session_id or str(uuid.uuid4())
    
    async def event_stream():
        try:
            history = get_session_history(session_id)
            add_to_history(session_id, "user", task.request)
            
            plan = generate_plan(task.request, history=history if history else None)
            
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'plan', 'plan': plan.model_dump()})}\n\n"
            
            progress_queue = asyncio.Queue()
            
            def on_progress(step_id: str, status: str, current: int, total: int):
                asyncio.get_event_loop().call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"step_id": step_id, "status": status}
                )
            
            executor = Executor(
                plan=plan,
                tool_registry=tool_registry,
                sandbox=sandbox,
                on_progress=on_progress,
                parallel=parallel,
            )
            
            exec_task = asyncio.create_task(executor.run())
            
            while not exec_task.done():
                try:
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    yield f"data: {json.dumps({'type': 'progress', **progress})}\n\n"
                except asyncio.TimeoutError:
                    continue
            
            while not progress_queue.empty():
                progress = progress_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'progress', **progress})}\n\n"
            
            results = await exec_task
            results_dict = {k: v.model_dump() for k, v in results.items()}
            yield f"data: {json.dumps({'type': 'result', 'results': results_dict})}\n\n"
            
            is_chat = len(plan.steps) == 1 and plan.steps[0].action == "chat_op"
            
            if is_chat:
                chat_result = list(results.values())[0]
                summary = str(chat_result.output) if chat_result.output else "Hello!"
            else:
                summary = summarize_results(task.request, results)
            
            add_to_history(session_id, "assistant", summary)
            yield f"data: {json.dumps({'type': 'summary', 'summary': summary})}\n\n"
            
        except LLMError as e:
            logger.error(f"LLM error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            logger.exception("Task execution failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/tasks", response_model=TaskResponse)
async def create_task_sync(task: TaskRequest, parallel: bool = True):
    """Create and execute a task from natural language (synchronous)."""
    try:
        plan = generate_plan(task.request)
    except LLMError as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Plan generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate plan: {e}")
    
    task_id = str(uuid.uuid4())

    executor = Executor(
        plan=plan, 
        tool_registry=tool_registry, 
        sandbox=sandbox,
        parallel=parallel,
    )
    results = await executor.run()

    return TaskResponse(task_id=task_id, plan=plan, results=results)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.2.0",
        "require_approval": settings.require_approval,
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _task_to_summary(task) -> TaskSummary:
    """Convert internal Task to TaskSummary."""
    step_count = len(task.plan.get("steps", [])) if task.plan else 0
    completed_steps = len([
        r for r in task.step_results.values()
        if r.get("status") == "success"
    ])
    
    return TaskSummary(
        id=task.id,
        request=task.request,
        state=TaskState(task.state.value),
        created_at=task.created_at,
        step_count=step_count,
        completed_steps=completed_steps,
        error=task.error,
    )


def _task_to_detail(task) -> TaskDetail:
    """Convert internal Task to TaskDetail."""
    plan = Plan(**task.plan) if task.plan else None
    step_results = {
        k: StepResult(**v) for k, v in task.step_results.items()
    } if task.step_results else {}
    
    workspace_files = task_manager.list_workspace_files(task.id)
    
    return TaskDetail(
        id=task.id,
        request=task.request,
        session_id=task.session_id,
        state=TaskState(task.state.value),
        created_at=task.created_at,
        updated_at=task.updated_at,
        plan=plan,
        step_results=step_results,
        current_step=task.current_step,
        summary=task.summary,
        error=task.error,
        workspace_path=task.workspace_path,
        workspace_files=workspace_files,
    )


async def _execute_task(task_id: str, plan: Plan, session_id: Optional[str]):
    """Execute a task and broadcast progress updates."""
    task_manager.update_state(task_id, TMState.EXECUTING)
    
    # Broadcast execution started
    await ws_manager.broadcast_to_task(task_id, {
        "type": "execution_started",
        "task_id": task_id,
    })
    
    progress_queue = asyncio.Queue()
    
    def on_progress(step_id: str, status: str, current: int, total: int):
        task_manager.update_step_progress(task_id, step_id, status, current, total)
        asyncio.get_event_loop().call_soon_threadsafe(
            progress_queue.put_nowait,
            {"step_id": step_id, "status": status, "current": current, "total": total}
        )
    
    try:
        executor = Executor(
            plan=plan,
            tool_registry=tool_registry,
            sandbox=sandbox,
            on_progress=on_progress,
            parallel=settings.parallel_execution,
        )
        
        # Start execution
        exec_task = asyncio.create_task(executor.run())
        
        # Stream progress updates via WebSocket
        while not exec_task.done():
            try:
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                await ws_manager.broadcast_to_task(task_id, {
                    "type": "step_progress",
                    "task_id": task_id,
                    **progress,
                })
            except asyncio.TimeoutError:
                continue
        
        # Drain remaining progress events
        while not progress_queue.empty():
            progress = progress_queue.get_nowait()
            await ws_manager.broadcast_to_task(task_id, {
                "type": "step_progress",
                "task_id": task_id,
                **progress,
            })
        
        # Get results
        results = await exec_task
        
        # Store results
        for step_id, result in results.items():
            task_manager.set_step_result(task_id, step_id, result.model_dump())
        
        # Broadcast results
        results_dict = {k: v.model_dump() for k, v in results.items()}
        await ws_manager.broadcast_to_task(task_id, {
            "type": "execution_results",
            "task_id": task_id,
            "results": results_dict,
        })
        
        # Check if any steps failed
        failed_steps = [r for r in results.values() if r.status == "error"]
        
        # Generate summary
        task = task_manager.get_task(task_id)
        is_chat = len(plan.steps) == 1 and plan.steps[0].action == "chat_op"
        
        if is_chat:
            chat_result = list(results.values())[0]
            summary = str(chat_result.output) if chat_result.output else "Hello!"
        else:
            summary = summarize_results(task.request, results)
        
        task_manager.set_summary(task_id, summary)
        
        # Add to conversation history
        if session_id:
            add_to_history(session_id, "assistant", summary)
        
        # Update final state
        if failed_steps:
            task_manager.update_state(task_id, TMState.FAILED)
        else:
            task_manager.update_state(task_id, TMState.COMPLETED)
        
        # Broadcast completion
        await ws_manager.broadcast_to_task(task_id, {
            "type": "task_completed",
            "task_id": task_id,
            "summary": summary,
            "state": task_manager.get_task(task_id).state.value,
        })
    
    except Exception as e:
        logger.exception(f"Task execution failed: {task_id}")
        task_manager.update_state(task_id, TMState.FAILED, str(e))
        
        await ws_manager.broadcast_to_task(task_id, {
            "type": "task_failed",
            "task_id": task_id,
            "error": str(e),
        })
