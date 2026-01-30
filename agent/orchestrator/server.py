"""LocalCowork API Server - Pure Agentic Architecture.

All task execution uses the ReAct agent (shell + python).
"""

import asyncio
import logging
import secrets
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import ValidationError

from agent.config import settings
from agent.llm.client import LLMError
from agent.orchestrator.deps import get_sandbox, get_task_manager
from agent.orchestrator.models import (
    ConversationMessage,
    TaskDetail,
    TaskRequest,
    TaskState,
    TaskSummary,
    WebSocketMessage,
    WSMessageType,
)
from agent.orchestrator.task_manager import TaskState as TMState
from agent.version import __version__

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LocalCowork API",
    description="Pure agentic local automation - shell + python",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - allow local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Key Authentication (optional)
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> bool:
    """Verify API key if configured, otherwise allow all requests."""
    if settings.api_key is None:
        # No API key configured - allow all (localhost only by default)
        return True

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Set X-API-Key header.",
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return True


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions gracefully."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
            if settings.ollama_model == "debug"
            else "An unexpected error occurred",
        },
    )


# =============================================================================
# Rate Limiter (simple in-memory implementation)
# =============================================================================


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for this client."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[client_id] = [
            ts for ts in self.requests[client_id] if ts > window_start
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until rate limit resets."""
        if not self.requests[client_id]:
            return 0
        oldest = min(self.requests[client_id])
        return max(0, int(self.window_seconds - (time.time() - oldest)))


rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests, window_seconds=settings.rate_limit_window
)


async def check_rate_limit(request: Request) -> bool:
    """Check rate limit for a request."""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        retry_after = rate_limiter.get_retry_after(client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )
    return True


# Shared dependencies - only sandbox needed for pure agentic execution
sandbox = get_sandbox()
task_manager = get_task_manager()

# Static files
STATIC_DIR = Path(__file__).parent / "static"

# Session storage
conversation_history: dict[str, list[ConversationMessage]] = defaultdict(list)
conversation_timestamps: dict[str, float] = {}

SESSION_TIMEOUT = settings.session_timeout
MAX_HISTORY = settings.max_history_messages


# =============================================================================
# WebSocket Manager
# =============================================================================


class ConnectionManager:
    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.task_subs: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)
        for subs in self.task_subs.values():
            subs.discard(ws)

    def subscribe(self, ws: WebSocket, task_id: str):
        self.task_subs[task_id].add(ws)

    async def broadcast(self, task_id: str, msg: WebSocketMessage | dict):
        """Broadcast a typed message to all subscribers of a task."""
        dead = set()
        data = msg.model_dump() if isinstance(msg, WebSocketMessage) else msg
        for conn in self.task_subs.get(task_id, set()):
            try:
                await conn.send_json(data)
            except Exception:
                dead.add(conn)
        for conn in dead:
            self.disconnect(conn)

    async def broadcast_all(self, msg: WebSocketMessage):
        """Broadcast a typed message to all connected clients."""
        dead = set()
        for conn in self.connections:
            try:
                await conn.send_json(msg.model_dump())
            except Exception:
                dead.add(conn)
        for conn in dead:
            self.disconnect(conn)

    async def send_step_output(self, task_id: str, step: str, output: Any):
        """Send step output to task subscribers."""
        await self.broadcast(
            task_id, WebSocketMessage.step_output(task_id, step, output)
        )

    async def send_task_complete(self, task_id: str, summary: str):
        """Notify subscribers that a task completed."""
        await self.broadcast(task_id, WebSocketMessage.task_complete(task_id, summary))

    async def send_task_error(self, task_id: str, error: str):
        """Notify subscribers of a task error."""
        await self.broadcast(task_id, WebSocketMessage.task_error(task_id, error))


ws = ConnectionManager()


# =============================================================================
# Session Helpers
# =============================================================================


def cleanup_sessions():
    now = time.time()
    expired = [
        s for s, t in conversation_timestamps.items() if now - t > SESSION_TIMEOUT
    ]
    for s in expired:
        conversation_history.pop(s, None)
        conversation_timestamps.pop(s, None)


def get_history(session_id: str) -> list[ConversationMessage]:
    cleanup_sessions()
    return conversation_history.get(session_id, [])


def add_message(session_id: str, role: str, content: str):
    conversation_history[session_id].append(
        ConversationMessage(role=role, content=content)
    )
    conversation_timestamps[session_id] = time.time()
    if len(conversation_history[session_id]) > MAX_HISTORY:
        conversation_history[session_id] = conversation_history[session_id][
            -MAX_HISTORY:
        ]


# =============================================================================
# WebSocket Endpoint
# =============================================================================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates.

    Supported messages:
    - {"type": "subscribe", "task_id": "..."} - Subscribe to task updates
    - {"type": "unsubscribe", "task_id": "..."} - Unsubscribe from task
    - {"type": "ping"} - Keep-alive ping
    """
    await ws.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_json()
            try:
                msg = WebSocketMessage.model_validate(raw_data)

                if msg.type == WSMessageType.SUBSCRIBE:
                    if msg.task_id:
                        ws.subscribe(websocket, msg.task_id)
                        response = WebSocketMessage.subscribed(msg.task_id)
                        await websocket.send_json(response.model_dump())
                    else:
                        error = WebSocketMessage.error("task_id required for subscribe")
                        await websocket.send_json(error.model_dump())

                elif msg.type == WSMessageType.UNSUBSCRIBE:
                    if msg.task_id:
                        ws.task_subs.get(msg.task_id, set()).discard(websocket)
                        await websocket.send_json(
                            {"type": "unsubscribed", "task_id": msg.task_id}
                        )

                elif msg.type == WSMessageType.PING:
                    response = WebSocketMessage.pong()
                    await websocket.send_json(response.model_dump())

            except ValidationError as e:
                error = WebSocketMessage.error(
                    f"Invalid message format: {e.error_count()} errors"
                )
                await websocket.send_json(error.model_dump())

    except WebSocketDisconnect:
        ws.disconnect(websocket)


# =============================================================================
# Main Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Serve web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/run")
async def run_task(
    request: TaskRequest,
    req: Request,
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
        await ws.broadcast(
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
        # pending_confirmations tracks confirmation requests per task
        pending_confirmations: dict[str, asyncio.Event] = {}
        confirmation_results: dict[str, bool] = {}

        async def on_confirm(command: str, reason: str, message: str) -> bool:
            """Request confirmation via WebSocket and wait for response."""
            confirm_id = str(uuid.uuid4())
            event = asyncio.Event()
            pending_confirmations[confirm_id] = event

            # Send confirmation request to connected clients
            await ws.broadcast(
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

        await ws.broadcast(
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


# Legacy alias
@app.post("/agent/run")
async def run_agent(
    request: TaskRequest,
    req: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
):
    """Alias for /run (backward compatibility)."""
    return await run_task(request, req, _auth, _rate)


@app.get("/tasks")
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


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, _auth: bool = Depends(verify_api_key)) -> TaskDetail:
    """Get task details."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _to_detail(task)


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, _auth: bool = Depends(verify_api_key)):
    """Cancel a task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.state in {TMState.COMPLETED, TMState.FAILED, TMState.CANCELLED}:
        raise HTTPException(status_code=400, detail=f"Task already {task.state.value}")

    task_manager.update_state(task_id, TMState.CANCELLED)
    await ws.broadcast(task_id, {"type": "cancelled", "task_id": task_id})
    return {"status": "cancelled", "task_id": task_id}


@app.get("/health")
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
