"""LocalCowork API Server - Pure Agentic Architecture.

All task execution uses the ReAct agent (shell + python).

This module is the main entry point that:
- Creates the FastAPI app
- Configures middleware
- Wires up all routes and WebSocket endpoints
"""

import logging

from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from agent.config import settings
from agent.orchestrator.middleware import (
    check_rate_limit,
    global_exception_handler,
    verify_api_key,
)
from agent.orchestrator.models import TaskDetail, TaskRequest, TaskSummary
from agent.orchestrator.routes import (
    cancel_task,
    get_task,
    health,
    list_tasks,
    root,
    run_task,
)
from agent.orchestrator.websocket import (
    stream_chat,
    stream_task,
    websocket_endpoint,
    ws_manager,
)
from agent.version import __version__

logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI App Setup
# =============================================================================

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

# Register exception handler
app.add_exception_handler(Exception, global_exception_handler)


# =============================================================================
# Route Registration
# =============================================================================


@app.get("/")
async def _root():
    """Serve web UI."""
    return await root()


@app.post("/run")
async def _run_task(
    request: TaskRequest,
    req: Request,
    _auth: bool = Depends(verify_api_key),
    _rate: bool = Depends(check_rate_limit),
):
    """Run a task using the ReAct agent."""
    return await run_task(request, req, ws_manager, _auth, _rate)


@app.post("/agent/run")
async def _run_agent(request: TaskRequest, _auth: bool = Depends(verify_api_key)):
    """Alias for /run (backward compatibility)."""
    # Create a dummy request object for the route

    class DummyRequest:
        def __init__(self):
            self.client = None

    return await run_task(request, DummyRequest(), ws_manager, _auth, True)


@app.get("/tasks")
async def _list_tasks(
    session_id: str | None = None,
    state: str | None = None,
    limit: int = 50,
    _auth: bool = Depends(verify_api_key),
) -> list[TaskSummary]:
    """List tasks."""
    return await list_tasks(session_id, state, limit, _auth)


@app.get("/tasks/{task_id}")
async def _get_task(task_id: str, _auth: bool = Depends(verify_api_key)) -> TaskDetail:
    """Get task details."""
    return await get_task(task_id, _auth)


@app.post("/tasks/{task_id}/cancel")
async def _cancel_task(task_id: str, _auth: bool = Depends(verify_api_key)):
    """Cancel a task."""
    return await cancel_task(task_id, ws_manager, _auth)


@app.get("/health")
async def _health():
    """Health check."""
    return await health()


# =============================================================================
# WebSocket Registration
# =============================================================================


@app.websocket("/ws")
async def _websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates."""
    await websocket_endpoint(websocket)


@app.websocket("/ws/stream/{task_id}")
async def _stream_task(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for streaming task execution."""
    await stream_task(websocket, task_id)


@app.websocket("/ws/chat")
async def _stream_chat(websocket: WebSocket):
    """Simple streaming chat WebSocket."""
    await stream_chat(websocket)
