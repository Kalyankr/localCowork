from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from agent.orchestrator.models import TaskRequest, TaskResponse, ConversationMessage
from agent.orchestrator.planner import generate_plan, summarize_results
from agent.orchestrator.executor import Executor
from agent.orchestrator.deps import get_tool_registry, get_sandbox
from agent.config import settings
from agent.llm.client import LLMError
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LocalCowork API",
    description="AI-powered local task automation",
    version="0.1.0",
)

# Use shared dependencies
tool_registry = get_tool_registry()
sandbox = get_sandbox()

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Conversation history storage (in-memory, keyed by session_id)
# In production, you might want to use Redis or a database
conversation_history: Dict[str, List[ConversationMessage]] = defaultdict(list)
conversation_timestamps: Dict[str, float] = {}  # For cleanup

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
    
    # Keep only last N messages per session to prevent memory bloat
    if len(conversation_history[session_id]) > MAX_HISTORY_MESSAGES:
        conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY_MESSAGES:]


@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/tasks/stream")
async def stream_task(task: TaskRequest, parallel: bool = True):
    """Stream task execution progress via SSE."""
    
    # Get or create session ID
    session_id = task.session_id or str(uuid.uuid4())
    
    async def event_stream():
        try:
            # Get conversation history for context
            history = get_session_history(session_id)
            
            # Add user message to history
            add_to_history(session_id, "user", task.request)
            
            # Generate plan with conversation context
            plan = generate_plan(task.request, history=history if history else None)
            
            # Send session_id back to client (for future requests)
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'plan', 'plan': plan.model_dump()})}\n\n"
            
            # Track progress
            progress_queue = asyncio.Queue()
            
            def on_progress(step_id: str, status: str, current: int, total: int):
                asyncio.get_event_loop().call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"step_id": step_id, "status": status}
                )
            
            # Create executor
            executor = Executor(
                plan=plan,
                tool_registry=tool_registry,
                sandbox=sandbox,
                on_progress=on_progress,
                parallel=parallel,
            )
            
            # Run execution in background
            exec_task = asyncio.create_task(executor.run())
            
            # Stream progress updates
            while not exec_task.done():
                try:
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    yield f"data: {json.dumps({'type': 'progress', **progress})}\n\n"
                except asyncio.TimeoutError:
                    continue
            
            # Drain remaining progress events
            while not progress_queue.empty():
                progress = progress_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'progress', **progress})}\n\n"
            
            # Get results
            results = await exec_task
            results_dict = {k: v.model_dump() for k, v in results.items()}
            yield f"data: {json.dumps({'type': 'result', 'results': results_dict})}\n\n"
            
            # Check if this was a chat response (skip summarizer for chat)
            is_chat = len(plan.steps) == 1 and plan.steps[0].action == "chat_op"
            
            if is_chat:
                # For chat, use the response directly
                chat_result = list(results.values())[0]
                summary = str(chat_result.output) if chat_result.output else "Hello!"
            else:
                # Generate summary for task results
                summary = summarize_results(task.request, results)
            
            # Add assistant response to history
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
async def create_task(task: TaskRequest, parallel: bool = True):
    """Create and execute a task from natural language."""
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
    return {"status": "ok"}
