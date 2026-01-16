from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from agent.orchestrator.models import TaskRequest, TaskResponse
from agent.orchestrator.planner import generate_plan, summarize_results
from agent.orchestrator.executor import Executor
from agent.tools import create_default_registry
from agent.sandbox.sandbox_runner import Sandbox
from agent.llm.client import LLMError
import uuid
import json
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LocalCowork API",
    description="AI-powered local task automation",
    version="0.1.0",
)

# Use shared registry
tool_registry = create_default_registry()
sandbox = Sandbox()

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/tasks/stream")
async def stream_task(task: TaskRequest, parallel: bool = True):
    """Stream task execution progress via SSE."""
    
    async def event_stream():
        try:
            # Generate plan
            plan = generate_plan(task.request)
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
            
            # Generate summary
            summary = summarize_results(task.request, results)
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
