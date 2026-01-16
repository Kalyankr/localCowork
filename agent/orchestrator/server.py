from fastapi import FastAPI, HTTPException
from agent.orchestrator.models import TaskRequest, TaskResponse
from agent.orchestrator.planner import generate_plan
from agent.orchestrator.executor import Executor
from agent.tools import create_default_registry
from agent.sandbox.sandbox_runner import Sandbox
from agent.llm.client import LLMError
import uuid
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LocalCowork API",
    description="AI-powered local task automation",
    version="0.1.0",
)

# Use shared registry
tool_registry = create_default_registry()
sandbox = Sandbox()


@app.post("/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
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

    executor = Executor(plan=plan, tool_registry=tool_registry, sandbox=sandbox)
    results = await executor.run()

    return TaskResponse(task_id=task_id, plan=plan, results=results)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
