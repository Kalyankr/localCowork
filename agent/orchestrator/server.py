from fastapi import FastAPI
from agent.orchestrator.models import TaskRequest, TaskResponse
from agent.orchestrator.planner import generate_plan
from agent.orchestrator.executor import Executor
from agent.orchestrator.tool_registry import ToolRegistry
from agent.tools import file_tools
from agent.sandbox.sandbox_runner import Sandbox
import uuid

app = FastAPI()

tool_registry = ToolRegistry()
tool_registry.register("file_op", file_tools.dispatch)

sandbox = Sandbox()


@app.post("/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    plan = generate_plan(task.request)
    task_id = str(uuid.uuid4())

    executor = Executor(plan=plan, tool_registry=tool_registry, sandbox=sandbox)
    results = await executor.run()

    # For now, we just run synchronously and ignore results in response.
    # Later: store by task_id and expose /tasks/{id}/status
    return TaskResponse(task_id=task_id, plan=plan)
