from typing import Dict, Any
from agent.orchestrator.models import Plan, StepResult
from agent.orchestrator.tool_registry import ToolRegistry
from agent.sandbox.sandbox_runner import Sandbox


class Executor:
    def __init__(self, plan: Plan, tool_registry: ToolRegistry, sandbox: Sandbox):
        self.plan = plan
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.results: Dict[str, StepResult] = {}

    async def run(self) -> Dict[str, StepResult]:
        # Simple sequential execution for now
        for step in self.plan.steps:
            result = await self.run_step(step)
            self.results[step.id] = result
        return self.results

    async def run_step(self, step) -> StepResult:
        try:
            if step.action == "python":
                out = await self.sandbox.run_python(step.args["code"])
                status = "success" if "error" not in out else "error"
                return StepResult(
                    step_id=step.id,
                    status=status,
                    output=out.get("output"),
                    error=out.get("error"),
                )

            # Tool-based actions
            tool = self.tool_registry.get(step.action)
            out = tool(**step.args)
            return StepResult(step_id=step.id, status="success", output=out)

        except Exception as e:
            return StepResult(step_id=step.id, status="error", error=str(e))
