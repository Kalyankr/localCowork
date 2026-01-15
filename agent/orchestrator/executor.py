from agent.orchestrator.models import StepResult
from agent.orchestrator.tool_registry import ToolRegistry
from agent.sandbox.sandbox_runner import Sandbox


class Executor:
    def __init__(self, plan, tool_registry: ToolRegistry, sandbox: Sandbox):
        self.plan = plan
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.context = {}  # store step outputs

    async def run(self):
        results = {}

        for step in self.plan.steps:
            result = await self.run_step(step)
            results[step.id] = result

            # Store output in context for later steps
            if result.status == "success" and result.output is not None:
                self.context[step.id] = result.output

        return results

    def resolve_args(self, args):
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value in self.context:
                resolved[key] = self.context[value]
            else:
                resolved[key] = value
        return resolved

    def inject_python_context(self, code):
        injected = "from pathlib import Path\nimport os\n"
        for name, value in self.context.items():
            injected += f"{name} = {repr(value)}\n"
        return injected + "\n" + code

    async def run_step(self, step):
        try:
            # Resolve arguments
            args = self.resolve_args(step.args)

            # Python execution
            if step.action == "python":
                code = self.inject_python_context(args["code"])
                out = await self.sandbox.run_python(code)

                status = "success" if "error" not in out else "error"
                return StepResult(
                    step_id=step.id,
                    status=status,
                    output=out.get("output"),
                    error=out.get("error"),
                )

            # Tool execution
            tool = self.tool_registry.get(step.action)
            out = tool(**args)

            return StepResult(step_id=step.id, status="success", output=out)

        except Exception as e:
            return StepResult(step_id=step.id, status="error", error=str(e))
