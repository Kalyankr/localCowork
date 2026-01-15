import os
import json
import re
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
            # Check dependencies
            failed_deps = [d for d in step.depends_on if d in results and results[d].status == "error"]
            if failed_deps:
                results[step.id] = StepResult(
                    step_id=step.id,
                    status="skipped",
                    error=f"Dependency failed: {', '.join(failed_deps)}"
                )
                continue

            result = await self.run_step(step)
            results[step.id] = result

            # Store output in context for later steps
            if result.status == "success" and result.output is not None:
                self.context[step.id] = result.output

        return results

    def resolve_args(self, args, depends_on=None):
        def _resolve(val):
            if isinstance(val, str):
                if val in self.context:
                    return self.context[val]
                
                # Check for path-like interpolation "var/path"
                if "/" in val:
                    parts = val.split("/", 1)
                    if parts[0] in self.context:
                        base = str(self.context[parts[0]])
                        return os.path.join(base, parts[1])
                
                return val
            
            if isinstance(val, list):
                return [_resolve(i) for i in val]
            
            if isinstance(val, dict):
                return {k: _resolve(v) for k, v in val.items()}
            
            return val

        return _resolve(args)

    def inject_python_context(self, code, step_id=None):
        injected = "from pathlib import Path\nimport os\nimport json\n"
        
        for name, value in self.context.items():
            if name == step_id and value is None:
                continue
            injected += f"{name} = {repr(value)}\n"

        if step_id and step_id not in self.context:
            injected += f"{step_id} = None\n"
        
        full_code = injected + "\n" + code
        
        # Append logic to capture ALL local variables and the specific result
        full_code += "\n\n# Capture state\n"
        full_code += "try:\n"
        full_code += "    __exclude = {'json', 'os', 'Path', 'injected', 'code', 'step_id', '__exclude', '__state', 'k', 'v'}\n"
        full_code += "    __state = {}\n"
        full_code += "    for k, v in __builtins__.list(locals().items()):\n"
        full_code += "        if not k.startswith('_') and k not in __exclude:\n"
        full_code += "            try:\n"
        full_code += "                json.dumps(v)\n"
        full_code += "                __state[k] = v\n"
        full_code += "            except:\n"
        full_code += "                pass\n"
        if step_id:
            full_code += f"    print(f'__RESULT__:' + json.dumps({step_id}))\n"
        full_code += "    print(f'__TRACE_VARS__:' + json.dumps(__state))\n"
        full_code += "except Exception as e:\n"
        full_code += "    print(f'__TRACE_ERROR__:' + str(e))\n"
        
        return full_code

    async def run_step(self, step):
        try:
            # Resolve arguments
            args = self.resolve_args(step.args, depends_on=step.depends_on)

            # Python execution
            if step.action == "python":
                code = self.inject_python_context(args["code"], step_id=step.id)
                out = await self.sandbox.run_python(code)

                output_text = out.get("output", "")
                error = out.get("error")
                status = "success" if not error else "error"
                
                output_val = None
                
                if "__RESULT__:" in output_text:
                    parts = output_text.split("__RESULT__:", 1)
                    try:
                        res_str = parts[1].strip().split("\n")[0]
                        output_val = json.loads(res_str)
                    except:
                        pass
                
                if "__TRACE_VARS__:" in output_text:
                    trace_parts = output_text.split("__TRACE_VARS__:", 1)
                    try:
                        trace_str = trace_parts[1].strip().split("\n")[0]
                        trace_vars = json.loads(trace_str)
                        for k, v in trace_vars.items():
                            # Skip internals and current step ID to prevent shadowing
                            if k.startswith("_") or k == step.id:
                                continue
                            self.context[k] = v
                    except:
                        pass
                
                if "__TRACE_ERROR__:" in output_text:
                    trace_err_parts = output_text.split("__TRACE_ERROR__:", 1)
                    error = trace_err_parts[1].strip().split("\n")[0]
                    status = "error"

                # Clean up output_text for display (remove markers)
                display_text = output_text
                for marker in ["__RESULT__:", "__TRACE_VARS__:", "__TRACE_ERROR__:"]:
                    if marker in display_text:
                        display_text = display_text.split(marker)[0]

                return StepResult(
                    step_id=step.id,
                    status=status,
                    output=output_val if output_val is not None else display_text.strip(),
                    error=error,
                )

            # Tool execution
            tool = self.tool_registry.get(step.action)
            out = tool(**args)

            return StepResult(step_id=step.id, status="success", output=out)

        except Exception as e:
            return StepResult(step_id=step.id, status="error", error=str(e))
