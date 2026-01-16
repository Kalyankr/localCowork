import os
import json
import re
import asyncio
import logging
from typing import Callable, Optional, List, Set
from agent.orchestrator.models import StepResult, Step
from agent.orchestrator.tool_registry import ToolRegistry
from agent.sandbox.sandbox_runner import Sandbox

logger = logging.getLogger(__name__)

# Type for progress callback: (step_id, status, current, total)
ProgressCallback = Callable[[str, str, int, int], None]


class Executor:
    """Executes a plan by running steps in dependency order with parallel execution."""
    
    def __init__(
        self, 
        plan, 
        tool_registry: ToolRegistry, 
        sandbox: Sandbox,
        on_progress: Optional[ProgressCallback] = None,
        parallel: bool = True,
    ):
        self.plan = plan
        self.tool_registry = tool_registry
        self.sandbox = sandbox
        self.context = {}  # store step outputs
        self.on_progress = on_progress
        self.parallel = parallel
        self._lock = asyncio.Lock()  # protect shared context in parallel mode

    def _get_ready_steps(
        self, 
        completed: Set[str], 
        failed: Set[str], 
        running: Set[str]
    ) -> List[Step]:
        """Get steps that are ready to run (all dependencies met)."""
        ready = []
        for step in self.plan.steps:
            if step.id in completed or step.id in failed or step.id in running:
                continue
            
            # Check if all dependencies are completed successfully
            deps_met = all(d in completed for d in step.depends_on)
            deps_failed = any(d in failed for d in step.depends_on)
            
            if deps_failed:
                # Mark as failed due to dependency
                failed.add(step.id)
            elif deps_met:
                ready.append(step)
        
        return ready

    async def run(self) -> dict:
        """Execute all steps in the plan with parallel execution for independent steps."""
        results = {}
        total_steps = len(self.plan.steps)
        completed: Set[str] = set()
        failed: Set[str] = set()
        running: Set[str] = set()
        
        logger.info(f"Executing plan with {total_steps} steps (parallel={self.parallel})")

        while len(completed) + len(failed) < total_steps:
            # Get steps ready to execute
            ready_steps = self._get_ready_steps(completed, failed, running)
            
            if not ready_steps and not running:
                # No steps ready and none running - check for failures
                for step in self.plan.steps:
                    if step.id not in completed and step.id not in failed:
                        failed_deps = [d for d in step.depends_on if d in failed]
                        if failed_deps:
                            results[step.id] = StepResult(
                                step_id=step.id,
                                status="skipped",
                                error=f"Dependency failed: {', '.join(failed_deps)}"
                            )
                            failed.add(step.id)
                break
            
            if not ready_steps:
                # Steps are running, wait a bit
                await asyncio.sleep(0.01)
                continue

            # Report progress
            if self.on_progress:
                for step in ready_steps:
                    self.on_progress(step.id, "starting", len(completed) + 1, total_steps)

            if self.parallel and len(ready_steps) > 1:
                # Run independent steps in parallel
                logger.info(f"Running {len(ready_steps)} steps in parallel: {[s.id for s in ready_steps]}")
                running.update(s.id for s in ready_steps)
                
                tasks = [self._run_step_wrapper(step, results, completed, failed, running, total_steps) 
                         for step in ready_steps]
                await asyncio.gather(*tasks)
            else:
                # Run sequentially
                for step in ready_steps:
                    running.add(step.id)
                    await self._run_step_wrapper(step, results, completed, failed, running, total_steps)

        return results

    async def _run_step_wrapper(
        self, 
        step: Step, 
        results: dict, 
        completed: Set[str], 
        failed: Set[str],
        running: Set[str],
        total_steps: int
    ):
        """Wrapper to run a step and update tracking sets."""
        logger.debug(f"Processing step: {step.id} (action={step.action})")
        
        result = await self.run_step(step)
        
        # Thread-safe update of shared state
        async with self._lock:
            results[step.id] = result
            running.discard(step.id)
            
            if result.status == "success":
                completed.add(step.id)
                if result.output is not None:
                    self.context[step.id] = result.output
            else:
                failed.add(step.id)
        
        logger.info(f"Step {step.id}: {result.status}")
        
        if self.on_progress:
            self.on_progress(step.id, result.status, len(completed), total_steps)

    def resolve_args(self, args, depends_on=None):
        def _resolve(val):
            if isinstance(val, str):
                if val in self.context:
                    return self.context[val]
                
                # Check for dictionary or list indexing: var['key'] or var[0]
                index_match = re.match(r"^(\w+)\[['\"]?([^'\"\]]+)['\"]?\]$", val)
                if index_match:
                    base_var, key = index_match.groups()
                    if base_var in self.context:
                        data = self.context[base_var]
                        if isinstance(data, dict):
                            return data.get(key)
                        if isinstance(data, list) and key.isdigit():
                            return data[int(key)]

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
