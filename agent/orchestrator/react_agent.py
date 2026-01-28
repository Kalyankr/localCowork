"""ReAct Agent: Observe → Reason → Act → Repeat

This module implements a truly agentic architecture that:
1. Observes the current state and tool outputs
2. Reasons about what to do next (with explicit thinking)
3. Acts by calling a single tool
4. Repeats until the goal is achieved or max iterations reached

Unlike one-shot planning, the ReAct loop makes decisions step-by-step,
allowing for dynamic adaptation and error recovery.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

from pydantic import BaseModel, Field

from agent.llm.client import call_llm_json
from agent.llm.prompts import REACT_STEP_PROMPT, REFLECTION_PROMPT
from agent.orchestrator.tool_registry import ToolRegistry
from agent.orchestrator.models import StepResult
from agent.sandbox.sandbox_runner import Sandbox

logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 15
# Maximum consecutive failures before stopping
MAX_CONSECUTIVE_FAILURES = 3


class Observation(BaseModel):
    """What the agent observes from the environment."""

    source: str  # "tool", "error", "initial", "reflection"
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)


class Thought(BaseModel):
    """The agent's reasoning about what to do."""

    reasoning: str  # Why the agent is taking this action
    confidence: float = 1.0  # 0-1 confidence score
    is_goal_complete: bool = False  # Whether the agent thinks the goal is done


class Action(BaseModel):
    """An action the agent decides to take."""

    tool: str  # Tool name (e.g., "file_op", "python", "done")
    args: Dict[str, Any] = {}
    description: str = ""  # Human-readable description


class AgentStep(BaseModel):
    """A single step in the agent's execution."""

    iteration: int
    observation: Observation
    thought: Thought
    action: Optional[Action] = None
    result: Optional[StepResult] = None


class AgentState(BaseModel):
    """The full state of an agent execution."""

    goal: str
    status: str = "running"  # running, completed, failed, max_iterations
    steps: List[AgentStep] = []
    context: Dict[str, Any] = {}  # Variables from tool outputs
    final_answer: Optional[str] = None
    error: Optional[str] = None


# Type for progress callback
ProgressCallback = Callable[
    [int, str, str, Optional[str]], None
]  # iteration, status, thought, action


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent.

    A pure agentic system that uses shell and Python to accomplish tasks.
    Instead of planning all steps upfront, this agent:
    1. Looks at the current state
    2. Thinks about what to do next
    3. Executes one action (shell or python)
    4. Observes the result
    5. Repeats until done

    No manual tool registration needed - the agent figures out what commands
    and code to run based on the task.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        tool_registry: Optional[ToolRegistry] = None,  # Optional, for backward compat
        on_progress: Optional[ProgressCallback] = None,
        max_iterations: int = MAX_ITERATIONS,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ):
        self.sandbox = sandbox
        self.tool_registry = tool_registry  # Optional fallback
        self.on_progress = on_progress
        self.max_iterations = max_iterations
        self.conversation_history = conversation_history or []

    async def run(self, goal: str) -> AgentState:
        """
        Execute the ReAct loop for a given goal.

        Args:
            goal: Natural language description of what to accomplish

        Returns:
            AgentState with full execution history
        """
        state = AgentState(goal=goal)
        consecutive_failures = 0

        # Initial observation - keep it minimal to avoid confusing the model
        initial_obs = Observation(source="initial", content="Ready to help.")

        logger.info(f"ReAct agent starting: {goal}")

        for iteration in range(1, self.max_iterations + 1):
            try:
                # Get the last observation (initial or from previous action)
                if iteration == 1:
                    current_obs = initial_obs
                else:
                    # Use the result of the last action as observation
                    last_step = state.steps[-1]
                    if last_step.result:
                        current_obs = Observation(
                            source="tool", content=self._format_result(last_step.result)
                        )
                    else:
                        current_obs = Observation(
                            source="error", content="No result from previous action"
                        )

                # Think: Ask LLM what to do next
                thought, action, direct_response = await self._think(
                    state, current_obs, iteration
                )

                # Create step record
                step = AgentStep(
                    iteration=iteration,
                    observation=current_obs,
                    thought=thought,
                    action=action,
                )

                # Report progress
                if self.on_progress:
                    action_desc = (
                        f"{action.tool}: {action.description}"
                        if action
                        else "thinking..."
                    )
                    self.on_progress(
                        iteration, "thinking", thought.reasoning[:100], action_desc
                    )

                # Check if this is a direct response (conversation mode)
                if direct_response and thought.is_goal_complete:
                    state.steps.append(step)
                    state.status = "completed"
                    state.final_answer = direct_response
                    logger.info("Conversation response - no verification needed")
                    break

                # Check if agent thinks we're done (task mode)
                if thought.is_goal_complete or (action and action.tool == "done"):
                    state.steps.append(step)
                    state.status = "completed"
                    state.final_answer = direct_response or thought.reasoning

                    # Run reflection to verify (only for tasks, not conversations)
                    if not direct_response:
                        reflection = await self._reflect(state)
                        if not reflection["verified"]:
                            # Agent was wrong, continue
                            logger.info(f"Reflection failed: {reflection['reason']}")
                            state.status = "running"
                            state.final_answer = None
                            # Add reflection as observation
                            state.steps[-1] = AgentStep(
                                iteration=iteration,
                                observation=Observation(
                                    source="reflection", content=reflection
                                ),
                                thought=thought,
                                action=None,
                            )
                            continue
                        else:
                            # Use reflection summary if available
                            if reflection.get("summary"):
                                state.final_answer = reflection["summary"]
                            logger.info("Goal verified by reflection")
                    break

                # Act: Execute the chosen action
                if action:
                    result = await self._execute_action(action, state.context)
                    step.result = result

                    # Update context with result
                    if result.status == "success" and result.output is not None:
                        # Use action description or tool name as key
                        key = self._make_context_key(action, iteration)
                        state.context[key] = result.output
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            state.status = "failed"
                            state.error = f"Too many consecutive failures. Last error: {result.error}"
                            state.steps.append(step)
                            break

                state.steps.append(step)

                if self.on_progress:
                    status = (
                        "success"
                        if step.result and step.result.status == "success"
                        else "error"
                        if step.result and step.result.status == "error"
                        else "executing"
                    )
                    self.on_progress(
                        iteration,
                        status,
                        thought.reasoning[:100],
                        f"{action.tool}" if action else "",
                    )

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                state.steps.append(
                    AgentStep(
                        iteration=iteration,
                        observation=Observation(source="error", content=str(e)),
                        thought=Thought(reasoning=f"Error occurred: {e}", confidence=0),
                        action=None,
                    )
                )
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    state.status = "failed"
                    state.error = str(e)
                    break

        if state.status == "running":
            state.status = "max_iterations"
            state.error = f"Reached maximum iterations ({self.max_iterations})"

        logger.info(f"ReAct agent finished: {state.status}")
        return state

    async def _think(
        self, state: AgentState, observation: Observation, iteration: int
    ) -> tuple[Thought, Optional[Action], Optional[str]]:
        """
        Ask the LLM to reason about what to do next.

        Returns:
            Tuple of (Thought, Action or None, Response or None)
        """
        # Build the prompt with history
        history = self._build_history(state)

        # Format conversation history
        conv_history = self._format_conversation_history()

        prompt = REACT_STEP_PROMPT.format(
            goal=state.goal,
            iteration=iteration,
            max_iterations=self.max_iterations,
            conversation_history=conv_history,
            history=history,
            observation=self._format_observation(observation),
            context=json.dumps(state.context, indent=2, default=str)[
                :2000
            ],  # Limit context size
        )

        try:
            response = call_llm_json(prompt)

            thought = Thought(
                reasoning=response.get(
                    "thought", response.get("reasoning", "No reasoning provided")
                ),
                confidence=response.get("confidence", 1.0),
                is_goal_complete=response.get("is_complete", False)
                or response.get("done", False),
            )

            # Check for direct response (conversation mode)
            direct_response = response.get("response")

            action = None
            if "action" in response and response["action"]:
                action_data = response["action"]
                if isinstance(action_data, dict) and action_data.get("tool"):
                    action = Action(
                        tool=action_data.get("tool", ""),
                        args=action_data.get("args", {}),
                        description=action_data.get("description", ""),
                    )

            return thought, action, direct_response

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return (
                Thought(reasoning=f"Error parsing response: {e}", confidence=0),
                None,
                None,
            )

    async def _execute_action(
        self, action: Action, context: Dict[str, Any]
    ) -> StepResult:
        """Execute a single action and return the result."""
        import subprocess
        import os

        try:
            # Handle Python code execution
            if action.tool == "python":
                code = action.args.get("code", "")
                full_code = self._inject_context(code, context)
                result = await self.sandbox.run_python(full_code)

                if result.get("error"):
                    return StepResult(
                        step_id=f"action_{action.tool}",
                        status="error",
                        output=result.get("output"),
                        error=result.get("error"),
                    )

                # Parse output for variables
                output = result.get("output", "")
                parsed_output = self._parse_python_output(output)

                return StepResult(
                    step_id=f"action_{action.tool}",
                    status="success",
                    output=parsed_output,
                )

            # Handle shell command execution
            if action.tool == "shell":
                command = action.args.get("command", "")
                cwd = action.args.get("cwd")

                # Expand ~ in cwd if provided, otherwise use home
                if cwd:
                    cwd = os.path.expanduser(cwd)
                else:
                    cwd = os.path.expanduser("~")

                # Expand ~ in command itself
                command = command.replace("~/", os.path.expanduser("~") + "/")

                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        timeout=120,  # 2 minutes for longer operations
                        cwd=cwd,
                        env={**os.environ, "HOME": os.path.expanduser("~")},
                    )

                    output = result.stdout.decode(errors="replace")
                    stderr = result.stderr.decode(errors="replace")

                    # Non-zero exit isn't always an error (e.g., grep no match)
                    # Return both stdout and stderr for context
                    if result.returncode != 0:
                        return StepResult(
                            step_id="shell",
                            status="error",
                            output=output if output else None,
                            error=f"Exit {result.returncode}: {stderr[:500]}"
                            if stderr
                            else f"Exit {result.returncode}",
                        )

                    # Include stderr in output if present (warnings, progress, etc.)
                    final_output = output
                    if stderr and not output:
                        final_output = stderr
                    elif stderr and output:
                        final_output = output + "\n[stderr]: " + stderr[:200]

                    return StepResult(
                        step_id="shell",
                        status="success",
                        output=final_output or "(no output)",
                    )
                except subprocess.TimeoutExpired:
                    return StepResult(
                        step_id="shell",
                        status="error",
                        error="Command timed out after 2 minutes",
                    )
                except Exception as e:
                    return StepResult(
                        step_id="shell", status="error", error=f"Shell error: {str(e)}"
                    )

            # Fallback: try registered tools (for backward compatibility)
            if self.tool_registry and self.tool_registry.has(action.tool):
                tool = self.tool_registry.get(action.tool)
                resolved_args = self._resolve_args(action.args, context)
                output = tool(**resolved_args)
                return StepResult(
                    step_id=f"action_{action.tool}", status="success", output=output
                )

            # Unknown tool
            return StepResult(
                step_id=f"action_{action.tool}",
                status="error",
                error=f"Unknown tool: {action.tool}. Use 'shell' or 'python'.",
            )

        except Exception as e:
            return StepResult(
                step_id=f"action_{action.tool}", status="error", error=str(e)
            )

    async def _reflect(self, state: AgentState) -> Dict[str, Any]:
        """
        Reflect on whether the goal was actually achieved.

        Returns:
            Dict with "verified" (bool), "reason" (str), and "summary" (str)
        """
        prompt = REFLECTION_PROMPT.format(
            goal=state.goal,
            steps_summary=self._summarize_steps(state),
            final_context=json.dumps(state.context, indent=2, default=str)[:3000],
        )

        try:
            response = call_llm_json(prompt)
            return {
                "verified": response.get(
                    "verified", response.get("goal_achieved", False)
                ),
                "reason": response.get("reason", response.get("explanation", "")),
                "summary": response.get("summary", ""),
                "suggestions": response.get("suggestions", []),
            }
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            # Default to verified if reflection fails
            return {
                "verified": True,
                "reason": "Reflection check failed, assuming complete",
                "summary": "",
            }

    def _build_history(self, state: AgentState) -> str:
        """Build a string representation of execution history."""
        if not state.steps:
            return "(no previous steps)"

        lines = []
        for step in state.steps[-5:]:  # Keep last 5 steps for context
            lines.append(f"Step {step.iteration}:")
            lines.append(f"  Thought: {step.thought.reasoning[:150]}...")
            if step.action:
                lines.append(f"  Action: {step.action.tool}({step.action.args})")
            if step.result:
                status = step.result.status
                output_preview = (
                    str(step.result.output)[:100] if step.result.output else ""
                )
                error = step.result.error or ""
                lines.append(f"  Result: {status} - {output_preview or error}")
            lines.append("")

        return "\n".join(lines)

    def _format_conversation_history(self) -> str:
        """Format conversation history for context."""
        if not self.conversation_history:
            return "(new conversation)"

        lines = []
        # Show last 5 exchanges (10 messages)
        recent = self.conversation_history[-10:]
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            prefix = "User:" if role == "user" else "Assistant:"
            lines.append(f"{prefix} {content}")

        return "\n".join(lines)

    def _format_observation(self, obs: Observation) -> str:
        """Format an observation for the prompt."""
        content = obs.content
        if isinstance(content, dict):
            content = json.dumps(content, indent=2, default=str)
        elif isinstance(content, list):
            content = json.dumps(content[:10], indent=2, default=str)  # Limit list size
            if len(obs.content) > 10:
                content += f"\n... and {len(obs.content) - 10} more items"
        return f"[{obs.source}] {content}"

    def _format_result(self, result: StepResult) -> str:
        """Format a step result for observation."""
        if result.status == "success":
            output = result.output
            if isinstance(output, (dict, list)):
                return json.dumps(output, indent=2, default=str)[:1000]
            return str(output)[:1000]
        else:
            return f"ERROR: {result.error}"

    def _make_context_key(self, action: Action, iteration: int) -> str:
        """Generate a context key for storing action results."""
        # Try to use a descriptive name
        if action.description:
            # Convert description to snake_case
            key = action.description.lower().replace(" ", "_")[:20]
            return f"{key}_{iteration}"
        return f"{action.tool}_result_{iteration}"

    def _inject_context(self, code: str, context: Dict[str, Any]) -> str:
        """Inject context variables into Python code."""
        injected = "import json\n"
        for name, value in context.items():
            # Make valid Python variable name
            safe_name = name.replace("-", "_").replace(" ", "_")
            injected += f"{safe_name} = {repr(value)}\n"
        return injected + "\n" + code

    def _parse_python_output(self, output: str) -> Any:
        """Parse Python execution output."""
        # Try to parse as JSON
        try:
            return json.loads(output.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        return output.strip()

    def _resolve_args(
        self, args: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve context references in arguments."""

        def _resolve(val):
            if isinstance(val, str):
                # Check if it's a context reference
                if val in context:
                    return context[val]
                return val
            if isinstance(val, list):
                return [_resolve(v) for v in val]
            if isinstance(val, dict):
                return {k: _resolve(v) for k, v in val.items()}
            return val

        return {k: _resolve(v) for k, v in args.items()}

    def _summarize_steps(self, state: AgentState) -> str:
        """Summarize all steps for reflection."""
        lines = []
        for step in state.steps:
            action_str = (
                f"{step.action.tool}({step.action.description})"
                if step.action
                else "no action"
            )
            result_str = step.result.status if step.result else "no result"
            lines.append(f"{step.iteration}. {action_str} → {result_str}")
        return "\n".join(lines)
