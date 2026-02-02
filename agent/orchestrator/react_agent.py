"""ReAct Agent: Observe → Reason → Act → Repeat

This module implements a truly agentic architecture that:
1. Observes the current state and tool outputs
2. Reasons about what to do next (with explicit thinking)
3. Acts by calling a single tool
4. Repeats until the goal is achieved or max iterations reached

Unlike one-shot planning, the ReAct loop makes decisions step-by-step,
allowing for dynamic adaptation and error recovery.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
from typing import Any

from agent.config import settings
from agent.llm.client import call_llm_json_async
from agent.llm.prompts import (
    ERROR_RECOVERY_PROMPT,
    MERGE_SUBTASKS_PROMPT,
    REACT_STEP_PROMPT,
    REFLECTION_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
)
from agent.orchestrator.agent_models import (
    Action,
    AgentState,
    AgentStep,
    ConfirmCallback,
    Observation,
    ProgressCallback,
    SubAgentState,
    SubTask,
    Thought,
)
from agent.orchestrator.models import StepResult
from agent.permissions import (
    AccessLevel,
    get_permission_error_message,
    validate_command_paths,
)
from agent.safety import (
    DangerLevel,
    analyze_command,
    analyze_python_code,
    format_confirmation_message,
    get_affected_paths,
)
from agent.sandbox.sandbox_runner import Sandbox
from agent.web import fetch_webpage, web_search

logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 15
# Maximum consecutive failures before stopping
MAX_CONSECUTIVE_FAILURES = 3
# Maximum sub-agents that can run in parallel
MAX_PARALLEL_SUBTASKS = 4
# Reduced iterations for sub-agents (they handle smaller tasks)
SUB_AGENT_MAX_ITERATIONS = 8


def _sanitize_error(error: str, tool: str = "command") -> str:
    """
    Sanitize error messages to be user-friendly.

    Removes raw code dumps and technical details, keeping only
    the essential error information.
    """
    if not error:
        return f"The {tool} encountered an issue."

    # Extract just the error type and message from Python tracebacks
    lines = error.strip().split("\n")

    # Look for common Python error patterns
    for line in reversed(lines):
        line = line.strip()
        # Match patterns like "NameError: name 'x' is not defined"
        if any(
            err in line
            for err in [
                "Error:",
                "Exception:",
                "error:",
                "ModuleNotFoundError",
                "ImportError",
                "FileNotFoundError",
                "PermissionError",
                "TypeError",
                "ValueError",
                "KeyError",
                "IndexError",
                "AttributeError",
                "SyntaxError",
                "NameError",
                "ZeroDivisionError",
                "RuntimeError",
                "OSError",
            ]
        ):
            # Clean up the error message
            if len(line) > 200:
                line = line[:200] + "..."
            return f"Error: {line}"

    # For shell errors, extract the main message
    if "Exit" in error and ":" in error:
        # Format: "Exit 1: error message"
        parts = error.split(":", 1)
        if len(parts) > 1:
            msg = parts[1].strip()
            if len(msg) > 200:
                msg = msg[:200] + "..."
            return f"Command failed: {msg}" if msg else "Command failed"
        return "Command failed"

    # For timeout errors
    if "timed out" in error.lower() or "timeout" in error.lower():
        return f"The {tool} took too long and was stopped."

    # For connection errors
    if "connection" in error.lower() or "connect" in error.lower():
        return "Connection error. Please check your network."

    # Generic fallback - don't expose raw technical details
    if len(error) > 150 or "\n" in error:
        return f"The {tool} encountered an error. Please try a different approach."

    return f"Error: {error}"


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

    Safety: Dangerous operations (file deletion, etc.) require explicit
    user confirmation via the on_confirm callback.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        on_progress: ProgressCallback | None = None,
        on_confirm: ConfirmCallback | None = None,  # Confirmation for dangerous ops
        max_iterations: int = MAX_ITERATIONS,
        conversation_history: list[dict[str, str]] | None = None,
        require_confirmation: bool = True,  # If False, skip confirmation prompts
    ):
        self.sandbox = sandbox
        self.on_progress = on_progress
        self.on_confirm = on_confirm
        self.max_iterations = max_iterations
        self.conversation_history = conversation_history or []
        self.require_confirmation = require_confirmation
        self._is_sub_agent = False  # Track if this is a sub-agent

    async def _should_decompose(self, goal: str) -> tuple[bool, list[SubTask]]:
        """
        Analyze if a task should be decomposed into parallel subtasks.

        Returns:
            Tuple of (should_parallelize, list of subtasks)
        """
        prompt = TASK_DECOMPOSITION_PROMPT.format(goal=goal)

        try:
            response = await call_llm_json_async(prompt)

            should_parallelize = response.get("should_parallelize", False)
            subtasks_data = response.get("subtasks", [])

            if not should_parallelize or len(subtasks_data) < 2:
                return False, []

            # Limit to MAX_PARALLEL_SUBTASKS
            subtasks_data = subtasks_data[:MAX_PARALLEL_SUBTASKS]

            subtasks = [
                SubTask(
                    id=str(st.get("id", i)),
                    description=st.get("description", ""),
                    dependencies=st.get("dependencies", []),
                )
                for i, st in enumerate(subtasks_data)
            ]

            # Filter to only independent subtasks (no dependencies)
            independent = [st for st in subtasks if not st.dependencies]

            if len(independent) < 2:
                return False, []

            logger.info(f"Task decomposed into {len(independent)} parallel subtasks")
            return True, independent

        except Exception as e:
            logger.warning(f"Task decomposition failed: {e}")
            return False, []

    async def _run_subtask(
        self, subtask: SubTask, parent_context: dict[str, Any], parent_goal: str = ""
    ) -> dict[str, Any]:
        """
        Run a single subtask using a sub-agent.

        Args:
            subtask: The subtask to execute
            parent_context: Context from parent agent (files read, data collected)
            parent_goal: The parent agent's goal for reference

        Returns:
            Dict with subtask result
        """
        logger.info(f"Sub-agent starting: {subtask.description}")

        # Build context-aware conversation history for sub-agent
        sub_agent_context = []
        if parent_goal:
            sub_agent_context.append(
                {
                    "role": "system",
                    "content": f"You are working on a subtask of: {parent_goal}",
                }
            )
        # Include relevant parent context as system message
        if parent_context:
            context_summary = json.dumps(parent_context, indent=2, default=str)[:2000]
            sub_agent_context.append(
                {
                    "role": "system",
                    "content": f"Available context from parent task:\n{context_summary}",
                }
            )

        # Create a sub-agent with reduced iterations and inherited context
        sub_agent = ReActAgent(
            sandbox=self.sandbox,
            on_progress=self.on_progress,
            on_confirm=self.on_confirm,
            max_iterations=SUB_AGENT_MAX_ITERATIONS,
            conversation_history=sub_agent_context,  # Pass parent context
            require_confirmation=self.require_confirmation,
        )
        sub_agent._is_sub_agent = True

        try:
            state = await sub_agent.run(subtask.description)

            # Extract meaningful result even if final_answer is None
            result_text = state.final_answer
            if not result_text and state.steps:
                # Try to extract result from last successful step
                for step in reversed(state.steps):
                    if (
                        step.result
                        and step.result.status == "success"
                        and step.result.output
                    ):
                        result_text = str(step.result.output)[:1000]
                        break
                if not result_text:
                    result_text = f"Subtask ran {len(state.steps)} steps but produced no explicit result."

            return {
                "id": subtask.id,
                "description": subtask.description,
                "status": state.status,
                "result": result_text,
                "error": state.error,
                "steps_count": len(state.steps),
            }
        except Exception as e:
            logger.error(f"Sub-agent failed: {e}")
            return {
                "id": subtask.id,
                "description": subtask.description,
                "status": "failed",
                "result": None,
                "error": str(e),
            }

    async def _run_parallel_subtasks(
        self,
        subtasks: list[SubTask],
        parent_context: dict[str, Any],
        parent_goal: str = "",
    ) -> list[dict[str, Any]]:
        """
        Run multiple subtasks in parallel.

        Args:
            subtasks: List of subtasks to run
            parent_context: Context from parent agent
            parent_goal: The parent agent's goal

        Returns:
            List of subtask results
        """
        if self.on_progress:
            self.on_progress(
                0,
                "parallel",
                f"Running {len(subtasks)} subtasks in parallel",
                ", ".join(st.description[:30] for st in subtasks),
            )

        # Run all subtasks concurrently with parent context
        tasks = [
            self._run_subtask(subtask, parent_context, parent_goal)
            for subtask in subtasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "id": subtasks[i].id,
                        "description": subtasks[i].description,
                        "status": "failed",
                        "result": None,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _merge_subtask_results(
        self, goal: str, results: list[dict[str, Any]]
    ) -> str:
        """
        Merge results from parallel subtasks into a unified response.

        Returns:
            Merged summary string
        """
        # Format results for the prompt
        results_text = ""
        for r in results:
            status = "✓" if r["status"] == "completed" else "✗"
            results_text += f"\n[{status}] {r['description']}:\n"
            if r["result"]:
                results_text += f"  {r['result']}\n"
            if r["error"]:
                results_text += f"  Error: {r['error']}\n"

        prompt = MERGE_SUBTASKS_PROMPT.format(
            goal=goal,
            subtask_results=results_text,
        )

        try:
            response = await call_llm_json_async(prompt)
            return response.get("summary", "Subtasks completed. See details above.")
        except Exception as e:
            logger.warning(f"Failed to merge subtask results: {e}")
            # Fallback: simple concatenation
            return "\n".join(
                f"- {r['description']}: {r['result'] or r['error']}" for r in results
            )

    async def run(self, goal: str) -> AgentState:
        """
        Execute the ReAct loop for a given goal.

        If the task can be decomposed into parallel subtasks, sub-agents
        will be spawned to handle them concurrently.

        Args:
            goal: Natural language description of what to accomplish

        Returns:
            AgentState with full execution history
        """
        state = AgentState(goal=goal, is_sub_agent=self._is_sub_agent)

        # Try to decompose into parallel subtasks (only for main agent)
        if not self._is_sub_agent:
            should_parallelize, subtasks = await self._should_decompose(goal)

            if should_parallelize and subtasks:
                logger.info(f"Running {len(subtasks)} parallel sub-agents")

                # Run subtasks in parallel with goal context
                results = await self._run_parallel_subtasks(
                    subtasks, state.context, goal
                )
                state.sub_agent_results = results

                # Merge results
                merged = await self._merge_subtask_results(goal, results)

                # Check if all succeeded
                all_success = all(r["status"] == "completed" for r in results)
                state.status = "completed" if all_success else "partial"
                state.final_answer = merged

                if self.on_progress:
                    self.on_progress(
                        len(subtasks),
                        "completed" if all_success else "partial",
                        f"Completed {sum(1 for r in results if r['status'] == 'completed')}/{len(results)} subtasks",
                        None,
                    )

                return state

        # Standard sequential ReAct loop
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
                        # Track reflection attempts to avoid infinite loops
                        reflection_attempts = getattr(state, "_reflection_attempts", 0)
                        max_reflection_attempts = 2  # Don't reflect more than twice

                        if reflection_attempts < max_reflection_attempts:
                            state._reflection_attempts = reflection_attempts + 1
                            reflection = await self._reflect(state)

                            if not reflection["verified"]:
                                # Check if last step was a successful file operation
                                last_step = state.steps[-1] if state.steps else None
                                was_file_success = (
                                    last_step
                                    and last_step.result
                                    and last_step.result.status == "success"
                                    and last_step.action
                                    and last_step.action.tool in ("shell", "python")
                                )

                                # If file op succeeded but reflection failed, trust the agent
                                if was_file_success and reflection_attempts > 0:
                                    logger.info(
                                        "Trusting agent completion after successful file operation"
                                    )
                                else:
                                    # Agent was wrong, continue
                                    logger.info(
                                        f"Reflection failed: {reflection['reason']}"
                                    )
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
                        else:
                            # Max reflection attempts reached, trust the agent
                            logger.info(
                                "Max reflection attempts reached, trusting agent completion"
                            )
                    break

                # Act: Execute the chosen action
                if action:
                    # Check for repeated similar commands
                    repeat_reason = self._is_repeated_command(action, state.steps)
                    if repeat_reason:
                        logger.warning(f"Detected repeated command: {repeat_reason}")
                        state.status = "completed"
                        # Generate context-aware completion message
                        state.final_answer = self._generate_stuck_message(
                            state, repeat_reason
                        )
                        state.steps.append(step)
                        break

                    result = await self._execute_action(action, state.context)
                    step.result = result

                    # Update context with result
                    if result.status == "success" and result.output is not None:
                        # Use action description or tool name as key
                        key = self._make_context_key(action, iteration)
                        state.context[key] = result.output
                        consecutive_failures = 0
                        recovery_attempts = 0  # Reset recovery counter on success
                    else:
                        consecutive_failures += 1

                        # Attempt automatic recovery before giving up
                        if consecutive_failures < MAX_CONSECUTIVE_FAILURES:
                            recovery_attempts = getattr(state, "_recovery_attempts", 0)
                            max_recovery = settings.max_recovery_attempts

                            if recovery_attempts < max_recovery:
                                # Notify user we're trying a different approach
                                if self.on_progress:
                                    self.on_progress(
                                        iteration,
                                        "retrying",
                                        "Trying a different approach...",
                                        action.tool if action else "",
                                    )

                                (
                                    recovery_action,
                                    user_msg,
                                ) = await self._attempt_recovery(
                                    state,
                                    action,
                                    result.error or "Unknown error",
                                    recovery_attempts + 1,
                                )

                                if recovery_action:
                                    # Store recovery attempt count
                                    state._recovery_attempts = recovery_attempts + 1
                                    # Execute the recovery action
                                    recovery_result = await self._execute_action(
                                        recovery_action, state.context
                                    )

                                    if recovery_result.status == "success":
                                        # Recovery succeeded!
                                        step.result = recovery_result
                                        step.action = recovery_action
                                        key = self._make_context_key(
                                            recovery_action, iteration
                                        )
                                        state.context[key] = recovery_result.output
                                        consecutive_failures = 0
                                        logger.info(
                                            f"Recovery successful on attempt {recovery_attempts + 1}"
                                        )
                                elif user_msg:
                                    # Recovery gave up with a message
                                    step.result = StepResult(
                                        step_id="recovery",
                                        status="error",
                                        error=user_msg,
                                    )

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
    ) -> tuple[Thought, Action | None, str | None]:
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
                : settings.context_limit_short
            ],  # Limit context size
            cwd=os.getcwd(),
            platform=f"{platform.system()} {platform.release()}",
        )

        try:
            response = await call_llm_json_async(prompt)

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

    async def _check_safety(self, action: Action) -> tuple[bool, str | None]:
        """
        Check if an action is safe to execute.

        Returns:
            Tuple of (is_safe, error_message)
            If dangerous but confirmed, returns (True, None)
            If blocked or user declined, returns (False, error_message)
        """
        if action.tool == "shell":
            command = action.args.get("command", "")

            # First check: File permission validation
            access_level, blocked_paths = validate_command_paths(command)

            if access_level == AccessLevel.SENSITIVE:
                return (
                    False,
                    f"🔒 Access denied: {get_permission_error_message(blocked_paths[0], access_level)}",
                )

            if access_level == AccessLevel.DENIED:
                return (
                    False,
                    f"🚫 Path not allowed: {', '.join(blocked_paths)}",
                )

            if access_level == AccessLevel.NEEDS_CONFIRMATION:
                if self.on_confirm:
                    message = (
                        f"⚠️ PATH ACCESS CONFIRMATION\n\n"
                        f"The command wants to access:\n"
                        f"  {', '.join(blocked_paths)}\n\n"
                        f"These paths are outside your allowed directories.\n"
                        f"Do you want to allow this? (y/N)"
                    )
                    confirmed = await self.on_confirm(
                        command, "Path outside allowed directories", message
                    )
                    if not confirmed:
                        return (
                            False,
                            f"❌ Path access denied by user: {', '.join(blocked_paths)}",
                        )
                elif self.require_confirmation:
                    return (
                        False,
                        f"🚫 Path access requires confirmation: {', '.join(blocked_paths)}",
                    )

            # Second check: Command safety analysis
            danger_level, reason = analyze_command(command)

            if danger_level == DangerLevel.BLOCKED:
                return False, f"🚫 Operation blocked: {reason}"

            if danger_level in (DangerLevel.DANGEROUS, DangerLevel.WARNING):
                if not self.require_confirmation:
                    logger.warning(f"Skipping confirmation (disabled): {reason}")
                    return True, None

                if self.on_confirm:
                    affected_paths = get_affected_paths(command)
                    message = format_confirmation_message(
                        command, danger_level, reason, affected_paths
                    )

                    confirmed = await self.on_confirm(command, reason, message)
                    if not confirmed:
                        return False, f"❌ Operation cancelled by user: {reason}"
                else:
                    # No confirmation callback - block dangerous operations
                    return (
                        False,
                        f"🚫 Dangerous operation requires confirmation: {reason}",
                    )

        elif action.tool == "python":
            code = action.args.get("code", "")
            danger_level, reason = analyze_python_code(code)

            if danger_level == DangerLevel.BLOCKED:
                return False, f"🚫 Operation blocked: {reason}"

            if danger_level == DangerLevel.DANGEROUS:
                if not self.require_confirmation:
                    logger.warning(f"Skipping confirmation (disabled): {reason}")
                    return True, None

                if self.on_confirm:
                    message = f"⚠️ CONFIRMATION REQUIRED\n\nReason: {reason}\n\nDo you want to proceed? (y/N)"
                    confirmed = await self.on_confirm(code[:200], reason, message)
                    if not confirmed:
                        return False, f"❌ Operation cancelled by user: {reason}"
                else:
                    return (
                        False,
                        f"🚫 Dangerous operation requires confirmation: {reason}",
                    )

        return True, None

    async def _execute_action(
        self, action: Action, context: dict[str, Any]
    ) -> StepResult:
        """Execute a single action and return the result."""
        try:
            # Safety check before execution
            is_safe, error = await self._check_safety(action)
            if not is_safe:
                return StepResult(
                    step_id=f"action_{action.tool}",
                    status="error",
                    error=error,
                )

            # Handle Python code execution
            if action.tool == "python":
                code = action.args.get("code", "")
                full_code = self._inject_context(code, context)
                result = await self.sandbox.run_python(full_code)

                if result.get("error"):
                    return StepResult(
                        step_id=f"action_{action.tool}",
                        status="error",
                        output=None,  # Don't expose raw output on error
                        error=_sanitize_error(result.get("error"), "Python script"),
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
                cwd = os.path.expanduser(cwd) if cwd else os.path.expanduser("~")

                # Expand ~ in command itself
                command = command.replace("~/", os.path.expanduser("~") + "/")

                try:
                    proc_result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        timeout=settings.shell_timeout,
                        cwd=cwd,
                        env={**os.environ, "HOME": os.path.expanduser("~")},
                    )

                    output = proc_result.stdout.decode(errors="replace")
                    stderr = proc_result.stderr.decode(errors="replace")

                    # Non-zero exit isn't always an error (e.g., grep no match)
                    # Return both stdout and stderr for context
                    if proc_result.returncode != 0:
                        raw_error = (
                            f"Exit {proc_result.returncode}: {stderr}"
                            if stderr
                            else f"Exit {proc_result.returncode}"
                        )
                        return StepResult(
                            step_id="shell",
                            status="error",
                            output=None,  # Don't expose raw output on error
                            error=_sanitize_error(raw_error, "shell command"),
                        )

                    # Only return stdout on success, ignore stderr (warnings/progress)
                    return StepResult(
                        step_id="shell",
                        status="success",
                        output=output.strip() or "(no output)",
                    )
                except subprocess.TimeoutExpired:
                    return StepResult(
                        step_id="shell",
                        status="error",
                        error="Command timed out after 2 minutes",
                    )
                except Exception as e:
                    return StepResult(
                        step_id="shell",
                        status="error",
                        error=_sanitize_error(str(e), "shell command"),
                    )

            # Handle web search
            if action.tool == "web_search":
                query = action.args.get("query", "")
                max_results = action.args.get("max_results", 5)
                result = web_search(query, max_results=max_results)

                if result.get("error"):
                    return StepResult(
                        step_id="web_search",
                        status="error",
                        error=result["error"],
                    )

                return StepResult(
                    step_id="web_search",
                    status="success",
                    output=result,
                )

            # Handle webpage fetching
            if action.tool == "fetch_webpage":
                url = action.args.get("url", "")
                result = fetch_webpage(url)

                if result.get("error"):
                    return StepResult(
                        step_id="fetch_webpage",
                        status="error",
                        error=result["error"],
                    )

                return StepResult(
                    step_id="fetch_webpage",
                    status="success",
                    output=result,
                )

            # Unknown tool
            return StepResult(
                step_id=f"action_{action.tool}",
                status="error",
                error=f"Unknown tool: {action.tool}. Use 'shell', 'python', 'web_search', or 'fetch_webpage'.",
            )

        except Exception as e:
            return StepResult(
                step_id=f"action_{action.tool}",
                status="error",
                error=_sanitize_error(str(e), action.tool),
            )

    async def _reflect(self, state: AgentState) -> dict[str, Any]:
        """
        Reflect on whether the goal was actually achieved.

        Returns:
            Dict with "verified" (bool), "reason" (str), and "summary" (str)
        """
        prompt = REFLECTION_PROMPT.format(
            goal=state.goal,
            steps_summary=self._summarize_steps(state),
            final_context=json.dumps(state.context, indent=2, default=str)[
                : settings.context_limit_medium
            ],
        )

        try:
            response = await call_llm_json_async(prompt)
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
            # Default to NOT verified if reflection fails - be conservative
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": "Reflection check failed - please verify manually",
                "summary": "Task may be incomplete. Please check the results.",
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
        # Show last 10 exchanges (20 messages) for better context
        max_messages = min(20, len(self.conversation_history))
        recent = self.conversation_history[-max_messages:]

        if len(self.conversation_history) > max_messages:
            lines.append(
                f"[Earlier: {len(self.conversation_history) - max_messages} messages omitted]"
            )

        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages but keep more context
            if len(content) > 500:
                content = content[:500] + "..."
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
            # Return sanitized error message
            return f"ERROR: {result.error}"  # Already sanitized by _sanitize_error

    def _make_context_key(self, action: Action, iteration: int) -> str:
        """Generate a context key for storing action results."""
        # Try to use a descriptive name
        if action.description:
            # Convert description to snake_case
            key = action.description.lower().replace(" ", "_")[:20]
            return f"{key}_{iteration}"
        return f"{action.tool}_result_{iteration}"

    def _inject_context(self, code: str, context: dict[str, Any]) -> str:
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

    async def _attempt_recovery(
        self,
        state: AgentState,
        failed_action: Action,
        error: str,
        attempt: int,
    ) -> tuple[Action | None, str | None]:
        """
        Attempt to recover from a failed action by asking LLM for alternative approach.

        Args:
            state: Current agent state
            failed_action: The action that failed
            error: The error message
            attempt: Current recovery attempt number

        Returns:
            Tuple of (new_action, user_message) where new_action is None if recovery failed
        """
        max_attempts = settings.max_recovery_attempts

        # Build history summary
        history = self._summarize_steps(state)

        # Get failed command details
        failed_tool = failed_action.tool
        if failed_tool == "shell":
            failed_command = failed_action.args.get("command", "")
        elif failed_tool == "python":
            failed_command = failed_action.args.get("code", "")[:200]
        else:
            failed_command = str(failed_action.args)

        prompt = ERROR_RECOVERY_PROMPT.format(
            goal=state.goal,
            attempt=attempt,
            max_attempts=max_attempts,
            failed_tool=failed_tool,
            failed_command=failed_command,
            error=error,
            history=history if history else "(no previous steps)",
        )

        try:
            response = await call_llm_json_async(prompt)

            if response.get("give_up"):
                user_message = response.get(
                    "user_message",
                    f"Unable to complete task after {attempt} attempts: {error}",
                )
                return None, user_message

            action_data = response.get("action")
            if action_data:
                new_action = Action(
                    tool=action_data.get("tool", "shell"),
                    args=action_data.get("args", {}),
                    description=response.get("new_approach", "Recovery attempt"),
                )
                logger.info(
                    f"Recovery attempt {attempt}: {new_action.tool} - {new_action.description}"
                )
                return new_action, None

        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")

        return None, f"Recovery failed: {error}"

    def _is_repeated_command(
        self, action: Action, steps: list[AgentStep]
    ) -> str | None:
        """Check if this command is essentially repeating previous commands.

        Detects:
        - exact_repeat: Same command executed 2+ times consecutively
        - search_loop: Multiple failed search attempts (3+ times)

        Note: "Similar command" detection is intentionally conservative to avoid
        false positives when the agent legitimately runs multiple shell commands
        or Python scripts as part of a multi-step task.

        Returns:
            None if not repeated, or a reason string if stuck.
        """
        if not steps or len(steps) < 2:
            return None

        # Only check shell commands for repeats - Python code varies too much
        if action.tool != "shell":
            return None

        current_cmd = action.args.get("command", "")
        if not current_cmd:
            return None

        # Normalize command for comparison
        current_cmd_normalized = current_cmd.strip().lower()
        current_words = current_cmd.split()
        current_base = current_words[0] if current_words else ""

        # Count patterns in recent steps
        exact_repeat_count = 0
        failed_search_count = 0
        search_commands = {
            "ls",
            "find",
            "locate",
            "grep",
        }

        # Check last 6 steps for patterns
        for step in steps[-6:]:
            if not step.action:
                continue

            # Only compare with shell commands
            if step.action.tool != "shell":
                continue

            prev_cmd = step.action.args.get("command", "")
            if not prev_cmd:
                continue

            prev_cmd_normalized = prev_cmd.strip().lower()
            prev_words = prev_cmd.split()
            prev_base = prev_words[0] if prev_words else ""

            # Track result status
            step_failed = step.result is not None and (
                step.result.status != "success" or not step.result.output
            )

            # Exact repeat detection (case-insensitive)
            if current_cmd_normalized == prev_cmd_normalized:
                exact_repeat_count += 1

            # Search loop detection - only for failed search commands
            if (
                current_base in search_commands
                and prev_base in search_commands
                and step_failed
            ):
                failed_search_count += 1

        # Thresholds for detecting loops
        if exact_repeat_count >= 2:
            logger.warning(
                f"Exact repeat detected: '{current_cmd[:50]}...' repeated {exact_repeat_count + 1} times"
            )
            return "exact_repeat"

        if failed_search_count >= 3:
            logger.warning(
                f"Search loop detected: {failed_search_count} failed search attempts"
            )
            return "search_loop"

        return None

    def _generate_stuck_message(self, state: AgentState, reason: str) -> str:
        """Generate a context-appropriate message when the agent gets stuck."""
        # Check if we have any successful results to summarize
        successful_results = []
        for step in state.steps:
            if step.result and step.result.status == "success" and step.result.output:
                successful_results.append(step)

        if successful_results:
            # We did something successfully - summarize what was done
            last_success = successful_results[-1]
            action_desc = ""
            result = last_success.result  # Already verified non-None in the filter
            if last_success.action and result:
                output_text = str(result.output) if result.output else ""
                if last_success.action.tool == "shell":
                    cmd = last_success.action.args.get("command", "")
                    if any(
                        w in cmd
                        for w in ["touch", "echo", "mkdir", "cp", "mv", ">", ">>"]
                    ):
                        action_desc = f"I executed the command and it completed successfully. Output: {output_text[:500] or 'No output'}"
                    else:
                        action_desc = f"Here's what I found: {output_text[:500] or 'Command completed.'}"
                elif last_success.action.tool == "python":
                    action_desc = f"The Python code executed successfully. Result: {output_text[:500] or 'No output'}"
                else:
                    action_desc = (
                        f"Completed successfully: {output_text[:200] or 'Done'}"
                    )
            return action_desc or "The task was completed."

        # No successes - give appropriate message based on reason
        reason_messages = {
            "search_loop": "I searched but couldn't find what you're looking for. The file may not exist, have a different name, or be in a different location.",
            "exact_repeat": "I detected a repeated command pattern. The previous attempts didn't produce the expected result. Could you provide more specific details or try a different approach?",
        }
        return reason_messages.get(
            reason,
            "I wasn't able to complete this task. Please try with more specific instructions.",
        )


# Re-export models for backward compatibility
__all__ = [
    "ReActAgent",
    "AgentState",
    "AgentStep",
    "Action",
    "Thought",
    "Observation",
    "ProgressCallback",
    "ConfirmCallback",
    "SubAgentState",
    "SubTask",
    "MAX_PARALLEL_SUBTASKS",
    "SUB_AGENT_MAX_ITERATIONS",
]
