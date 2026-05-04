"""Trainable ReAct agent using Microsoft Trace.

Wraps the core prompts, tool-selection logic, and reasoning steps
as Trace-optimizable parameters. An LLM-based optimizer (OptoPrime)
can then rewrite these parameters based on task feedback.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    from opto import trace
    from opto.optimizers import OptoPrime

    TRACE_AVAILABLE = True
except ImportError:
    TRACE_AVAILABLE = False


def _require_trace() -> None:
    if not TRACE_AVAILABLE:
        raise ImportError(
            "trace-opt is not installed. Install with: pip install localcowork[trace]"
        )


# ---------------------------------------------------------------------------
# Default prompt values (mirrors agent/llm/prompts.py originals)
# These are the starting points; the optimizer will rewrite them.
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_IDENTITY = (
    "You are LocalCowork, an AI assistant with full access to the user's machine."
)

_DEFAULT_REACT_INSTRUCTION = (
    "Most tasks need 1-2 commands. Don't explore - act directly."
)

_DEFAULT_REFLECTION_INSTRUCTION = (
    "Verify if the goal was achieved. "
    "Check the steps taken and the final context carefully."
)

_DEFAULT_TOOL_SELECTION_INSTRUCTION = (
    "Given a user goal, select the minimal set of tools needed. "
    "Prefer direct actions over exploration."
)

_DEFAULT_ERROR_RECOVERY_INSTRUCTION = (
    "The previous approach failed. Try a DIFFERENT approach. "
    "Analyze WHY the command failed and consider alternative methods."
)


# ---------------------------------------------------------------------------
# Trainable Agent Model
# ---------------------------------------------------------------------------


def create_trainable_agent(
    optimizer_model: str = "gpt-4o",
) -> "TrainableReActAgent":
    """Create a trainable agent with Trace-wrapped parameters.

    Args:
        optimizer_model: LLM model the Trace optimizer uses internally
                         to propose parameter updates (default: gpt-4o).
    """
    _require_trace()
    return TrainableReActAgent(optimizer_model=optimizer_model)


if TRACE_AVAILABLE:

    @trace.model
    class TrainableReActAgent:
        """ReAct agent with Trace-trainable parameters.

        Each ``trace.node(…, trainable=True)`` is a parameter the optimizer
        can rewrite.  The ``@trace.bundle(trainable=True)`` methods are
        Python functions whose *code* can be rewritten.
        """

        def __init__(self, optimizer_model: str = "gpt-4o") -> None:
            # --- Trainable prompt fragments ---------------------------------
            self.system_identity = trace.node(
                _DEFAULT_SYSTEM_IDENTITY,
                trainable=True,
                description="System identity prompt shown at top of every ReAct step.",
            )
            self.react_instruction = trace.node(
                _DEFAULT_REACT_INSTRUCTION,
                trainable=True,
                description="Core instruction guiding how the agent approaches tasks.",
            )
            self.reflection_instruction = trace.node(
                _DEFAULT_REFLECTION_INSTRUCTION,
                trainable=True,
                description="Instruction for the reflection/verification step.",
            )
            self.tool_selection_instruction = trace.node(
                _DEFAULT_TOOL_SELECTION_INSTRUCTION,
                trainable=True,
                description="Instruction guiding which tools to pick for a task.",
            )
            self.error_recovery_instruction = trace.node(
                _DEFAULT_ERROR_RECOVERY_INSTRUCTION,
                trainable=True,
                description="Instruction for recovering from failed actions.",
            )
            self.optimizer_model = optimizer_model

        # --- Trainable functions -------------------------------------------

        @trace.bundle(trainable=True)
        def classify_task_intent(self, goal: str) -> str:
            """Classify a user goal into an intent category.

            Categories: file_ops, shell, code, web, memory, conversation, multi.
            Return the single best category as a string.
            """
            goal_lower = goal.lower()
            if any(
                w in goal_lower for w in ("file", "read", "write", "edit", "create")
            ):
                return "file_ops"
            if any(w in goal_lower for w in ("search", "web", "url", "http")):
                return "web"
            if any(w in goal_lower for w in ("run", "install", "list", "find", "git")):
                return "shell"
            if any(
                w in goal_lower for w in ("python", "script", "data", "csv", "plot")
            ):
                return "code"
            if any(w in goal_lower for w in ("remember", "recall", "memory")):
                return "memory"
            return "conversation"

        @trace.bundle(trainable=True)
        def should_reflect(self, step_count: int, last_status: str) -> bool:
            """Decide whether to run reflection after the agent says it's done.

            Args:
                step_count: Number of steps the agent took.
                last_status: Status of the last tool execution ("success" / "error").

            Returns:
                True if reflection is warranted.
            """
            if step_count <= 1 and last_status == "success":
                return False  # trivial single-step success — skip reflection
            return True

        @trace.bundle(trainable=True)
        def format_observation_for_llm(self, raw_output: str, tool: str) -> str:
            """Condense raw tool output into an observation string for the LLM.

            Keep the most relevant information and discard noise.
            """
            lines = raw_output.strip().splitlines()
            if len(lines) > 30:
                return "\n".join(lines[:15] + ["... (truncated) ..."] + lines[-10:])
            return raw_output.strip()

        # --- Non-trainable helpers -----------------------------------------

        def get_prompt_overrides(self) -> dict[str, str]:
            """Return current (possibly optimized) prompt values.

            The caller injects these into the existing REACT_STEP_PROMPT
            template at runtime.
            """
            return {
                "system_identity": self.system_identity.data
                if hasattr(self.system_identity, "data")
                else str(self.system_identity),
                "react_instruction": self.react_instruction.data
                if hasattr(self.react_instruction, "data")
                else str(self.react_instruction),
                "reflection_instruction": self.reflection_instruction.data
                if hasattr(self.reflection_instruction, "data")
                else str(self.reflection_instruction),
                "tool_selection_instruction": self.tool_selection_instruction.data
                if hasattr(self.tool_selection_instruction, "data")
                else str(self.tool_selection_instruction),
                "error_recovery_instruction": self.error_recovery_instruction.data
                if hasattr(self.error_recovery_instruction, "data")
                else str(self.error_recovery_instruction),
            }

        def parameters(self) -> list:
            """All trainable parameters (for the optimizer)."""
            return self.__dict__  # trace.model handles this

else:
    # Stub when trace-opt is not installed
    class TrainableReActAgent:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise ImportError(
                "trace-opt is not installed. "
                "Install with: pip install localcowork[trace]"
            )
