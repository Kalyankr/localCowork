"""Evaluation tasks and feedback functions for Trace optimization.

Provides a benchmark suite of tasks with expected outcomes, plus
feedback functions that score agent performance. The Trace optimizer
uses this feedback to improve the agent's trainable parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Task definition
# ---------------------------------------------------------------------------


@dataclass
class EvalTask:
    """A single evaluation task with expected outcome."""

    id: str
    goal: str
    category: str  # file_ops, shell, code, web, memory, conversation
    expected_keywords: list[str] = field(default_factory=list)
    expected_tool: str | None = None
    max_steps: int = 5  # ideal max steps to solve this
    timeout: float = 120.0


# ---------------------------------------------------------------------------
# Built-in benchmark tasks
# ---------------------------------------------------------------------------

EVAL_TASKS: list[EvalTask] = [
    # --- Conversation (should complete in 1 step, no tool use) ---
    EvalTask(
        id="conv_greeting",
        goal="Hey, what can you do?",
        category="conversation",
        expected_keywords=["help", "file", "search"],
        max_steps=1,
    ),
    EvalTask(
        id="conv_joke",
        goal="Tell me a programming joke",
        category="conversation",
        expected_keywords=[""],
        max_steps=1,
    ),
    # --- File operations ---
    EvalTask(
        id="file_read",
        goal="Read the contents of pyproject.toml",
        category="file_ops",
        expected_keywords=["localcowork", "dependencies"],
        expected_tool="read_file",
        max_steps=2,
    ),
    EvalTask(
        id="file_list",
        goal="List all Python files in the agent/ directory",
        category="file_ops",
        expected_keywords=[".py"],
        expected_tool="shell",
        max_steps=2,
    ),
    EvalTask(
        id="file_write",
        goal="Create a file called /tmp/trace_test.txt with the text 'hello trace'",
        category="file_ops",
        expected_keywords=["trace_test.txt", "wrote"],
        expected_tool="write_file",
        max_steps=2,
    ),
    # --- Shell ---
    EvalTask(
        id="shell_pwd",
        goal="What is the current working directory?",
        category="shell",
        expected_keywords=["/"],
        expected_tool="shell",
        max_steps=2,
    ),
    EvalTask(
        id="shell_python_version",
        goal="What Python version is installed?",
        category="shell",
        expected_keywords=["python", "3."],
        expected_tool="shell",
        max_steps=2,
    ),
    # --- Code / data ---
    EvalTask(
        id="code_calc",
        goal="Calculate the sum of numbers from 1 to 100 using Python",
        category="code",
        expected_keywords=["5050"],
        expected_tool="python",
        max_steps=2,
    ),
    # --- Web ---
    EvalTask(
        id="web_search",
        goal="Search the web for 'Python asyncio tutorial'",
        category="web",
        expected_keywords=["asyncio"],
        expected_tool="web_search",
        max_steps=2,
    ),
    # --- Memory ---
    EvalTask(
        id="memory_store",
        goal="Remember that the project uses pytest for testing",
        category="memory",
        expected_keywords=["remember", "pytest"],
        expected_tool="memory_store",
        max_steps=2,
    ),
]


# ---------------------------------------------------------------------------
# Feedback scoring
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Result of running a single eval task."""

    task_id: str
    success: bool
    steps_taken: int
    wall_time: float  # seconds
    agent_status: str  # completed / failed / max_iterations
    final_answer: str | None
    tools_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def score_result(task: EvalTask, result: TaskResult) -> dict[str, Any]:
    """Compute a multi-dimensional score for a task result.

    Returns a dict with:
        score: float 0-1 (overall quality)
        feedback: str   (natural language for the Trace optimizer)
        breakdown: dict (individual scoring dimensions)
    """
    breakdown: dict[str, float] = {}

    # 1. Completion: did the agent finish successfully?
    completion = 1.0 if result.success else 0.0
    breakdown["completion"] = completion

    # 2. Keyword match: does the answer contain expected keywords?
    keyword_score = 0.0
    if result.final_answer and task.expected_keywords:
        answer_lower = result.final_answer.lower()
        matches = sum(
            1 for kw in task.expected_keywords if kw and kw.lower() in answer_lower
        )
        non_empty = [kw for kw in task.expected_keywords if kw]
        keyword_score = matches / max(len(non_empty), 1)
    elif not task.expected_keywords:
        keyword_score = 1.0  # no keywords to check
    breakdown["keyword_match"] = keyword_score

    # 3. Efficiency: fewer steps is better
    if result.steps_taken <= task.max_steps:
        efficiency = 1.0
    else:
        overshoot = result.steps_taken - task.max_steps
        efficiency = max(0.0, 1.0 - overshoot * 0.2)
    breakdown["efficiency"] = efficiency

    # 4. Tool correctness: did it use the expected tool?
    tool_score = 1.0
    if task.expected_tool:
        tool_score = 1.0 if task.expected_tool in result.tools_used else 0.3
    breakdown["tool_choice"] = tool_score

    # 5. Speed: bonus for fast completion
    speed_score = 1.0 if result.wall_time < task.timeout * 0.5 else 0.7
    breakdown["speed"] = speed_score

    # Weighted overall score
    score = (
        completion * 0.35
        + keyword_score * 0.25
        + efficiency * 0.20
        + tool_score * 0.10
        + speed_score * 0.10
    )

    # Build natural language feedback for the optimizer
    feedback_parts: list[str] = []
    if not result.success:
        feedback_parts.append(
            f"Task FAILED (status={result.agent_status}). "
            f"Errors: {'; '.join(result.errors) if result.errors else 'unknown'}."
        )
    else:
        feedback_parts.append("Task completed successfully.")

    if keyword_score < 1.0 and task.expected_keywords:
        missing = [
            kw
            for kw in task.expected_keywords
            if kw and kw.lower() not in (result.final_answer or "").lower()
        ]
        if missing:
            feedback_parts.append(
                f"Answer missing expected content: {', '.join(missing)}."
            )

    if efficiency < 1.0:
        feedback_parts.append(
            f"Took {result.steps_taken} steps (ideal: <={task.max_steps}). "
            "Try to be more direct."
        )

    if tool_score < 1.0:
        feedback_parts.append(
            f"Expected tool '{task.expected_tool}' but used: "
            f"{', '.join(result.tools_used) or 'none'}."
        )

    feedback = " ".join(feedback_parts)

    return {
        "score": round(score, 3),
        "feedback": feedback,
        "breakdown": breakdown,
    }


def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate stats from a list of score dicts."""
    if not scores:
        return {"mean_score": 0.0, "pass_rate": 0.0, "total": 0}

    total = len(scores)
    mean_score = sum(s["score"] for s in scores) / total
    pass_rate = sum(1 for s in scores if s["score"] >= 0.7) / total

    # Per-dimension averages
    dims = scores[0]["breakdown"].keys()
    dim_avgs = {
        dim: round(sum(s["breakdown"][dim] for s in scores) / total, 3) for dim in dims
    }

    return {
        "mean_score": round(mean_score, 3),
        "pass_rate": round(pass_rate, 3),
        "total": total,
        "dimension_averages": dim_avgs,
    }
