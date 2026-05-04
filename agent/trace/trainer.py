"""Training loop: run eval tasks, collect feedback, optimize parameters.

This is the main entry point for Trace-based optimization of the
LocalCowork ReAct agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import structlog

from agent.config import settings
from agent.sandbox.sandbox_runner import Sandbox
from agent.tools.builtin import register_builtin_tools
from agent.trace.evaluation import (
    EVAL_TASKS,
    EvalTask,
    TaskResult,
    aggregate_scores,
    score_result,
)
from agent.trace.trainable_agent import (
    TRACE_AVAILABLE,
    TrainableReActAgent,
    _require_trace,
)

logger = structlog.get_logger(__name__)

# Where optimized parameters are persisted
_PARAMS_DIR = Path.home() / ".localcowork" / "trace_params"


def _ensure_params_dir() -> Path:
    _PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    return _PARAMS_DIR


# ---------------------------------------------------------------------------
# Run a single eval task through the real ReAct agent
# ---------------------------------------------------------------------------


async def _run_eval_task(
    task: EvalTask, prompt_overrides: dict[str, str]
) -> TaskResult:
    """Execute one eval task with the real ReActAgent and return structured results."""
    from agent.orchestrator.react_agent import ReActAgent

    sandbox = Sandbox()
    register_builtin_tools(sandbox)

    # Create the agent with reduced iterations for eval
    agent = ReActAgent(
        sandbox=sandbox,
        max_iterations=min(task.max_steps + 3, settings.max_agent_iterations),
        require_confirmation=False,  # non-interactive during training
    )

    errors: list[str] = []
    start = time.monotonic()

    try:
        state = await asyncio.wait_for(
            agent.run(task.goal),
            timeout=task.timeout,
        )
        wall_time = time.monotonic() - start

        tools_used = [step.action.tool for step in state.steps if step.action]

        return TaskResult(
            task_id=task.id,
            success=state.status == "completed",
            steps_taken=len(state.steps),
            wall_time=wall_time,
            agent_status=state.status,
            final_answer=state.final_answer,
            tools_used=tools_used,
            errors=[state.error] if state.error else [],
        )
    except asyncio.TimeoutError:
        return TaskResult(
            task_id=task.id,
            success=False,
            steps_taken=0,
            wall_time=task.timeout,
            agent_status="timeout",
            final_answer=None,
            errors=["Task timed out"],
        )
    except Exception as e:
        return TaskResult(
            task_id=task.id,
            success=False,
            steps_taken=0,
            wall_time=time.monotonic() - start,
            agent_status="error",
            final_answer=None,
            errors=[str(e)],
        )


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


async def run_training(
    epochs: int = 3,
    tasks: list[EvalTask] | None = None,
    optimizer_model: str = "gpt-4o",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the Trace optimization loop.

    1. For each epoch, run all eval tasks through the real agent.
    2. Score results and build natural-language feedback.
    3. Back-propagate feedback through the Trace graph.
    4. Let the optimizer update the trainable parameters.
    5. Save the best parameters to disk.

    Args:
        epochs: Number of training epochs.
        tasks: Evaluation tasks to use (defaults to built-in suite).
        optimizer_model: LLM used by the Trace optimizer.
        verbose: Print progress to stdout.

    Returns:
        Summary dict with scores per epoch and final parameters.
    """
    _require_trace()
    from opto.optimizers import OptoPrime

    eval_tasks = tasks or EVAL_TASKS
    trainable = TrainableReActAgent(optimizer_model=optimizer_model)
    optimizer = OptoPrime(trainable.parameters())

    best_score = -1.0
    best_params: dict[str, str] = {}
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Epoch {epoch}/{epochs}")
            print(f"{'=' * 60}")

        overrides = trainable.get_prompt_overrides()
        epoch_scores: list[dict[str, Any]] = []

        for i, task in enumerate(eval_tasks, 1):
            if verbose:
                print(f"  [{i}/{len(eval_tasks)}] {task.id}: {task.goal[:60]}...")

            result = await _run_eval_task(task, overrides)
            scored = score_result(task, result)
            epoch_scores.append(scored)

            if verbose:
                status = "PASS" if scored["score"] >= 0.7 else "FAIL"
                print(
                    f"    {status}  score={scored['score']}  steps={result.steps_taken}"
                )

            # --- Trace backward pass per task ---
            # We create a trace node for the score so the optimizer
            # can relate feedback to the trainable parameters.
            try:
                from opto import trace as opto_trace

                # Use classify_task_intent (a trainable bundle) so
                # the computation graph links back to trainable params.
                intent = trainable.classify_task_intent(task.goal)
                should_ref = trainable.should_reflect(
                    result.steps_taken,
                    "success" if result.success else "error",
                )

                # Back-propagate the feedback
                optimizer.zero_feedback()
                optimizer.backward(intent, scored["feedback"])
                optimizer.step()
            except Exception as e:
                logger.warning("trace_backward_failed", task=task.id, error=str(e))

        agg = aggregate_scores(epoch_scores)
        history.append({"epoch": epoch, **agg})

        if verbose:
            print(f"\n  Epoch {epoch} summary:")
            print(f"    Mean score : {agg['mean_score']}")
            print(f"    Pass rate  : {agg['pass_rate']}")
            print(f"    Dimensions : {agg.get('dimension_averages', {})}")

        # Save best
        if agg["mean_score"] > best_score:
            best_score = agg["mean_score"]
            best_params = trainable.get_prompt_overrides()

    # Persist best parameters
    if best_params:
        save_params(best_params)
        if verbose:
            print(f"\n  Best parameters saved (score={best_score})")
            print(f"  Location: {_PARAMS_DIR / 'optimized_params.json'}")

    return {
        "epochs": epochs,
        "best_score": best_score,
        "history": history,
        "optimized_params": best_params,
    }


# ---------------------------------------------------------------------------
# Parameter persistence
# ---------------------------------------------------------------------------


def save_params(params: dict[str, str]) -> Path:
    """Save optimized parameters to disk."""
    path = _ensure_params_dir() / "optimized_params.json"
    path.write_text(json.dumps(params, indent=2))
    logger.info("trace_params_saved", path=str(path))
    return path


def load_params() -> dict[str, str] | None:
    """Load optimized parameters if they exist."""
    path = _PARAMS_DIR / "optimized_params.json"
    if path.exists():
        try:
            params = json.loads(path.read_text())
            logger.info("trace_params_loaded", path=str(path))
            return params
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("trace_params_load_failed", path=str(path), error=str(e))
            return None
    return None


def clear_params() -> None:
    """Remove saved optimized parameters."""
    path = _PARAMS_DIR / "optimized_params.json"
    if path.exists():
        path.unlink()
        logger.info("trace_params_cleared")
