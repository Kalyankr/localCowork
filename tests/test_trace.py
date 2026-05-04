"""Tests for the Trace optimization integration.

Covers:
- evaluation.py: scoring logic, aggregation
- trainer.py: parameter save / load / clear
- trainable_agent.py: trainable agent construction & helpers
- react_agent.py: resilient load_params usage
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.trace.evaluation import (
    EvalTask,
    TaskResult,
    aggregate_scores,
    score_result,
)
from agent.trace.trainer import clear_params, load_params, save_params


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(**overrides) -> EvalTask:
    defaults = dict(
        id="t1",
        goal="Do something",
        category="shell",
        expected_keywords=["ok"],
        expected_tool="shell",
        max_steps=2,
        timeout=60.0,
    )
    defaults.update(overrides)
    return EvalTask(**defaults)


def _make_result(**overrides) -> TaskResult:
    defaults = dict(
        task_id="t1",
        success=True,
        steps_taken=1,
        wall_time=5.0,
        agent_status="completed",
        final_answer="ok done",
        tools_used=["shell"],
        errors=[],
    )
    defaults.update(overrides)
    return TaskResult(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation scoring
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreResult:
    """Tests for score_result()."""

    def test_perfect_score(self):
        task = _make_task()
        result = _make_result()
        scored = score_result(task, result)

        assert scored["score"] == 1.0
        assert scored["breakdown"]["completion"] == 1.0
        assert scored["breakdown"]["keyword_match"] == 1.0
        assert scored["breakdown"]["efficiency"] == 1.0
        assert scored["breakdown"]["tool_choice"] == 1.0
        assert scored["breakdown"]["speed"] == 1.0
        assert "successfully" in scored["feedback"]

    def test_failed_task_scores_low(self):
        task = _make_task()
        result = _make_result(success=False, agent_status="failed", errors=["boom"])
        scored = score_result(task, result)

        assert scored["breakdown"]["completion"] == 0.0
        assert scored["score"] < 0.7
        assert "FAILED" in scored["feedback"]

    def test_missing_keywords_penalized(self):
        task = _make_task(expected_keywords=["alpha", "beta"])
        result = _make_result(final_answer="only alpha here")
        scored = score_result(task, result)

        assert scored["breakdown"]["keyword_match"] == 0.5
        assert "beta" in scored["feedback"]

    def test_no_keywords_gives_full_score(self):
        task = _make_task(expected_keywords=[])
        result = _make_result(final_answer="anything")
        scored = score_result(task, result)

        assert scored["breakdown"]["keyword_match"] == 1.0

    def test_empty_keyword_strings_ignored(self):
        task = _make_task(expected_keywords=[""])
        result = _make_result(final_answer="whatever")
        scored = score_result(task, result)

        # [""] is truthy but has no non-empty keywords → 0 matches / max(0,1) = 0.0
        assert scored["breakdown"]["keyword_match"] == 0.0

    def test_too_many_steps_penalized(self):
        task = _make_task(max_steps=2)
        result = _make_result(steps_taken=5)
        scored = score_result(task, result)

        assert scored["breakdown"]["efficiency"] < 1.0

    def test_efficiency_floors_at_zero(self):
        task = _make_task(max_steps=1)
        result = _make_result(steps_taken=100)
        scored = score_result(task, result)

        assert scored["breakdown"]["efficiency"] == 0.0

    def test_wrong_tool_penalized(self):
        task = _make_task(expected_tool="python")
        result = _make_result(tools_used=["shell"])
        scored = score_result(task, result)

        assert scored["breakdown"]["tool_choice"] == 0.3
        assert "python" in scored["feedback"]

    def test_no_expected_tool_gives_full_score(self):
        task = _make_task(expected_tool=None)
        result = _make_result(tools_used=["anything"])
        scored = score_result(task, result)

        assert scored["breakdown"]["tool_choice"] == 1.0

    def test_slow_task_penalized(self):
        task = _make_task(timeout=10.0)
        result = _make_result(wall_time=8.0)  # > 50% of timeout
        scored = score_result(task, result)

        assert scored["breakdown"]["speed"] == 0.7

    def test_none_final_answer_keyword_handling(self):
        task = _make_task(expected_keywords=["word"])
        result = _make_result(final_answer=None)
        scored = score_result(task, result)

        assert scored["breakdown"]["keyword_match"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregate scores
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregateScores:
    def test_empty_list(self):
        agg = aggregate_scores([])
        assert agg["mean_score"] == 0.0
        assert agg["total"] == 0

    def test_single_score(self):
        scores = [score_result(_make_task(), _make_result())]
        agg = aggregate_scores(scores)
        assert agg["total"] == 1
        assert agg["mean_score"] == 1.0
        assert agg["pass_rate"] == 1.0

    def test_mixed_scores(self):
        task = _make_task()
        good = score_result(task, _make_result())
        bad = score_result(
            task,
            _make_result(success=False, agent_status="failed", final_answer=None),
        )
        agg = aggregate_scores([good, bad])
        assert agg["total"] == 2
        assert 0.0 < agg["mean_score"] < 1.0
        assert "dimension_averages" in agg


# ═══════════════════════════════════════════════════════════════════════════
#  Parameter persistence (save / load / clear)
# ═══════════════════════════════════════════════════════════════════════════


class TestParamPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.trace.trainer._PARAMS_DIR", tmp_path / "trace_params"
        )
        params = {"system_identity": "I am optimized", "react_instruction": "Be fast"}
        save_params(params)

        loaded = load_params()
        assert loaded == params

    def test_load_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.trace.trainer._PARAMS_DIR", tmp_path / "nonexistent")
        assert load_params() is None

    def test_clear_removes_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.trace.trainer._PARAMS_DIR", tmp_path / "trace_params"
        )
        save_params({"key": "value"})
        clear_params()
        assert load_params() is None

    def test_clear_noop_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.trace.trainer._PARAMS_DIR", tmp_path / "nonexistent")
        clear_params()  # should not raise

    def test_load_handles_corrupted_json(self, tmp_path, monkeypatch):
        params_dir = tmp_path / "trace_params"
        params_dir.mkdir(parents=True)
        (params_dir / "optimized_params.json").write_text("{bad json!!!")
        monkeypatch.setattr("agent.trace.trainer._PARAMS_DIR", params_dir)

        # Should not crash — returns None on corrupt file
        assert load_params() is None


# ═══════════════════════════════════════════════════════════════════════════
#  Trainable agent (requires trace-opt)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainableAgent:
    def test_trace_available(self):
        from agent.trace.trainable_agent import TRACE_AVAILABLE

        assert TRACE_AVAILABLE is True

    def test_create_agent(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        overrides = agent.get_prompt_overrides()

        assert "system_identity" in overrides
        assert "react_instruction" in overrides
        assert "reflection_instruction" in overrides
        assert "tool_selection_instruction" in overrides
        assert "error_recovery_instruction" in overrides

    def test_classify_task_intent(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        result = agent.classify_task_intent("read the file config.yaml")
        # Result is a trace node; extract .data
        intent = result.data if hasattr(result, "data") else result
        assert intent == "file_ops"

    def test_classify_conversation(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        result = agent.classify_task_intent("hello how are you")
        intent = result.data if hasattr(result, "data") else result
        assert intent == "conversation"

    def test_should_reflect_trivial_success(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        result = agent.should_reflect(1, "success")
        val = result.data if hasattr(result, "data") else result
        assert val is False

    def test_should_reflect_multi_step(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        result = agent.should_reflect(3, "success")
        val = result.data if hasattr(result, "data") else result
        assert val is True

    def test_format_observation_short(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        result = agent.format_observation_for_llm("hello world", "shell")
        val = result.data if hasattr(result, "data") else result
        assert val == "hello world"

    def test_format_observation_truncates_long(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        long_output = "\n".join(f"line {i}" for i in range(50))
        result = agent.format_observation_for_llm(long_output, "shell")
        val = result.data if hasattr(result, "data") else result
        assert "truncated" in val

    def test_get_prompt_overrides_returns_strings(self):
        from agent.trace.trainable_agent import create_trainable_agent

        agent = create_trainable_agent()
        overrides = agent.get_prompt_overrides()
        for key, val in overrides.items():
            assert isinstance(val, str), f"{key} should be a string, got {type(val)}"


# ═══════════════════════════════════════════════════════════════════════════
#  ReAct agent resilience — load_params crash safety
# ═══════════════════════════════════════════════════════════════════════════


class TestReActAgentParamLoading:
    def test_agent_works_without_saved_params(self, mock_sandbox, monkeypatch):
        """Agent should start fine when no optimized params exist."""
        monkeypatch.setattr(
            "agent.trace.trainer._PARAMS_DIR", Path("/tmp/nonexistent_trace_dir")
        )
        from agent.orchestrator.react_agent import ReActAgent

        agent = ReActAgent(sandbox=mock_sandbox, require_confirmation=False)
        assert "system_identity" in agent._prompt_params

    def test_agent_survives_corrupted_params(self, mock_sandbox, tmp_path, monkeypatch):
        """Agent should fall back to defaults if params file is corrupt."""
        params_dir = tmp_path / "bad_params"
        params_dir.mkdir()
        (params_dir / "optimized_params.json").write_text("NOT VALID JSON")
        monkeypatch.setattr("agent.trace.trainer._PARAMS_DIR", params_dir)

        from agent.orchestrator.react_agent import ReActAgent

        agent = ReActAgent(sandbox=mock_sandbox, require_confirmation=False)
        assert agent._prompt_params["system_identity"] is not None
