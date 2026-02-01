"""Tests for sub-agent parallel execution functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSubTaskModel:
    """Tests for SubTask model."""

    def test_subtask_creation(self):
        """SubTask should be created with required fields."""
        from agent.orchestrator.agent_models import SubTask

        subtask = SubTask(
            id="1",
            description="Organize Downloads folder",
        )

        assert subtask.id == "1"
        assert subtask.description == "Organize Downloads folder"
        assert subtask.dependencies == []
        assert subtask.status == "pending"
        assert subtask.result is None
        assert subtask.error is None

    def test_subtask_with_dependencies(self):
        """SubTask should track dependencies."""
        from agent.orchestrator.agent_models import SubTask

        subtask = SubTask(
            id="2",
            description="Create chart from data",
            dependencies=["1"],
        )

        assert subtask.dependencies == ["1"]


class TestSubAgentState:
    """Tests for SubAgentState model."""

    def test_sub_agent_state_initialization(self):
        """SubAgentState should initialize correctly."""
        from agent.orchestrator.agent_models import SubAgentState

        state = SubAgentState(main_goal="Process multiple files")

        assert state.main_goal == "Process multiple files"
        assert state.subtasks == []
        assert state.completed_subtasks == {}
        assert state.status == "planning"

    def test_sub_agent_state_with_subtasks(self):
        """SubAgentState should track subtasks."""
        from agent.orchestrator.agent_models import SubAgentState, SubTask

        subtask1 = SubTask(id="1", description="Task 1")
        subtask2 = SubTask(id="2", description="Task 2")

        state = SubAgentState(
            main_goal="Do things",
            subtasks=[subtask1, subtask2],
        )

        assert len(state.subtasks) == 2


class TestAgentStateSubAgentFields:
    """Tests for sub-agent fields in AgentState."""

    def test_agent_state_has_sub_agent_fields(self):
        """AgentState should have sub-agent tracking fields."""
        from agent.orchestrator.agent_models import AgentState

        state = AgentState(goal="Test")

        assert hasattr(state, "is_sub_agent")
        assert hasattr(state, "parent_goal")
        assert hasattr(state, "sub_agent_results")
        assert state.is_sub_agent is False
        assert state.sub_agent_results == []

    def test_agent_state_as_sub_agent(self):
        """AgentState should track sub-agent status."""
        from agent.orchestrator.agent_models import AgentState

        state = AgentState(
            goal="Subtask",
            is_sub_agent=True,
            parent_goal="Main task",
        )

        assert state.is_sub_agent is True
        assert state.parent_goal == "Main task"


class TestTaskDecomposition:
    """Tests for task decomposition logic."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        """Create a ReActAgent instance for testing."""
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(
            sandbox=mock_sandbox,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_should_decompose_parallel_task(self, mock_llm, agent):
        """Agent should identify parallelizable tasks."""
        mock_llm.return_value = {
            "should_parallelize": True,
            "reasoning": "Two independent tasks",
            "subtasks": [
                {"id": "1", "description": "Organize files", "dependencies": []},
                {"id": "2", "description": "Summarize notes", "dependencies": []},
            ],
        }

        should_parallel, subtasks = await agent._should_decompose(
            "Organize my files and summarize my notes"
        )

        assert should_parallel is True
        assert len(subtasks) == 2
        assert subtasks[0].description == "Organize files"
        assert subtasks[1].description == "Summarize notes"

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_should_not_decompose_simple_task(self, mock_llm, agent):
        """Agent should not decompose simple tasks."""
        mock_llm.return_value = {
            "should_parallelize": False,
            "reasoning": "Single simple command",
            "subtasks": [],
        }

        should_parallel, subtasks = await agent._should_decompose("List files")

        assert should_parallel is False
        assert subtasks == []

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_should_not_decompose_dependent_tasks(self, mock_llm, agent):
        """Agent should not parallelize tasks with dependencies."""
        mock_llm.return_value = {
            "should_parallelize": False,
            "reasoning": "Tasks depend on each other",
            "subtasks": [],
        }

        should_parallel, subtasks = await agent._should_decompose(
            "Read the CSV and then create a chart"
        )

        assert should_parallel is False

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_decompose_filters_dependent_subtasks(self, mock_llm, agent):
        """Agent should filter out subtasks with dependencies."""
        mock_llm.return_value = {
            "should_parallelize": True,
            "reasoning": "Multiple tasks",
            "subtasks": [
                {"id": "1", "description": "Read data", "dependencies": []},
                {"id": "2", "description": "Process data", "dependencies": ["1"]},
                {"id": "3", "description": "Search web", "dependencies": []},
            ],
        }

        should_parallel, subtasks = await agent._should_decompose("Complex task")

        # Should only return independent subtasks
        assert should_parallel is True
        assert len(subtasks) == 2
        descriptions = [s.description for s in subtasks]
        assert "Read data" in descriptions
        assert "Search web" in descriptions
        assert "Process data" not in descriptions

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_decompose_handles_llm_error(self, mock_llm, agent):
        """Agent should handle LLM errors gracefully."""
        mock_llm.side_effect = Exception("LLM error")

        should_parallel, subtasks = await agent._should_decompose("Any task")

        assert should_parallel is False
        assert subtasks == []


class TestParallelExecution:
    """Tests for parallel sub-agent execution."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        """Create a ReActAgent instance for testing."""
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(
            sandbox=mock_sandbox,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_run_subtask_creates_sub_agent(self, mock_llm, agent):
        """Running a subtask should create a sub-agent."""
        from agent.orchestrator.agent_models import SubTask

        mock_llm.return_value = {
            "thought": "Simple task",
            "is_complete": True,
            "response": "Done organizing files",
        }

        subtask = SubTask(id="1", description="Organize files")
        result = await agent._run_subtask(subtask, {}, "Parent task goal")

        assert result["id"] == "1"
        assert result["description"] == "Organize files"
        assert result["status"] == "completed"
        assert result["result"] is not None

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_run_subtask_handles_max_iterations(self, mock_llm, agent):
        """Subtask should complete even when hitting edge cases."""
        from agent.orchestrator.agent_models import SubTask

        # Sub-agent runs commands but never says is_complete=True
        # Agent has repeat detection, so it may complete early
        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            return {
                "thought": f"Working iteration {call_count[0]}",
                "is_complete": False,
                "action": {"tool": "shell", "args": {"command": f"echo step{call_count[0]}"}},
            }

        mock_llm.side_effect = mock_response

        subtask = SubTask(id="1", description="Long running task")
        result = await agent._run_subtask(subtask, {}, "Parent task")

        # Should eventually stop (completed via repeat detection, or max_iterations)
        assert result["status"] in ("completed", "max_iterations")

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_run_parallel_subtasks(self, mock_llm, agent):
        """Multiple subtasks should run in parallel."""
        from agent.orchestrator.agent_models import SubTask

        mock_llm.return_value = {
            "thought": "Task done",
            "is_complete": True,
            "response": "Completed",
        }

        subtasks = [
            SubTask(id="1", description="Task 1"),
            SubTask(id="2", description="Task 2"),
            SubTask(id="3", description="Task 3"),
        ]

        results = await agent._run_parallel_subtasks(subtasks, {}, "Test parallel run")

        assert len(results) == 3
        assert all(r["status"] == "completed" for r in results)

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_parallel_execution_handles_mixed_results(self, mock_llm, agent):
        """Parallel execution should handle mixed success/max_iterations."""
        from agent.orchestrator.agent_models import SubTask

        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            # Task 2 never completes (hits max iterations)
            if call_count[0] % 3 == 2:
                return {
                    "thought": "Still working on task 2",
                    "is_complete": False,
                    "action": {"tool": "shell", "args": {"command": "echo working"}},
                }
            return {
                "thought": "Done",
                "is_complete": True,
                "response": f"Task complete",
            }

        mock_llm.side_effect = mock_response

        subtasks = [
            SubTask(id="1", description="Task 1"),
            SubTask(id="2", description="Task 2"),
            SubTask(id="3", description="Task 3"),
        ]

        results = await agent._run_parallel_subtasks(subtasks, {}, "Mixed results test")

        assert len(results) == 3
        statuses = [r["status"] for r in results]
        # At least one should complete, one might hit max_iterations
        assert "completed" in statuses or "max_iterations" in statuses


class TestResultMerging:
    """Tests for merging sub-agent results."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        """Create a ReActAgent instance for testing."""
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(
            sandbox=mock_sandbox,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_merge_successful_results(self, mock_llm, agent):
        """Merging should combine successful results."""
        mock_llm.return_value = {
            "success": True,
            "summary": "All tasks completed successfully. Files organized and notes summarized.",
        }

        results = [
            {"id": "1", "description": "Organize files", "status": "completed", "result": "Organized 10 files", "error": None},
            {"id": "2", "description": "Summarize notes", "status": "completed", "result": "Created summary", "error": None},
        ]

        merged = await agent._merge_subtask_results("Organize and summarize", results)

        assert "completed" in merged.lower() or "organized" in merged.lower() or "tasks" in merged.lower()

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_merge_handles_llm_error_with_fallback(self, mock_llm, agent):
        """Merging should fallback to simple concat on LLM error."""
        mock_llm.side_effect = Exception("LLM error")

        results = [
            {"id": "1", "description": "Task 1", "status": "completed", "result": "Result 1", "error": None},
            {"id": "2", "description": "Task 2", "status": "completed", "result": "Result 2", "error": None},
        ]

        merged = await agent._merge_subtask_results("Goal", results)

        # Should contain the individual results
        assert "Task 1" in merged or "Result 1" in merged
        assert "Task 2" in merged or "Result 2" in merged


class TestSubAgentIntegration:
    """Integration tests for sub-agent flow."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        """Create a ReActAgent instance for testing."""
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(
            sandbox=mock_sandbox,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_full_parallel_flow(self, mock_llm, agent):
        """Test complete parallel execution flow."""
        call_count = [0]

        def mock_response(*args, **kwargs):
            call_count[0] += 1
            prompt = args[0] if args else ""

            # First call: decomposition
            if call_count[0] == 1:
                return {
                    "should_parallelize": True,
                    "reasoning": "Independent tasks",
                    "subtasks": [
                        {"id": "1", "description": "Search Python", "dependencies": []},
                        {"id": "2", "description": "Search JavaScript", "dependencies": []},
                    ],
                }
            # Sub-agent calls: complete immediately
            elif call_count[0] <= 3:
                return {
                    "thought": "Search done",
                    "is_complete": True,
                    "response": f"Found results for search {call_count[0] - 1}",
                }
            # Merge call
            else:
                return {
                    "success": True,
                    "summary": "Found Python and JavaScript results",
                }

        mock_llm.side_effect = mock_response

        state = await agent.run("Search for Python and JavaScript tutorials")

        assert state.status in ("completed", "partial")
        assert len(state.sub_agent_results) == 2

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_sub_agent_does_not_decompose(self, mock_llm, agent):
        """Sub-agents should not try to decompose further."""
        agent._is_sub_agent = True

        mock_llm.return_value = {
            "thought": "Simple response",
            "is_complete": True,
            "response": "Done",
        }

        state = await agent.run("Do something")

        # Should complete without checking for decomposition
        assert state.status == "completed"
        assert state.is_sub_agent is True

    @pytest.mark.asyncio
    async def test_progress_callback_receives_parallel_status(self, mock_sandbox):
        """Progress callback should receive 'parallel' status."""
        from agent.orchestrator.react_agent import ReActAgent

        progress_calls = []

        def on_progress(iteration, status, thought, action):
            progress_calls.append((iteration, status, thought, action))

        agent = ReActAgent(
            sandbox=mock_sandbox,
            on_progress=on_progress,
            max_iterations=5,
        )

        with patch("agent.orchestrator.react_agent.call_llm_json_async") as mock_llm:
            call_count = [0]

            def mock_response(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {
                        "should_parallelize": True,
                        "reasoning": "Two tasks",
                        "subtasks": [
                            {"id": "1", "description": "Task A", "dependencies": []},
                            {"id": "2", "description": "Task B", "dependencies": []},
                        ],
                    }
                elif call_count[0] <= 3:
                    return {"thought": "Done", "is_complete": True, "response": "OK"}
                else:
                    return {"success": True, "summary": "All done"}

            mock_llm.side_effect = mock_response

            await agent.run("Do parallel things")

        # Should have received a 'parallel' status
        statuses = [call[1] for call in progress_calls]
        assert "parallel" in statuses
