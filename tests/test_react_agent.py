"""Tests for the ReAct agent."""

import pytest
from unittest.mock import patch


class TestReActAgent:
    """Tests for the ReActAgent class."""

    @pytest.fixture
    def agent(self, mock_sandbox):
        """Create a ReActAgent instance for testing."""
        from agent.orchestrator.react_agent import ReActAgent

        return ReActAgent(
            sandbox=mock_sandbox,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json")
    async def test_agent_handles_greeting(self, mock_llm, agent):
        """Agent should handle greetings without running tools."""
        mock_llm.return_value = {
            "thought": "User is greeting me",
            "is_complete": True,
            "response": "Hello! How can I help you?",
        }

        state = await agent.run("Hello!")

        assert state.status == "completed"
        assert state.final_answer is not None
        assert "Hello" in state.final_answer or "help" in state.final_answer.lower()

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json")
    async def test_agent_executes_shell_command(self, mock_llm, agent):
        """Agent should execute shell commands when requested."""
        # First call: agent decides to run a command
        # Second call: agent reports completion
        mock_llm.side_effect = [
            {
                "thought": "User wants to see files, I'll run ls",
                "is_complete": False,
                "action": {"tool": "shell", "args": {"command": "ls"}},
            },
            {
                "thought": "Command executed successfully",
                "is_complete": True,
                "response": "Found 3 files in the directory",
            },
        ]

        state = await agent.run("List files")

        assert state.status == "completed"
        assert len(state.steps) >= 1

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json")
    async def test_agent_respects_max_iterations(self, mock_llm, agent):
        """Agent should stop after max iterations."""
        # Always return incomplete to trigger max iterations
        mock_llm.return_value = {
            "thought": "Still working on it",
            "is_complete": False,
            "action": {"tool": "shell", "args": {"command": "echo test"}},
        }

        state = await agent.run("Keep working forever")

        assert state.status == "max_iterations"
        assert len(state.steps) <= agent.max_iterations

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json")
    async def test_agent_handles_unknown_tool(self, mock_llm, agent):
        """Agent should handle unknown tool gracefully."""
        mock_llm.side_effect = [
            {
                "thought": "I'll use a non-existent tool",
                "is_complete": False,
                "action": {"tool": "unknown_tool", "args": {}},
            },
            {
                "thought": "That tool doesn't exist, let me finish",
                "is_complete": True,
                "response": "I encountered an issue but recovered",
            },
        ]

        state = await agent.run("Do something impossible")

        # Should eventually complete (possibly with error handling)
        assert state.status in ["completed", "failed"]


class TestAgentState:
    """Tests for AgentState model."""

    def test_agent_state_initialization(self):
        """AgentState should initialize with correct defaults."""
        from agent.orchestrator.react_agent import AgentState

        state = AgentState(goal="Test goal")

        assert state.goal == "Test goal"
        assert state.status == "running"
        assert state.steps == []
        assert state.final_answer is None

    def test_agent_state_with_steps(self):
        """AgentState should track steps correctly."""
        from agent.orchestrator.react_agent import (
            AgentState,
            AgentStep,
            Observation,
            Thought,
        )

        state = AgentState(goal="Test")
        step = AgentStep(
            iteration=1,
            observation=Observation(source="initial", content="Ready"),
            thought=Thought(reasoning="Starting work", is_goal_complete=False),
        )
        state.steps.append(step)

        assert len(state.steps) == 1
        assert state.steps[0].iteration == 1


class TestObservation:
    """Tests for Observation model."""

    def test_observation_creation(self):
        """Observation should be created with required fields."""
        from agent.orchestrator.react_agent import Observation

        obs = Observation(source="tool", content={"result": "success"})

        assert obs.source == "tool"
        assert obs.content == {"result": "success"}
        assert obs.timestamp is not None


class TestThought:
    """Tests for Thought model."""

    def test_thought_defaults(self):
        """Thought should have correct defaults."""
        from agent.orchestrator.react_agent import Thought

        thought = Thought(reasoning="Thinking about the problem")

        assert thought.reasoning == "Thinking about the problem"
        assert thought.confidence == 1.0
        assert thought.is_goal_complete is False


class TestAction:
    """Tests for Action model."""

    def test_action_with_args(self):
        """Action should store tool and args."""
        from agent.orchestrator.react_agent import Action

        action = Action(
            tool="shell", args={"command": "ls -la"}, description="List files"
        )

        assert action.tool == "shell"
        assert action.args["command"] == "ls -la"
        assert action.description == "List files"
