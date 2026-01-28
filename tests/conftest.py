"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    workspace = tempfile.mkdtemp(prefix="localcowork_test_")
    yield Path(workspace)
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def sample_files(tmp_path):
    """Create sample files for testing."""
    # Create various file types
    (tmp_path / "document.txt").write_text("Hello, World!")
    (tmp_path / "data.json").write_text('{"key": "value"}')
    (tmp_path / "script.py").write_text("print('hello')")
    (tmp_path / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n')
    
    # Create subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("nested content")
    
    return tmp_path


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock LLM responses for testing without Ollama."""
    def mock_call_llm(prompt, force_json=False):
        return "Mocked LLM response"
    
    def mock_call_llm_json(prompt):
        return {
            "thought": "This is a test",
            "is_complete": True,
            "response": "Test response"
        }
    
    monkeypatch.setattr("agent.llm.client.call_llm", mock_call_llm)
    monkeypatch.setattr("agent.llm.client.call_llm_json", mock_call_llm_json)


@pytest.fixture
def mock_sandbox():
    """Mock sandbox for testing without Docker."""
    sandbox = MagicMock()
    sandbox.run_python = AsyncMock(return_value={"output": "test output"})
    return sandbox


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "List files in my home directory"},
        {"role": "assistant", "content": "Here are the files..."},
    ]


@pytest.fixture(scope="session")
def test_settings():
    """Create test settings with safe defaults."""
    from agent.config import Settings
    
    return Settings(
        ollama_url="http://localhost:11434/api/generate",
        ollama_model="test-model",
        sandbox_timeout=5,
        require_approval=False,
        workspace_dir="/tmp/localcowork_test",
    )


# Async test support
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
