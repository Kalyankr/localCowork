"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import shutil


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
    def mock_call_llm(prompt):
        return "Mocked LLM response"
    
    def mock_call_llm_json(prompt):
        return {"steps": [{"id": "test", "action": "chat_op", "args": {"op": "respond", "message": "test"}}]}
    
    monkeypatch.setattr("agent.llm.client.call_llm", mock_call_llm)
    monkeypatch.setattr("agent.llm.client.call_llm_json", mock_call_llm_json)
