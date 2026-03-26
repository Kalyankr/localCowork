"""Tests for the localcowork doctor command."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

from agent.cli.doctor import (
    _check_database,
    _check_disk_space,
    _check_docker,
    _check_model,
    _check_ollama,
    _check_python,
    _check_workspace,
    run_doctor,
)


class TestCheckPython:
    def test_passes_on_312(self):
        ok, detail = _check_python()
        # We're running on 3.12+, so this should pass
        assert ok is True
        assert "3.12" in detail or "3.13" in detail or "3.14" in detail


class TestCheckOllama:
    def test_healthy(self):
        with patch("agent.llm.client.check_ollama_health", return_value=(True, None)):
            ok, detail = _check_ollama()
            assert ok is True
            assert detail == "Connected"

    def test_unhealthy(self):
        with patch(
            "agent.llm.client.check_ollama_health",
            return_value=(False, "Connection refused"),
        ):
            ok, detail = _check_ollama()
            assert ok is False
            assert "Connection refused" in detail

    def test_strips_prefix(self):
        with patch(
            "agent.llm.client.check_ollama_health",
            return_value=(False, "Unknown error: something bad"),
        ):
            ok, detail = _check_ollama()
            assert ok is False
            assert not detail.startswith("Unknown error:")
            assert "something bad" in detail

    def test_exception(self):
        with patch(
            "agent.llm.client.check_ollama_health", side_effect=RuntimeError("boom")
        ):
            ok, detail = _check_ollama()
            assert ok is False
            assert "boom" in detail


class TestCheckModel:
    def test_model_exists(self):
        with patch("agent.llm.client.check_model_exists", return_value=True):
            ok, detail = _check_model()
            assert ok is True

    def test_model_missing_shows_available(self):
        with (
            patch("agent.llm.client.check_model_exists", return_value=False),
            patch("agent.llm.client.list_models", return_value=["llama3", "codellama"]),
        ):
            ok, detail = _check_model()
            assert ok is False
            assert "not found" in detail
            assert "llama3" in detail

    def test_model_missing_no_models(self):
        with (
            patch("agent.llm.client.check_model_exists", return_value=False),
            patch("agent.llm.client.list_models", return_value=[]),
        ):
            ok, detail = _check_model()
            assert ok is False
            assert "no models pulled" in detail


class TestCheckDocker:
    def test_docker_not_required(self):
        with patch("agent.cli.doctor.settings") as mock_settings:
            mock_settings.use_docker = False
            ok, detail = _check_docker()
            assert ok is None
            assert "Not required" in detail

    def test_docker_not_in_path(self):
        with (
            patch("agent.cli.doctor.settings") as mock_settings,
            patch("agent.cli.doctor.shutil.which", return_value=None),
        ):
            mock_settings.use_docker = True
            ok, detail = _check_docker()
            assert ok is False
            assert "Not found" in detail


class TestCheckDatabase:
    def test_writable_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        # Create the DB with expected tables
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, key TEXT)"
        )
        conn.commit()
        conn.close()

        with patch("agent.cli.doctor.settings") as mock_settings:
            mock_settings.db_path = db_path
            ok, detail = _check_database()
            assert ok is True
            assert "0 tasks" in detail

    def test_missing_db_creates_it(self, tmp_path):
        db_path = str(tmp_path / "new_dir" / "new.db")
        with patch("agent.cli.doctor.settings") as mock_settings:
            mock_settings.db_path = db_path
            ok, detail = _check_database()
            assert ok is True


class TestCheckDiskSpace:
    def test_enough_space(self):
        ok, detail = _check_disk_space()
        # Assuming CI/dev has > 1 GB free
        assert ok is True
        assert "GB" in detail

    def test_reports_percentage(self):
        ok, detail = _check_disk_space()
        assert "%" in detail


class TestCheckWorkspace:
    def test_writable_workspace(self, tmp_path):
        ws = str(tmp_path / "ws")
        with patch("agent.cli.doctor.settings") as mock_settings:
            mock_settings.workspace_path = ws
            ok, detail = _check_workspace()
            assert ok is True
            assert ws in detail

    def test_creates_missing_dir(self, tmp_path):
        ws = str(tmp_path / "a" / "b" / "c")
        with patch("agent.cli.doctor.settings") as mock_settings:
            mock_settings.workspace_path = ws
            ok, detail = _check_workspace()
            assert ok is True
            assert Path(ws).is_dir()


class TestRunDoctor:
    def test_returns_zero_on_all_pass(self, tmp_path):
        db_path = str(tmp_path / "doc.db")
        ws = str(tmp_path / "ws")
        with (
            patch("agent.llm.client.check_ollama_health", return_value=(True, None)),
            patch("agent.llm.client.check_model_exists", return_value=True),
            patch("agent.cli.doctor.settings") as mock_settings,
        ):
            mock_settings.ollama_model = "mistral"
            mock_settings.ollama_url = "http://localhost:11434/api/generate"
            mock_settings.safety_profile = "strict"
            mock_settings.max_agent_iterations = 15
            mock_settings.use_docker = False
            mock_settings.db_path = db_path
            mock_settings.workspace_path = ws
            result = run_doctor()
            assert result == 0

    def test_returns_one_on_failure(self, tmp_path):
        db_path = str(tmp_path / "doc.db")
        ws = str(tmp_path / "ws")
        with (
            patch(
                "agent.llm.client.check_ollama_health",
                return_value=(False, "down"),
            ),
            patch("agent.llm.client.check_model_exists", return_value=False),
            patch("agent.llm.client.list_models", return_value=[]),
            patch("agent.cli.doctor.settings") as mock_settings,
        ):
            mock_settings.ollama_model = "mistral"
            mock_settings.ollama_url = "http://localhost:11434/api/generate"
            mock_settings.safety_profile = "strict"
            mock_settings.max_agent_iterations = 15
            mock_settings.use_docker = False
            mock_settings.db_path = db_path
            mock_settings.workspace_path = ws
            result = run_doctor()
            assert result == 1
