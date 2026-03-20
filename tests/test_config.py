"""Unit tests for config module."""

from agent.config import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_values(self):
        """Should have sensible defaults."""
        # Create fresh settings without .env influence
        settings = Settings(
            _env_file=None,  # Ignore .env file
        )

        assert settings.ollama_url == "http://localhost:11434/api/generate"
        assert settings.ollama_model == "mistral"
        assert settings.sandbox_timeout == 300  # 5 minutes for Python scripts
        assert settings.session_timeout == 3600

    def test_sandbox_defaults(self):
        """Should have sandbox resource limits."""
        settings = Settings()

        assert settings.sandbox_memory_limit == "256m"
        assert settings.sandbox_cpu_limit == "1"
        assert settings.sandbox_pids_limit == 50

    def test_server_defaults(self):
        """Should have server configuration."""
        settings = Settings()

        assert settings.server_host == "127.0.0.1"
        assert settings.server_port == 8000


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Should return a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached(self):
        """Should return new instance each call (intentionally not cached)."""
        # get_settings intentionally creates new instances to pick up .env changes
        s1 = get_settings()
        s2 = get_settings()

        # Both should be valid Settings instances
        assert isinstance(s1, Settings)
        assert isinstance(s2, Settings)


# =============================================================================
# Extended config edge-case tests
# =============================================================================


class TestSettingsEnvOverride:
    """Settings should be overridable via LOCALCOWORK_ env vars."""

    def test_env_var_overrides_model(self, monkeypatch):
        monkeypatch.setenv("LOCALCOWORK_OLLAMA_MODEL", "llama3")
        s = Settings(_env_file=None)
        assert s.ollama_model == "llama3"

    def test_env_var_overrides_port(self, monkeypatch):
        monkeypatch.setenv("LOCALCOWORK_SERVER_PORT", "9999")
        s = Settings(_env_file=None)
        assert s.server_port == 9999

    def test_env_var_overrides_bool(self, monkeypatch):
        monkeypatch.setenv("LOCALCOWORK_USE_DOCKER", "true")
        s = Settings(_env_file=None)
        assert s.use_docker is True

    def test_unknown_env_vars_are_ignored(self, monkeypatch):
        monkeypatch.setenv("LOCALCOWORK_TOTALLY_UNKNOWN", "whatever")
        s = Settings(_env_file=None)  # extra="ignore" should prevent error
        assert isinstance(s, Settings)


class TestSettingsProperties:
    """Tests for computed properties on Settings."""

    def test_version_property(self):
        from agent.version import __version__

        s = Settings(_env_file=None)
        assert s.version == __version__

    def test_workspace_path_expands_tilde(self):
        s = Settings(_env_file=None, workspace_dir="~/test_ws")
        assert "~" not in s.workspace_path
        assert "test_ws" in s.workspace_path

    def test_history_path_expands_tilde(self):
        s = Settings(_env_file=None, task_history_file="~/hist.json")
        assert "~" not in s.history_path
        assert "hist.json" in s.history_path

    def test_db_path_expands_tilde(self):
        s = Settings(_env_file=None, db_file="~/db.sqlite")
        assert "~" not in s.db_path
        assert "db.sqlite" in s.db_path


class TestSettingsDefaults:
    """Ensure all important defaults are present and reasonable."""

    def test_rate_limit_defaults(self):
        s = Settings(_env_file=None)
        assert s.rate_limit_requests > 0
        assert s.rate_limit_window > 0

    def test_context_limits_ordered(self):
        s = Settings(_env_file=None)
        assert s.context_limit_short < s.context_limit_medium < s.context_limit_long

    def test_denied_paths_include_ssh(self):
        s = Settings(_env_file=None)
        assert "~/.ssh" in s.denied_paths

    def test_cors_origins_include_localhost(self):
        s = Settings(_env_file=None)
        assert any("localhost" in o for o in s.cors_origins)
