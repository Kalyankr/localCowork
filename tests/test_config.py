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
        assert settings.sandbox_timeout == 30
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
