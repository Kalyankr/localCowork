"""Unit tests for config module."""

import pytest
import os

from agent.config import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        settings = Settings()
        
        assert settings.ollama_url == "http://localhost:11434/api/generate"
        assert settings.ollama_model == "mistral"
        assert settings.ollama_timeout == 120
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
        """Should return same instance (cached)."""
        # Clear cache first
        get_settings.cache_clear()
        
        s1 = get_settings()
        s2 = get_settings()
        
        assert s1 is s2
