"""Centralized configuration for LocalCowork using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables with the
    LOCALCOWORK_ prefix. For example:
        LOCALCOWORK_OLLAMA_MODEL=llama2
        LOCALCOWORK_SANDBOX_TIMEOUT=60
    """
    
    model_config = SettingsConfigDict(
        env_prefix="LOCALCOWORK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # LLM Settings
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "mistral"
    ollama_timeout: int = 120
    max_json_retries: int = 2
    max_tokens: int = 2048
    
    # Sandbox Settings
    sandbox_timeout: int = 30
    sandbox_memory_limit: str = "256m"
    sandbox_cpu_limit: str = "1"
    sandbox_pids_limit: int = 50
    docker_image: str = "python:3.12-slim"
    
    # Server Settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    session_timeout: int = 3600  # 1 hour
    max_history_messages: int = 20
    
    # Execution Settings
    max_code_retries: int = 2
    parallel_execution: bool = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience alias
settings = get_settings()
