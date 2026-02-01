"""Centralized configuration for LocalCowork using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict

# Import version from single source of truth
from agent.version import __version__


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

    # Version (read-only, from version.py)
    @property
    def version(self) -> str:
        return __version__

    # LLM Settings
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "mistral"
    ollama_timeout: int = 120
    max_json_retries: int = 2
    max_tokens: int = 2048
    num_ctx: int = 8192  # Context window size (increase for longer prompts)

    # Sandbox Settings
    sandbox_timeout: int = 300  # 5 minutes for Python scripts
    sandbox_memory_limit: str = "256m"
    sandbox_cpu_limit: str = "1"
    sandbox_pids_limit: int = 50
    docker_image: str = "python:3.12-slim"
    sandbox_user_id: str = "1000:1000"  # UID:GID for sandbox container

    # Server Settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    session_timeout: int = 3600  # 1 hour
    max_history_messages: int = 20
    api_key: str | None = None  # Optional API key for authentication

    # CORS Settings
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    # Rate Limiting
    rate_limit_requests: int = 60
    rate_limit_window: int = 60  # seconds

    # Context Limits (for LLM prompts)
    context_limit_short: int = 2000
    context_limit_medium: int = 3000
    context_limit_long: int = 5000
    output_limit: int = 50000
    shell_timeout: int = 600  # 10 minutes for shell commands

    # Execution Settings
    max_code_retries: int = 2
    max_agent_iterations: int = 15  # Max ReAct loop iterations
    parallel_execution: bool = True

    # Task Management Settings
    workspace_dir: str = "~/.localcowork/workspaces"
    task_history_file: str = "~/.localcowork/task_history.json"
    require_approval: bool = (
        True  # If True, plans require user approval before execution
    )
    max_task_history: int = 100  # Max tasks to keep in history
    workspace_cleanup_days: int = 7  # Days before cleaning up old workspaces
    web_search_limit: int = 5  # Max search results to fetch

    @property
    def workspace_path(self) -> str:
        """Expand workspace directory path."""
        from pathlib import Path

        return str(Path(self.workspace_dir).expanduser())

    @property
    def history_path(self) -> str:
        """Expand task history file path."""
        from pathlib import Path

        return str(Path(self.task_history_file).expanduser())


def get_settings() -> Settings:
    """Get settings instance.

    Creates a new instance each time to pick up .env changes.
    For performance-critical code, cache the result yourself.
    """
    return Settings()


# Default settings instance (loaded at import time)
# If you change .env, restart the app or call get_settings() again
settings = get_settings()
