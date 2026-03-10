"""Structured logging configuration for LocalCowork.

Uses structlog integrated with Python's stdlib logging to provide:
- Structured key-value logging (e.g., logger.info("event", task_id=id, tool="shell"))
- Correlation IDs via contextvars (task_id automatically in all log entries)
- JSON output mode for server/production use
- Human-readable console output for CLI (compatible with Rich)
"""

import logging
import sys

import structlog


def configure_logging(
    verbose: bool = False,
    json_output: bool = False,
    rich_console: object | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        verbose: Enable DEBUG level logging.
        json_output: Use JSON renderer (for server/production).
        rich_console: Rich Console instance for RichHandler (CLI mode).
    """
    # Choose renderer based on output mode
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # ProcessorFormatter bridges structlog → stdlib LogRecord
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Set up the handler
    if rich_console is not None and not json_output:
        from rich.logging import RichHandler

        handler = RichHandler(
            console=rich_console,
            show_time=False,
            show_path=False,
            markup=True,
        )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(formatter)

    # Configure root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)

    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "ollama", "uvicorn", "asyncio"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def bind_task_context(**kwargs: object) -> None:
    """Bind key-value pairs to the current async context.

    These will be automatically included in all subsequent log entries
    within the same asyncio task / thread.

    Example:
        bind_task_context(task_id="abc-123", session_id="sess-1")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_task_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def unbind_task_context(*keys: str) -> None:
    """Remove specific keys from the bound context."""
    structlog.contextvars.unbind_contextvars(*keys)
