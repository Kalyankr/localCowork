"""LocalCowork CLI - Modular command-line interface.

This module provides a clean, professional CLI for LocalCowork with:
- Task execution (run, plan)
- Web server (serve)
- Task management (tasks, status, approve, reject)
"""

import logging
import typer

from agent.cli.app import app, APP_NAME, APP_VERSION, version_callback
from agent.cli.console import console
from agent.cli.commands import run, plan, serve, tasks, status, approve, reject

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


@app.callback()
def main_callback(
    version: bool = typer.Option(
        None, "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
):
    """LocalCowork - AI-powered local task automation."""
    pass


# Register commands
app.command()(run)
app.command()(plan)
app.command()(serve)
app.command()(tasks)
app.command()(status)
app.command()(approve)
app.command()(reject)


def main():
    """Entry point for the CLI."""
    app()


__all__ = ["app", "main", "console"]
