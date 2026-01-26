"""LocalCowork CLI - Claude CLI-style interface.

Usage:
    localcowork                    # Interactive chat mode
    localcowork "question"         # Quick question
    localcowork run "task"         # Execute a task with tools
    localcowork init               # Initialize workspace
"""

import sys
import logging
import typer

from agent.cli.app import app, APP_NAME, APP_VERSION, version_callback
from agent.cli.console import console
from agent.cli.commands import run, plan, serve, tasks, status, approve, reject, ask, models, config, doctor, init
from agent.cli.commands.ask import _check_connection, _single_question, _interactive_mode

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    prompt: str = typer.Argument(None, help="Question to ask (starts interactive mode if omitted)"),
    version: bool = typer.Option(
        None, "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model to use"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Force interactive mode"),
):
    """🤖 LocalCowork - Your local AI assistant.
    
    Just type a question to chat, or use commands for more control.
    
    Examples:
        localcowork "What is Python?"      # Quick question
        localcowork                         # Interactive chat
        localcowork run "summarize *.pdf"   # Execute task with tools
        localcowork init                    # Set up workspace
    """
    # If a subcommand is invoked, skip default behavior
    if ctx.invoked_subcommand is not None:
        return
    
    from agent.config import settings
    active_model = model or settings.ollama_model
    
    # Check connection first
    if not _check_connection():
        raise typer.Exit(code=1)
    
    # Direct question or interactive mode
    if prompt and not interactive:
        _single_question(prompt, active_model, stream=True)
    else:
        _interactive_mode(active_model, stream=True)


# Register commands
app.command()(run)
app.command()(plan)
app.command()(serve)
app.command()(tasks)
app.command()(status)
app.command()(approve)
app.command()(reject)
app.command(name="chat")(ask)  # Alias for explicit chat
app.command()(models)
app.command()(config)
app.command()(doctor)
app.command()(init)


def main():
    """Entry point for the CLI."""
    app()


__all__ = ["app", "main", "console"]
