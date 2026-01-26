"""Main Typer application and shared utilities."""

import typer
from typing import Optional

from agent.cli.console import console, print_header, Icons

# App metadata
APP_NAME = "LocalCowork"
APP_VERSION = "0.2.0"
APP_DESCRIPTION = "Privacy-first AI assistant for local tasks"

# Main Typer app
app = typer.Typer(
    name="localcowork",
    help=f"{Icons.ROBOT} {APP_DESCRIPTION}",
    add_completion=False,
    pretty_exceptions_show_locals=False,
    no_args_is_help=False,  # Allow no args to start interactive mode
)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        print_header(f"{APP_NAME} v{APP_VERSION}", APP_DESCRIPTION)
        raise typer.Exit()


def show_banner():
    """Show the app banner."""
    console.print()
    console.print(f"[bold cyan]{Icons.ROBOT} {APP_NAME}[/bold cyan] [dim]v{APP_VERSION}[/dim]")
    console.print(f"[dim]{APP_DESCRIPTION}[/dim]")
    console.print()


# Common options
ModelOption = typer.Option(
    None,
    "--model", "-m",
    help="Ollama model to use (defaults to config)"
)

VerboseOption = typer.Option(
    False,
    "--verbose", "-v",
    help="Show detailed output"
)

DryRunOption = typer.Option(
    False,
    "--dry-run", "-n",
    help="Preview without executing"
)

AgenticOption = typer.Option(
    False,
    "--agentic", "-a",
    help="Use autonomous ReAct agent mode"
)
