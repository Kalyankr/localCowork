"""LocalCowork CLI - Pure agentic interface.

Two ways to use:
- `localcowork` - Terminal agent (interactive)
- `localcowork serve` - Web UI
"""

import warnings

import typer

from agent.cli.console import Icons, console
from agent.config import settings
from agent.logging import configure_logging
from agent.version import __version__

# Suppress Python deprecation warnings to keep CLI output clean
# These come from third-party libraries and clutter the user experience
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure structured logging with Rich handler for console-aware output
# This integrates properly with Rich's Live display and prevents
# log messages from corrupting the CLI spinner
configure_logging(rich_console=console, json_output=settings.json_logs)

app = typer.Typer(
    name="localcowork",
    help=f"{Icons.ROBOT} Your local AI agent",
    add_completion=False,
    pretty_exceptions_show_locals=False,
    no_args_is_help=False,
)


def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]LocalCowork[/bold cyan] v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", "-V", callback=version_callback, is_eager=True
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """🤖 LocalCowork - Your local AI agent.

    Just run `localcowork` and start typing.
    The agent handles questions AND tasks automatically.

    Examples:
        localcowork                   # Start agent
        localcowork serve             # Start web UI
    """
    # Skip if subcommand (like serve)
    if ctx.invoked_subcommand is not None:
        return

    if verbose:
        configure_logging(
            verbose=True, rich_console=console, json_output=settings.json_logs
        )

    from agent.cli.agent_loop import run_agent

    run_agent(model)


@app.command()
def serve(
    host: str = typer.Option(settings.server_host, "--host", "-h"),
    port: int = typer.Option(settings.server_port, "--port", "-p"),
):
    """Start the web UI."""
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    console.print(f"\n  {Icons.ROBOT} [bold]LocalCowork[/bold] → {url}\n")
    webbrowser.open(url)
    uvicorn.run(
        "agent.orchestrator.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="warning",
    )


def cli():
    app()
