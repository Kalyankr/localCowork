"""LocalCowork CLI - Pure agentic interface.

Three ways to use:
- `localcowork` - Terminal agent (interactive)
- `localcowork serve` - Web UI
- `localcowork doctor` - Diagnose setup
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
        localcowork doctor            # Check setup
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


@app.command()
def doctor():
    """Diagnose your LocalCowork setup.

    Checks Ollama, model, Docker, database, disk space, and Python version.
    """
    from agent.cli.doctor import run_doctor

    run_doctor()


@app.command()
def optimize(
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Optimizer LLM model"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show progress"),
):
    """Optimize agent prompts using Microsoft Trace.

    Runs evaluation tasks, scores results, and uses Trace's OptoPrime
    optimizer to improve the agent's prompts and heuristics.

    Requires: pip install localcowork[trace]
    Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY env var for the optimizer LLM.
    """
    import asyncio

    try:
        from agent.trace.trainer import run_training
    except ImportError:
        console.print(
            "[red]trace-opt not installed.[/red] Run: pip install localcowork[trace]"
        )
        raise typer.Exit(1)

    console.print(f"\n  {Icons.ROBOT} [bold]Trace Optimization[/bold]")
    console.print(f"  Epochs: {epochs} | Optimizer model: {model}\n")

    result = asyncio.run(
        run_training(epochs=epochs, optimizer_model=model, verbose=verbose)
    )

    console.print(f"\n  [green]Best score: {result['best_score']}[/green]")
    console.print("  Optimized parameters saved to ~/.localcowork/trace_params/\n")


@app.command()
def optimize_status():
    """Show current Trace optimization status and saved parameters."""
    from agent.trace.trainer import load_params

    params = load_params()
    if params:
        console.print(f"\n  {Icons.ROBOT} [bold]Optimized Parameters[/bold]\n")
        for key, value in params.items():
            preview = value[:80] + "..." if len(value) > 80 else value
            console.print(f"  [cyan]{key}[/cyan]: {preview}")
        console.print()
    else:
        console.print("\n  No optimized parameters found. Run: localcowork optimize\n")


@app.command()
def optimize_reset():
    """Clear saved Trace-optimized parameters (revert to defaults)."""
    from agent.trace.trainer import clear_params

    clear_params()
    console.print("\n  Optimized parameters cleared. Using defaults.\n")


def cli():
    app()
