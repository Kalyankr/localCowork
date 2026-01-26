"""Serve command - web UI server."""

import typer
import webbrowser
from rich.panel import Panel
from rich import box

from agent.cli.console import console, Icons


def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Enable auto-reload on changes"),
):
    """Start the web UI server."""
    import uvicorn

    url = f"http://{host}:{port}"
    
    console.print()
    console.print(Panel.fit(
        f"[bold green]{Icons.ROCKET} LocalCowork Server[/bold green]\n\n"
        f"[dim]URL:[/dim] {url}\n"
        f"[dim]Reload:[/dim] {'enabled' if reload else 'disabled'}\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        border_style="green",
        box=box.ROUNDED,
    ))

    if open_browser:
        webbrowser.open(url)

    log_level = "warning"
    uvicorn.run(
        "agent.orchestrator.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
