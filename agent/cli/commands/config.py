"""Config commands - model management and settings."""

import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from agent.cli.console import console, Icons, print_error, print_success, print_info


def models(
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all model details"),
):
    """List available Ollama models."""
    from agent.llm.client import check_ollama_health, list_models
    from agent.config import settings
    
    # Health check
    with console.status("[cyan]Connecting to Ollama...[/cyan]", spinner="dots"):
        healthy, error = check_ollama_health()
    
    if not healthy:
        print_error("Cannot connect to Ollama", error)
        raise typer.Exit(code=1)
    
    available = list_models()
    
    if not available:
        print_info("No models found. Install one with: [cyan]ollama pull llama3[/cyan]")
        raise typer.Exit(code=0)
    
    current = settings.ollama_model
    
    console.print()
    table = Table(title=f"{Icons.ROBOT} Available Models", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("Status", width=10)
    
    for model in sorted(available):
        status = f"[green]{Icons.SUCCESS} active[/green]" if model == current else ""
        table.add_row(model, status)
    
    console.print(table)
    console.print()
    console.print(f"[dim]Default model:[/dim] {current}")
    console.print(f"[dim]Change with:[/dim] localcowork ask -m <model> ...")


def config(
    show: bool = typer.Option(True, "--show/--no-show", help="Show current config"),
):
    """Show current configuration."""
    from agent.config import settings
    
    console.print()
    console.print(Panel(
        f"[bold]Current Configuration[/bold]\n\n"
        f"[dim]Ollama URL:[/dim]     {settings.ollama_url}\n"
        f"[dim]Model:[/dim]          {settings.ollama_model}\n"
        f"[dim]Timeout:[/dim]        {settings.ollama_timeout}s\n"
        f"[dim]Max Tokens:[/dim]     {settings.max_tokens}\n"
        f"[dim]Sandbox:[/dim]        {'enabled' if settings.sandbox_enabled else 'disabled'}\n"
        f"[dim]Workspace:[/dim]      {settings.workspace_path}",
        title=f"{Icons.FOLDER} Config",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print()
    console.print(f"[dim]Config file: ~/.localcowork/config.yaml[/dim]")


def doctor():
    """Check system health and dependencies."""
    from agent.llm.client import check_ollama_health, list_models, check_model_exists
    from agent.config import settings
    import shutil
    
    console.print()
    console.print(Panel.fit(f"[bold]{Icons.ROBOT} LocalCowork Health Check[/bold]", border_style="cyan"))
    console.print()
    
    checks = []
    
    # Check Ollama connection
    console.print("[dim]Checking Ollama connection...[/dim]")
    healthy, error = check_ollama_health()
    if healthy:
        checks.append(("Ollama connection", True, "Connected"))
    else:
        checks.append(("Ollama connection", False, error or "Not running"))
    
    # Check model availability
    if healthy:
        console.print("[dim]Checking model availability...[/dim]")
        model = settings.ollama_model
        exists = check_model_exists(model)
        if exists:
            checks.append((f"Model ({model})", True, "Available"))
        else:
            available = list_models()
            if available:
                checks.append((f"Model ({model})", False, f"Not found. Try: {available[0]}"))
            else:
                checks.append((f"Model ({model})", False, "No models installed"))
    
    # Check Docker
    console.print("[dim]Checking Docker...[/dim]")
    docker_path = shutil.which("docker")
    if docker_path:
        checks.append(("Docker", True, "Available"))
    else:
        checks.append(("Docker", False, "Not found (sandbox disabled)"))
    
    # Check workspace
    console.print("[dim]Checking workspace...[/dim]")
    import os
    workspace = settings.workspace_path
    if os.path.isdir(workspace):
        checks.append(("Workspace", True, workspace))
    else:
        checks.append(("Workspace", False, f"Not found: {workspace}"))
    
    # Display results
    console.print()
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Check", style="cyan", width=25)
    table.add_column("Status", width=10)
    table.add_column("Details", style="dim", max_width=40)
    
    all_passed = True
    for name, passed, detail in checks:
        status = f"[green]{Icons.SUCCESS}[/green]" if passed else f"[red]{Icons.ERROR}[/red]"
        if not passed:
            all_passed = False
        table.add_row(name, status, detail)
    
    console.print(table)
    console.print()
    
    if all_passed:
        print_success("All checks passed! LocalCowork is ready to use.")
    else:
        console.print("[yellow]Some checks failed. See details above.[/yellow]")
    
    raise typer.Exit(code=0 if all_passed else 1)
