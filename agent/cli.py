import typer
import asyncio
import logging
import json
import webbrowser
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich import box

from agent.orchestrator.planner import generate_plan
from agent.orchestrator.executor import Executor
from agent.tools import create_default_registry
from agent.sandbox.sandbox_runner import Sandbox
from agent.llm.client import LLMError

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()

def version_callback(value: bool):
    if value:
        console.print(f"[bold cyan]localcowork[/bold cyan] version {__version__}")
        raise typer.Exit()

app = typer.Typer(
    name="localcowork",
    help="🤖 Your Local AI Assistant — run natural language tasks from the CLI.",
    add_completion=False,
    no_args_is_help=True,
)

# Use shared registry (lazy loaded)
_tool_registry = None
_sandbox = None

def get_tools():
    global _tool_registry, _sandbox
    if _tool_registry is None:
        _tool_registry = create_default_registry()
        _sandbox = Sandbox()
    return _tool_registry, _sandbox


@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", "-V", callback=version_callback, is_eager=True, help="Show version and exit"),
):
    """LocalCowork - AI-powered local task automation."""
    pass


@app.command()
def run(
    request: str = typer.Argument(..., help="Natural language task description"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    no_parallel: bool = typer.Option(False, "--no-parallel", "-s", help="Run steps sequentially"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show plan without executing"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (just summary)"),
    output_json: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Run a natural-language task."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("agent").setLevel(logging.DEBUG)

    tool_registry, sandbox = get_tools()

    # Planning phase
    if not quiet and not output_json:
        console.print()
        console.print(f"[dim]Task:[/dim] {request}")
        console.print()
        with console.status("[bold cyan]Planning...[/bold cyan]", spinner="dots"):
            try:
                plan = generate_plan(request)
            except LLMError as e:
                console.print(f"[bold red]✗ LLM Error:[/bold red] {e}")
                raise typer.Exit(code=1)
            except Exception as e:
                console.print(f"[bold red]✗ Planning failed:[/bold red] {e}")
                logger.exception("Plan generation failed")
                raise typer.Exit(code=1)
    else:
        try:
            plan = generate_plan(request)
        except (LLMError, Exception) as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            raise typer.Exit(code=1)

    # Display plan
    if not quiet and not output_json:
        plan_table = Table(
            title="📋 Plan",
            box=box.ROUNDED,
            title_style="bold cyan",
            show_header=True,
            header_style="bold",
        )
        plan_table.add_column("#", style="dim", width=3)
        plan_table.add_column("Step", style="cyan", width=20)
        plan_table.add_column("Action", style="yellow", width=12)
        plan_table.add_column("Description", style="white")
        plan_table.add_column("Depends On", style="dim", width=15)

        for i, step in enumerate(plan.steps, 1):
            deps = ", ".join(step.depends_on) if step.depends_on else "—"
            desc = step.description or "—"
            plan_table.add_row(str(i), step.id, step.action, desc[:50], deps)

        console.print(plan_table)
        console.print()

    # Dry run exits here
    if dry_run:
        if output_json:
            print(json.dumps({"plan": plan.model_dump()}))
        else:
            console.print("[yellow]Dry run — no changes made.[/yellow]")
        raise typer.Exit(code=0)

    # Confirmation
    if not yes and not quiet and not output_json:
        if not typer.confirm("Execute this plan?", default=True):
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    # Execution phase
    step_status = {}
    total_steps = len(plan.steps)
    completed_count = 0

    def build_progress_table() -> Table:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Status", width=10)
        table.add_column("Step", style="cyan", width=20)
        table.add_column("Info", style="dim")

        for step in plan.steps:
            status = step_status.get(step.id, "pending")
            desc = (step.description or step.action)[:35]
            
            status_icons = {
                "pending": "[dim]○ pending[/dim]",
                "starting": "[yellow]● running[/yellow]",
                "success": "[green]✓ done[/green]",
                "error": "[red]✗ failed[/red]",
                "skipped": "[dim]◌ skipped[/dim]",
            }
            status_text = status_icons.get(status, f"[yellow]{status}[/yellow]")
            table.add_row(status_text, step.id, desc)

        return table

    def on_progress(step_id: str, status: str, current: int, total: int):
        nonlocal completed_count
        step_status[step_id] = status
        if status in ("success", "error", "skipped"):
            completed_count += 1

    parallel_mode = not no_parallel
    executor = Executor(
        plan=plan,
        tool_registry=tool_registry,
        sandbox=sandbox,
        on_progress=on_progress,
        parallel=parallel_mode,
    )

    if not quiet and not output_json:
        mode = "[green]parallel[/green]" if parallel_mode else "[yellow]sequential[/yellow]"
        console.print(f"[bold]Executing[/bold] ({mode})")
        console.print()

        with Live(build_progress_table(), console=console, refresh_per_second=4) as live:
            async def run_and_update():
                async def updater():
                    while completed_count < total_steps:
                        live.update(build_progress_table())
                        await asyncio.sleep(0.2)

                update_task = asyncio.create_task(updater())
                try:
                    return await executor.run()
                finally:
                    update_task.cancel()
                    try:
                        await update_task
                    except asyncio.CancelledError:
                        pass

            results = asyncio.run(run_and_update())
            live.update(build_progress_table())
    else:
        results = asyncio.run(executor.run())

    # Results summary
    success_count = sum(1 for r in results.values() if r.status == "success")
    failed_count = sum(1 for r in results.values() if r.status in ("error", "skipped"))

    if output_json:
        # Generate summary for JSON output too
        from agent.orchestrator.planner import summarize_results
        summary = summarize_results(request, results)
        output = {
            "plan": plan.model_dump(),
            "results": {k: v.model_dump() for k, v in results.items()},
            "summary": summary,
            "stats": {"success": success_count, "failed": failed_count},
        }
        print(json.dumps(output, indent=2))
        raise typer.Exit(code=0 if failed_count == 0 else 1)

    if not quiet:
        console.print()
        console.print(f"[bold]Result:[/bold] [green]{success_count} succeeded[/green], [red]{failed_count} failed[/red]")

        # Show errors
        for step_id, result in results.items():
            if result.error:
                console.print(f"  [red]✗ {step_id}:[/red] {result.error}")

        console.print()

    # Generate summary
    if not quiet:
        with console.status("[bold cyan]Summarizing...[/bold cyan]", spinner="dots"):
            from agent.orchestrator.planner import summarize_results
            summary = summarize_results(request, results)
        console.print(Panel(summary, title="[bold green]Summary[/bold green]", border_style="green", box=box.ROUNDED))
    else:
        from agent.orchestrator.planner import summarize_results
        summary = summarize_results(request, results)
        console.print(summary)

    raise typer.Exit(code=0 if failed_count == 0 else 1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
):
    """Start the web UI server."""
    import uvicorn

    url = f"http://{host}:{port}"
    console.print()
    console.print(Panel.fit(
        f"[bold green]LocalCowork Server[/bold green]\n\n"
        f"[dim]URL:[/dim] {url}\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        border_style="green",
        box=box.ROUNDED,
    ))

    if open_browser:
        webbrowser.open(url)

    uvicorn.run("agent.orchestrator.server:app", host=host, port=port, reload=True, log_level="warning")


@app.command()
def plan(
    request: str = typer.Argument(..., help="Natural language task description"),
):
    """Generate and display a plan without executing it."""
    console.print()
    with console.status("[bold cyan]Planning...[/bold cyan]", spinner="dots"):
        try:
            plan = generate_plan(request)
        except LLMError as e:
            console.print(f"[bold red]✗ LLM Error:[/bold red] {e}")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[bold red]✗ Planning failed:[/bold red] {e}")
            raise typer.Exit(code=1)

    print(plan.model_dump_json(indent=2))


def main():
    app()
