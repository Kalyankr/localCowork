import typer
import asyncio
import logging
import json
import webbrowser
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich import box

from agent.orchestrator.planner import generate_plan
from agent.orchestrator.executor import Executor
from agent.orchestrator.deps import get_tool_registry, get_sandbox
from agent.llm.client import LLMError

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()

def friendly_error(error: str) -> tuple[str, str]:
    """Convert raw Python errors to user-friendly messages.
    
    Returns: (friendly_message, technical_detail)
    """
    error_lower = error.lower()
    
    # File/path errors
    if "filenotfounderror" in error_lower or "no such file" in error_lower:
        return "File not found", error.split(":")[-1].strip() if ":" in error else error
    if "permissionerror" in error_lower or "permission denied" in error_lower:
        return "Permission denied", "Check file permissions or run with elevated access"
    if "isadirectoryerror" in error_lower:
        return "Expected file, got directory", error.split(":")[-1].strip() if ":" in error else error
    
    # Network errors
    if "connectionerror" in error_lower or "connection refused" in error_lower:
        return "Connection failed", "Service may be offline or unreachable"
    if "timeouterror" in error_lower or "timed out" in error_lower:
        return "Request timed out", "Try again or check network connection"
    
    # Docker/sandbox errors
    if "docker" in error_lower:
        return "Docker error", "Ensure Docker is running and accessible"
    if "container" in error_lower:
        return "Sandbox error", "Failed to run code in isolated environment"
    
    # JSON/parsing errors
    if "jsondecodeerror" in error_lower or "json" in error_lower:
        return "Invalid data format", "Could not parse response"
    
    # Python runtime errors
    if "nameerror" in error_lower:
        return "Code error", "Variable or function not defined"
    if "typeerror" in error_lower:
        return "Type mismatch", "Wrong data type used in operation"
    if "valueerror" in error_lower:
        return "Invalid value", error.split(":")[-1].strip() if ":" in error else error
    if "keyerror" in error_lower:
        return "Missing key", error.split(":")[-1].strip() if ":" in error else error
    if "indexerror" in error_lower:
        return "Index out of range", "List or array access failed"
    if "attributeerror" in error_lower:
        return "Missing attribute", "Object doesn't have expected property"
    if "importerror" in error_lower or "modulenotfounderror" in error_lower:
        return "Missing dependency", error.split(":")[-1].strip() if ":" in error else error
    if "zerodivisionerror" in error_lower:
        return "Math error", "Division by zero"
    if "memoryerror" in error_lower:
        return "Out of memory", "Task requires too much memory"
    
    # Dependency errors
    if "dependency failed" in error_lower:
        return "Skipped", "Previous step failed"
    
    # LLM errors
    if "ollama" in error_lower or "llm" in error_lower:
        return "AI service error", "Check if Ollama is running"
    
    # Generic fallback - try to extract useful part
    if len(error) > 80:
        # Trim long errors, keep the meaningful part
        if ":" in error:
            parts = error.split(":")
            return "Error", parts[-1].strip()[:60]
        return "Error", error[:60] + "…"
    
    return "Error", error


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

# Use shared dependencies (lazy loaded via deps.py)
_tool_registry = None
_sandbox = None

def get_tools():
    global _tool_registry, _sandbox
    if _tool_registry is None:
        _tool_registry = get_tool_registry()
        _sandbox = get_sandbox()
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
    show_plan: bool = typer.Option(False, "--show-plan", "-p", help="Expand plan details"),
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
        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            try:
                plan = generate_plan(request)
            except LLMError as e:
                friendly_msg, detail = friendly_error(str(e))
                console.print(f"[bold red]✗ AI Error:[/bold red] {friendly_msg}")
                console.print(f"  [dim]{detail}[/dim]")
                if verbose:
                    console.print(f"  [dim]Technical: {e}[/dim]")
                raise typer.Exit(code=1)
            except Exception as e:
                friendly_msg, detail = friendly_error(str(e))
                console.print(f"[bold red]✗ Planning failed:[/bold red] {friendly_msg}")
                console.print(f"  [dim]{detail}[/dim]")
                logger.exception("Plan generation failed")
                raise typer.Exit(code=1)
    else:
        try:
            plan = generate_plan(request)
        except (LLMError, Exception) as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            raise typer.Exit(code=1)

    # Check if this is a chat (conversational) request
    is_chat = len(plan.steps) == 1 and plan.steps[0].action == "chat_op"

    # For chat, use simplified flow
    if is_chat and not output_json:
        executor = Executor(plan=plan, tool_registry=tool_registry, sandbox=sandbox, parallel=False)
        results = asyncio.run(executor.run())
        chat_result = list(results.values())[0]
        if chat_result.output:
            console.print(Panel(str(chat_result.output), border_style="cyan", box=box.ROUNDED))
        else:
            console.print("[dim]No response[/dim]")
        raise typer.Exit(code=0)

    # Display plan (collapsible accordion-style)
    if not quiet and not output_json:
        # Count actions by type
        action_counts = {}
        for step in plan.steps:
            action_counts[step.action] = action_counts.get(step.action, 0) + 1
        
        actions_summary = ", ".join(f"{v}× {k}" for k, v in action_counts.items())
        
        # Always show compact summary
        console.print(
            Panel(
                f"[bold]{len(plan.steps)} steps[/bold] — {actions_summary}",
                title="📋 Plan",
                title_align="left",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        
        # Expand details if --show-plan or user wants to see
        expand_plan = show_plan or dry_run
        if not expand_plan and not yes:
            expand_plan = typer.confirm("Show plan details?", default=False)
        
        if expand_plan:
            # Build tree view of the plan
            tree = Tree("[bold cyan]Steps[/bold cyan]", guide_style="dim")
            for i, step in enumerate(plan.steps, 1):
                deps = f" [dim](← {', '.join(step.depends_on)})[/dim]" if step.depends_on else ""
                desc = f" — {step.description[:40]}…" if step.description and len(step.description) > 40 else (f" — {step.description}" if step.description else "")
                branch = tree.add(f"[yellow]{step.action}[/yellow]{deps}")
                branch.add(f"[dim]{i}. {step.id}{desc}[/dim]")
            
            console.print(tree)
        
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
        
        # Compact result summary
        result_icon = "✓" if failed_count == 0 else "⚠"
        result_style = "green" if failed_count == 0 else "yellow"
        console.print(
            Panel(
                f"[{result_style}]{result_icon}[/{result_style}] [green]{success_count} succeeded[/green], [red]{failed_count} failed[/red]",
                title="[bold]Result[/bold]",
                title_align="left",
                border_style=result_style,
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

        # Show errors in expandable tree if any
        errors = [(step_id, result.error) for step_id, result in results.items() if result.error]
        if errors:
            # Show friendly error summary first
            console.print()
            for step_id, error in errors:
                friendly_msg, detail = friendly_error(error)
                console.print(f"  [red]✗[/red] [bold]{step_id}:[/bold] {friendly_msg} — [dim]{detail}[/dim]")
            
            # Offer to show raw errors for debugging
            if verbose or typer.confirm("Show technical details?", default=False):
                error_tree = Tree("[dim]Technical Details[/dim]", guide_style="dim")
                for step_id, error in errors:
                    error_tree.add(f"[dim]{step_id}: {error}[/dim]")
                console.print(error_tree)

        console.print()

    # Check if this was a chat response (single chat_op step)
    is_chat = len(plan.steps) == 1 and plan.steps[0].action == "chat_op"
    
    if is_chat:
        # For chat, just show the response directly - no summarizer
        chat_result = list(results.values())[0]
        if chat_result.output:
            console.print(Panel(str(chat_result.output), border_style="cyan", box=box.ROUNDED))
    elif not quiet:
        # Generate summary for task results
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
