import typer
import asyncio
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.status import Status

from agent.orchestrator.planner import generate_plan
from agent.orchestrator.executor import Executor
from agent.tools import create_default_registry
from agent.sandbox.sandbox_runner import Sandbox
from agent.llm.client import LLMError

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during normal operation
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(
    name="localcowork",
    help="Your Local AI Assistant - run natural language tasks from the CLI.",
    add_completion=False,
)

# Use shared registry
tool_registry = create_default_registry()
sandbox = Sandbox()


def format_output_item(item) -> str:
    if isinstance(item, dict):
        path = item.get("path", "")
        name = item.get("name", "")
        is_dir = item.get("is_dir", False)
        if is_dir:
            return f"[bold blue]📁 {name}[/bold blue]"
        else:
            return f"[green]📄 {name}[/green]"
            
    p = Path(str(item)).expanduser()
    if p.is_dir():
        return f"[bold blue]📁 {p.name}[/bold blue]"
    else:
        return f"[green]📄 {p.name}[/green]"


@app.command()
def run(
    request: str,
    no_parallel: bool = typer.Option(False, "--no-parallel", "-s", help="Run steps sequentially instead of in parallel"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run a natural-language task directly from the CLI."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("agent").setLevel(logging.DEBUG)

    console.print(Panel.fit(f"[bold cyan]🤖 Generating Plan[/bold cyan]\n{request}"))

    try:
        plan = generate_plan(request)
    except LLMError as e:
        console.print(f"[bold red]❌ LLM Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to generate plan:[/bold red] {e}")
        logger.exception("Plan generation failed")
        raise typer.Exit(code=1)

    # Pretty-print the plan JSON
    plan_json = plan.model_dump_json(indent=2)
    console.print(
        Panel(
            Syntax(plan_json, "json", theme="monokai", line_numbers=False),
            title="📋 Plan",
            border_style="cyan",
        )
    )

    # Progress tracking state
    step_status = {}
    total_steps = len(plan.steps)
    
    def on_progress(step_id: str, status: str, current: int, total: int):
        """Callback for step progress updates."""
        step_status[step_id] = status
    
    # Create executor with parallel mode
    parallel_mode = not no_parallel
    executor = Executor(
        plan=plan, 
        tool_registry=tool_registry, 
        sandbox=sandbox,
        on_progress=on_progress,
        parallel=parallel_mode,
    )

    mode_text = "parallel" if parallel_mode else "sequential"
    console.print(Panel.fit(f"[bold green]⚡ Executing Steps[/bold green] ({mode_text} mode)"))

    # Execute with live progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]Running {total_steps} steps...", total=total_steps)
        
        async def run_with_progress():
            results = await executor.run()
            return results
        
        # Run executor
        results = asyncio.run(run_with_progress())
        
        # Update final progress
        completed = sum(1 for r in results.values() if r.status == "success")
        progress.update(task, completed=total_steps, description=f"[green]✓ Completed {completed}/{total_steps} steps")

    # Display results in a table
    table = Table(title="Execution Results", show_lines=True)
    table.add_column("Step ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Output / Error", style="white")

    for step_id, result in results.items():
        output = result.output
        error = result.error

        if isinstance(output, list):
            text = "\n".join(format_output_item(item) for item in output)
        elif isinstance(output, bool):
            text = "[green]✔ Yes[/green]" if output else "[red]✘ No[/red]"
        elif output:
            text = str(output)
        else:
            text = ""

        if error:
            if text:
                text += f"\n\n[red]Error: {error}[/red]"
            else:
                text = f"[red]{error}[/red]"

        table.add_row(step_id, result.status, text)

    console.print(table)

    # FINAL SUMMARY
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Generating final summary...[/bold cyan]"),
        transient=True,
    ) as progress:
        progress.add_task("Summarizing...", total=None)
        from agent.orchestrator.planner import summarize_results
        summary = summarize_results(request, results)

    console.print(Panel(summary, title="Summary", border_style="green"))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI server."""
    import uvicorn

    console.print(
        Panel.fit(f"[bold green]Starting API server[/bold green]\n{host}:{port}")
    )
    uvicorn.run("agent.orchestrator.server:app", host=host, port=port, reload=True)


def main():
    app()
