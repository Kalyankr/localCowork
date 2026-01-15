import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent.orchestrator.planner import generate_plan
from agent.orchestrator.executor import Executor
from agent.orchestrator.tool_registry import ToolRegistry
from agent.tools import file_tools, markdown_tools, data_tools, pdf_tools
from agent.sandbox.sandbox_runner import Sandbox

console = Console()
app = typer.Typer()

tool_registry = ToolRegistry()
tool_registry.register("file_op", file_tools.dispatch)
tool_registry.register("markdown_op", markdown_tools.dispatch)
tool_registry.register("data_op", data_tools.dispatch)
tool_registry.register("pdf_op", pdf_tools.dispatch)

sandbox = Sandbox()


def format_output_item(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_dir():
        return f"[bold blue]📁 {p.name}[/bold blue]"
    else:
        return f"[green]📄 {p.name}[/green]"


@app.command()
def run(request: str):
    """Run a natural-language task directly from the CLI."""

    console.print(Panel.fit(f"[bold cyan] Generating Plan[/bold cyan]\n{request}"))

    plan = generate_plan(request)

    # Pretty-print the plan JSON
    plan_json = plan.model_dump_json(indent=2)
    console.print(
        Panel(
            Syntax(plan_json, "json", theme="monokai", line_numbers=False),
            title="Plan",
            border_style="cyan",
        )
    )

    executor = Executor(plan=plan, tool_registry=tool_registry, sandbox=sandbox)

    console.print(Panel.fit("[bold green] Executing Steps[/bold green]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Running...", total=None)
        import asyncio

        results = results = asyncio.run(executor.run())
        progress.update(task, description="Done")

    # Display results in a table
    table = Table(title="Execution Results", show_lines=True)
    table.add_column("Step ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Output / Error", style="white")

    for step_id, result in results.items():
        for step_id, result in results.items():
            if isinstance(result.output, bool):
                if result.output:
                    console.print(
                        f"\n[bold green]There IS a sample file in your Downloads: {result.output}[/bold green]"
                    )
                else:
                    console.print(
                        "\n[bold red]There is NO sample file in your Downloads.[/bold red]"
                    )
                break
        output = result.output
        error = result.error

        if isinstance(output, list):
            text = "\n".join(format_output_item(item) for item in output)

        elif isinstance(output, bool):
            text = "[green]✔ Yes[/green]" if output else "[red]✘ No[/red]"

        elif output is not None:
            text = str(output)

        else:
            text = f"[red]{error}[/red]"

        table.add_row(step_id, result.status, text)

    console.print(table)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI server."""
    import uvicorn

    console.print(
        Panel.fit(f"[bold green]Starting API server[/bold green]\n{host}:{port}")
    )
    uvicorn.run("server:app", host=host, port=port, reload=True)


def main():
    app()
