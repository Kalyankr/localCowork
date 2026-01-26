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
from agent.orchestrator.react_agent import ReActAgent
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
    agentic: bool = typer.Option(False, "--agentic", "-a", help="Use ReAct agent (step-by-step reasoning)"),
):
    """Run a natural-language task."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("agent").setLevel(logging.DEBUG)

    tool_registry, sandbox = get_tools()

    # Agentic mode uses ReAct loop instead of one-shot planning
    if agentic:
        _run_agentic(request, tool_registry, sandbox, verbose, quiet, output_json)
        return

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


def _run_agentic(request: str, tool_registry, sandbox, verbose: bool, quiet: bool, output_json: bool):
    """Run task using ReAct agent (step-by-step reasoning)."""
    
    if not quiet and not output_json:
        console.print()
        console.print(f"[dim]Goal:[/dim] {request}")
        console.print()
        console.print("[bold cyan]🤖 Running in agentic mode (ReAct loop)[/bold cyan]")
        console.print("[dim]The agent will reason step-by-step, adapting as it goes...[/dim]")
        console.print()
    
    # Progress table for agentic mode
    agent_steps = []
    
    def build_agent_table() -> Table:
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("#", width=3, style="dim")
        table.add_column("Thinking", width=50)
        table.add_column("Action", width=20, style="cyan")
        table.add_column("Status", width=10)
        
        for step in agent_steps:
            iteration, status, thought, action = step
            status_icons = {
                "thinking": "[yellow]○[/yellow]",
                "success": "[green]✓[/green]",
                "error": "[red]✗[/red]",
                "completed": "[green]★[/green]",
            }
            status_icon = status_icons.get(status, "[dim]?[/dim]")
            thought_short = (thought[:47] + "...") if len(thought) > 50 else thought
            table.add_row(str(iteration), thought_short, action or "-", status_icon)
        
        return table
    
    def on_progress(iteration: int, status: str, thought: str, action: str):
        # Update or add step
        for i, step in enumerate(agent_steps):
            if step[0] == iteration:
                agent_steps[i] = (iteration, status, thought, action)
                return
        agent_steps.append((iteration, status, thought, action))
    
    # Create agent
    agent = ReActAgent(
        tool_registry=tool_registry,
        sandbox=sandbox,
        on_progress=on_progress,
        max_iterations=15,
    )
    
    # Run with live display
    if not quiet and not output_json:
        with Live(build_agent_table(), console=console, refresh_per_second=4) as live:
            async def run_with_updates():
                result = await agent.run(request)
                return result
            
            # Run agent with periodic table updates
            async def run_agent():
                task = asyncio.create_task(run_with_updates())
                while not task.done():
                    live.update(build_agent_table())
                    await asyncio.sleep(0.25)
                live.update(build_agent_table())
                return await task
            
            state = asyncio.run(run_agent())
    else:
        state = asyncio.run(agent.run(request))
    
    # Output results
    if output_json:
        output = {
            "mode": "agentic",
            "goal": state.goal,
            "status": state.status,
            "steps": [
                {
                    "iteration": s.iteration,
                    "thought": s.thought.reasoning,
                    "action": s.action.tool if s.action else None,
                    "result": s.result.status if s.result else None,
                }
                for s in state.steps
            ],
            "final_answer": state.final_answer,
            "error": state.error,
        }
        print(json.dumps(output, indent=2, default=str))
        raise typer.Exit(code=0 if state.status == "completed" else 1)
    
    if not quiet:
        console.print()
        
        # Status summary
        status_style = "green" if state.status == "completed" else "yellow" if state.status == "max_iterations" else "red"
        status_icon = "✓" if state.status == "completed" else "⚠" if state.status == "max_iterations" else "✗"
        
        console.print(
            Panel(
                f"[{status_style}]{status_icon}[/{status_style}] {state.status.replace('_', ' ').title()} after {len(state.steps)} steps",
                title="[bold]Agent Result[/bold]",
                title_align="left",
                border_style=status_style,
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        
        if state.final_answer:
            console.print()
            console.print(Panel(
                state.final_answer,
                title="[bold green]Summary[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            ))
        
        if state.error:
            console.print()
            console.print(f"[red]Error:[/red] {state.error}")
        
        # Show verbose step details if requested
        if verbose and state.steps:
            console.print()
            tree = Tree("[dim]Step Details[/dim]", guide_style="dim")
            for step in state.steps:
                step_branch = tree.add(f"[yellow]Step {step.iteration}[/yellow]")
                step_branch.add(f"[dim]Thought: {step.thought.reasoning[:100]}...[/dim]")
                if step.action:
                    step_branch.add(f"[cyan]Action: {step.action.tool}[/cyan]")
                if step.result:
                    result_style = "green" if step.result.status == "success" else "red"
                    step_branch.add(f"[{result_style}]Result: {step.result.status}[/{result_style}]")
            console.print(tree)
    
    raise typer.Exit(code=0 if state.status == "completed" else 1)


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


@app.command()
def tasks(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of tasks to show"),
    state: str = typer.Option(None, "--state", "-s", help="Filter by state (pending, executing, completed, failed)"),
    all_tasks: bool = typer.Option(False, "--all", "-a", help="Show all states"),
):
    """List recent tasks and their status."""
    from agent.orchestrator.task_manager import get_task_manager, TaskState
    
    tm = get_task_manager()
    
    states = None
    if state and not all_tasks:
        try:
            states = [TaskState(state)]
        except ValueError:
            console.print(f"[red]Invalid state:[/red] {state}")
            console.print(f"[dim]Valid states: pending, planning, awaiting_approval, approved, executing, completed, failed, rejected, cancelled[/dim]")
            raise typer.Exit(code=1)
    
    task_list = tm.get_tasks(states=states, limit=limit)
    
    if not task_list:
        console.print("[dim]No tasks found.[/dim]")
        raise typer.Exit(code=0)
    
    table = Table(title="Recent Tasks", box=box.ROUNDED)
    table.add_column("ID", style="cyan", width=12)
    table.add_column("State", width=18)
    table.add_column("Request", max_width=40)
    table.add_column("Created", style="dim", width=16)
    
    state_styles = {
        "pending": "dim",
        "planning": "magenta",
        "awaiting_approval": "yellow",
        "approved": "blue",
        "executing": "cyan",
        "completed": "green",
        "failed": "red",
        "rejected": "red",
        "cancelled": "dim",
    }
    
    for task in task_list:
        state_style = state_styles.get(task.state.value, "white")
        request_preview = task.request[:40] + "…" if len(task.request) > 40 else task.request
        created = task.created_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(
            task.id[:12],
            f"[{state_style}]{task.state.value}[/{state_style}]",
            request_preview,
            created,
        )
    
    console.print(table)


@app.command()
def status(
    task_id: str = typer.Argument(..., help="Task ID to check"),
):
    """Get detailed status of a task."""
    from agent.orchestrator.task_manager import get_task_manager
    
    tm = get_task_manager()
    task = tm.get_task(task_id)
    
    if not task:
        # Try prefix match
        matching = [t for t in tm.get_tasks(limit=100) if t.id.startswith(task_id)]
        if len(matching) == 1:
            task = matching[0]
        elif len(matching) > 1:
            console.print(f"[yellow]Multiple tasks match '{task_id}':[/yellow]")
            for t in matching[:5]:
                console.print(f"  {t.id}")
            raise typer.Exit(code=1)
        else:
            console.print(f"[red]Task not found:[/red] {task_id}")
            raise typer.Exit(code=1)
    
    # Task details
    console.print()
    console.print(Panel(
        f"[bold]{task.request}[/bold]\n\n"
        f"[dim]ID:[/dim] {task.id}\n"
        f"[dim]State:[/dim] {task.state.value}\n"
        f"[dim]Created:[/dim] {task.created_at}\n"
        f"[dim]Workspace:[/dim] {task.workspace_path or 'N/A'}",
        title="Task Details",
        border_style="cyan",
    ))
    
    # Plan steps
    if task.plan:
        console.print()
        tree = Tree("[bold]Execution Plan[/bold]", guide_style="dim")
        for step in task.plan.get("steps", []):
            result = task.step_results.get(step["id"], {})
            status = result.get("status", "pending")
            status_icon = {"success": "✓", "error": "✗", "skipped": "◌"}.get(status, "○")
            status_color = {"success": "green", "error": "red", "skipped": "dim"}.get(status, "yellow")
            tree.add(f"[{status_color}]{status_icon}[/{status_color}] {step['id']} — {step.get('description', step['action'])}")
        console.print(tree)
    
    # Summary
    if task.summary:
        console.print()
        console.print(Panel(task.summary, title="Summary", border_style="green"))
    
    # Error
    if task.error:
        console.print()
        console.print(Panel(task.error, title="Error", border_style="red"))


@app.command()
def approve(
    task_id: str = typer.Argument(..., help="Task ID to approve"),
):
    """Approve a pending task for execution."""
    from agent.orchestrator.task_manager import get_task_manager, TaskState
    
    tm = get_task_manager()
    task = tm.get_task(task_id)
    
    if not task:
        console.print(f"[red]Task not found:[/red] {task_id}")
        raise typer.Exit(code=1)
    
    if task.state != TaskState.AWAITING_APPROVAL:
        console.print(f"[yellow]Task is not awaiting approval (current state: {task.state.value})[/yellow]")
        raise typer.Exit(code=1)
    
    # Show plan for review
    if task.plan:
        console.print()
        console.print("[bold]Execution Plan:[/bold]")
        for i, step in enumerate(task.plan.get("steps", []), 1):
            console.print(f"  {i}. {step['action']} — {step.get('description', '')}")
        console.print()
    
    if typer.confirm("Approve this plan for execution?", default=True):
        tm.update_state(task_id, TaskState.APPROVED)
        console.print(f"[green]✓ Task approved:[/green] {task_id}")
        console.print("[dim]Task will be executed on next server request.[/dim]")
    else:
        tm.update_state(task_id, TaskState.REJECTED)
        console.print(f"[red]✗ Task rejected:[/red] {task_id}")


@app.command()
def reject(
    task_id: str = typer.Argument(..., help="Task ID to reject"),
    reason: str = typer.Option(None, "--reason", "-r", help="Reason for rejection"),
):
    """Reject a pending task."""
    from agent.orchestrator.task_manager import get_task_manager, TaskState
    
    tm = get_task_manager()
    task = tm.get_task(task_id)
    
    if not task:
        console.print(f"[red]Task not found:[/red] {task_id}")
        raise typer.Exit(code=1)
    
    if task.state not in (TaskState.AWAITING_APPROVAL, TaskState.PENDING, TaskState.PLANNING):
        console.print(f"[yellow]Task cannot be rejected (current state: {task.state.value})[/yellow]")
        raise typer.Exit(code=1)
    
    tm.update_state(task_id, TaskState.REJECTED, reason)
    console.print(f"[red]✗ Task rejected:[/red] {task_id}")


def main():
    app()
