"""Run command - main task execution."""

import asyncio
import json
import logging
import typer
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich import box

from agent.cli.console import (
    console, Icons, friendly_error, format_status,
    print_success, print_error
)
from agent.orchestrator.planner import generate_plan, summarize_results
from agent.orchestrator.executor import Executor
from agent.orchestrator.react_agent import ReActAgent
from agent.orchestrator.deps import get_tool_registry, get_sandbox
from agent.llm.client import LLMError

logger = logging.getLogger(__name__)

# Lazy-loaded shared dependencies
_tool_registry = None
_sandbox = None


def get_tools():
    """Get or create shared tool registry and sandbox."""
    global _tool_registry, _sandbox
    if _tool_registry is None:
        _tool_registry = get_tool_registry()
        _sandbox = get_sandbox()
    return _tool_registry, _sandbox


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

    # Agentic mode uses ReAct loop
    if agentic:
        _run_agentic(request, tool_registry, sandbox, verbose, quiet, output_json)
        return

    # Planning phase
    task_plan = _generate_plan(request, quiet, output_json, verbose)
    if task_plan is None:
        raise typer.Exit(code=1)

    # Check if this is a chat request
    is_chat = len(task_plan.steps) == 1 and task_plan.steps[0].action == "chat_op"

    # For chat, use simplified flow
    if is_chat and not output_json:
        executor = Executor(plan=task_plan, tool_registry=tool_registry, sandbox=sandbox, parallel=False)
        results = asyncio.run(executor.run())
        chat_result = list(results.values())[0]
        if chat_result.output:
            console.print(Panel(str(chat_result.output), border_style="cyan", box=box.ROUNDED))
        else:
            console.print("[dim]No response[/dim]")
        raise typer.Exit(code=0)

    # Display plan
    if not quiet and not output_json:
        _display_plan(task_plan, show_plan, dry_run, yes)

    # Dry run exits here
    if dry_run:
        if output_json:
            print(json.dumps({"plan": task_plan.model_dump()}))
        else:
            console.print("[yellow]Dry run — no changes made.[/yellow]")
        raise typer.Exit(code=0)

    # Confirmation
    if not yes and not quiet and not output_json:
        if not typer.confirm("Execute this plan?", default=True):
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    # Execute
    results = _execute_plan(task_plan, tool_registry, sandbox, no_parallel, quiet, output_json)

    # Show results
    _display_results(request, task_plan, results, quiet, output_json, verbose)


def _generate_plan(request: str, quiet: bool, output_json: bool, verbose: bool):
    """Generate execution plan from request."""
    if not quiet and not output_json:
        console.print()
        console.print(f"[dim]Task:[/dim] {request}")
        console.print()
        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            try:
                return generate_plan(request)
            except LLMError as e:
                friendly_msg, detail = friendly_error(str(e))
                print_error(f"AI Error: {friendly_msg}", detail)
                if verbose:
                    console.print(f"  [dim]Technical: {e}[/dim]")
                return None
            except Exception as e:
                friendly_msg, detail = friendly_error(str(e))
                print_error(f"Planning failed: {friendly_msg}", detail)
                logger.exception("Plan generation failed")
                return None
    else:
        try:
            return generate_plan(request)
        except (LLMError, Exception) as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            return None


def _display_plan(task_plan, show_plan: bool, dry_run: bool, yes: bool):
    """Display the execution plan."""
    # Count actions by type
    action_counts = {}
    for step in task_plan.steps:
        action_counts[step.action] = action_counts.get(step.action, 0) + 1
    
    actions_summary = ", ".join(f"{v}× {k}" for k, v in action_counts.items())
    
    console.print(
        Panel(
            f"[bold]{len(task_plan.steps)} steps[/bold] — {actions_summary}",
            title=f"{Icons.PLAN} Plan",
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    
    # Expand details if requested
    expand_plan = show_plan or dry_run
    if not expand_plan and not yes:
        expand_plan = typer.confirm("Show plan details?", default=False)
    
    if expand_plan:
        tree = Tree("[bold cyan]Steps[/bold cyan]", guide_style="dim")
        for i, step in enumerate(task_plan.steps, 1):
            deps = f" [dim](← {', '.join(step.depends_on)})[/dim]" if step.depends_on else ""
            desc = step.description[:40] + "…" if step.description and len(step.description) > 40 else (step.description or "")
            branch = tree.add(f"[yellow]{step.action}[/yellow]{deps}")
            if desc:
                branch.add(f"[dim]{i}. {step.id} — {desc}[/dim]")
            else:
                branch.add(f"[dim]{i}. {step.id}[/dim]")
        console.print(tree)
    
    console.print()


def _execute_plan(task_plan, tool_registry, sandbox, no_parallel: bool, quiet: bool, output_json: bool):
    """Execute the plan and return results."""
    step_status = {}
    total_steps = len(task_plan.steps)
    completed_count = 0

    def build_progress_table() -> Table:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Status", width=14)
        table.add_column("Step", style="cyan", width=20)
        table.add_column("Info", style="dim", max_width=35)

        for step in task_plan.steps:
            status = step_status.get(step.id, "pending")
            desc = (step.description or step.action)[:35]
            table.add_row(format_status(status), step.id, desc)

        return table

    def on_progress(step_id: str, status: str, current: int, total: int):
        nonlocal completed_count
        step_status[step_id] = status
        if status in ("success", "error", "skipped"):
            completed_count += 1

    parallel_mode = not no_parallel
    executor = Executor(
        plan=task_plan,
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
    
    return results


def _display_results(request: str, task_plan, results: dict, quiet: bool, output_json: bool, verbose: bool):
    """Display execution results."""
    success_count = sum(1 for r in results.values() if r.status == "success")
    failed_count = sum(1 for r in results.values() if r.status in ("error", "skipped"))

    if output_json:
        summary = summarize_results(request, results)
        output = {
            "plan": task_plan.model_dump(),
            "results": {k: v.model_dump() for k, v in results.items()},
            "summary": summary,
            "stats": {"success": success_count, "failed": failed_count},
        }
        print(json.dumps(output, indent=2))
        raise typer.Exit(code=0 if failed_count == 0 else 1)

    if not quiet:
        console.print()
        
        # Result summary
        result_icon = Icons.SUCCESS if failed_count == 0 else Icons.WARNING
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

        # Show errors
        errors = [(step_id, result.error) for step_id, result in results.items() if result.error]
        if errors:
            console.print()
            for step_id, error in errors:
                friendly_msg, detail = friendly_error(error)
                console.print(f"  [red]{Icons.ERROR}[/red] [bold]{step_id}:[/bold] {friendly_msg} — [dim]{detail}[/dim]")
            
            if verbose or typer.confirm("Show technical details?", default=False):
                error_tree = Tree("[dim]Technical Details[/dim]", guide_style="dim")
                for step_id, error in errors:
                    error_tree.add(f"[dim]{step_id}: {error}[/dim]")
                console.print(error_tree)

        console.print()

    # Check if this was chat
    is_chat = len(task_plan.steps) == 1 and task_plan.steps[0].action == "chat_op"
    
    if is_chat:
        chat_result = list(results.values())[0]
        if chat_result.output:
            console.print(Panel(str(chat_result.output), border_style="cyan", box=box.ROUNDED))
    elif not quiet:
        with console.status("[bold cyan]Summarizing...[/bold cyan]", spinner="dots"):
            summary = summarize_results(request, results)
        console.print(Panel(summary, title="[bold green]Summary[/bold green]", border_style="green", box=box.ROUNDED))
    else:
        summary = summarize_results(request, results)
        console.print(summary)

    raise typer.Exit(code=0 if failed_count == 0 else 1)


def _run_agentic(request: str, tool_registry, sandbox, verbose: bool, quiet: bool, output_json: bool):
    """Run task using ReAct agent (step-by-step reasoning)."""
    
    if not quiet and not output_json:
        console.print()
        console.print(f"[dim]Goal:[/dim] {request}")
        console.print()
        console.print(f"[bold cyan]{Icons.ROBOT} Running in agentic mode (ReAct loop)[/bold cyan]")
        console.print("[dim]The agent will reason step-by-step, adapting as it goes...[/dim]")
        console.print()
    
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
                "thinking": f"[yellow]{Icons.PENDING}[/yellow]",
                "success": f"[green]{Icons.SUCCESS}[/green]",
                "error": f"[red]{Icons.ERROR}[/red]",
                "completed": f"[green]{Icons.STAR}[/green]",
            }
            status_icon = status_icons.get(status, "[dim]?[/dim]")
            thought_short = (thought[:47] + "...") if len(thought) > 50 else thought
            table.add_row(str(iteration), thought_short, action or "-", status_icon)
        
        return table
    
    def on_progress(iteration: int, status: str, thought: str, action: str):
        for i, step in enumerate(agent_steps):
            if step[0] == iteration:
                agent_steps[i] = (iteration, status, thought, action)
                return
        agent_steps.append((iteration, status, thought, action))
    
    agent = ReActAgent(
        tool_registry=tool_registry,
        sandbox=sandbox,
        on_progress=on_progress,
        max_iterations=15,
    )
    
    if not quiet and not output_json:
        with Live(build_agent_table(), console=console, refresh_per_second=4) as live:
            async def run_agent():
                task = asyncio.create_task(agent.run(request))
                while not task.done():
                    live.update(build_agent_table())
                    await asyncio.sleep(0.25)
                live.update(build_agent_table())
                return await task
            
            state = asyncio.run(run_agent())
    else:
        state = asyncio.run(agent.run(request))
    
    _display_agent_results(state, quiet, output_json, verbose)


def _display_agent_results(state, quiet: bool, output_json: bool, verbose: bool):
    """Display agent execution results."""
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
        
        status_style = "green" if state.status == "completed" else "yellow" if state.status == "max_iterations" else "red"
        status_icon = Icons.SUCCESS if state.status == "completed" else Icons.WARNING if state.status == "max_iterations" else Icons.ERROR
        
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


def plan(
    request: str = typer.Argument(..., help="Natural language task description"),
):
    """Generate and display a plan without executing it."""
    console.print()
    with console.status("[bold cyan]Planning...[/bold cyan]", spinner="dots"):
        try:
            task_plan = generate_plan(request)
        except LLMError as e:
            print_error("LLM Error", str(e))
            raise typer.Exit(code=1)
        except Exception as e:
            print_error("Planning failed", str(e))
            raise typer.Exit(code=1)

    print(task_plan.model_dump_json(indent=2))
