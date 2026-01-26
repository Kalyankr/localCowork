"""Task management commands."""

import typer
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box

from agent.cli.console import console, Icons


def tasks(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of tasks to show"),
    state: str = typer.Option(None, "--state", "-s", help="Filter by state"),
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
            console.print(f"[red]{Icons.ERROR} Invalid state:[/red] {state}")
            console.print("[dim]Valid states: pending, planning, awaiting_approval, approved, executing, completed, failed, rejected, cancelled[/dim]")
            raise typer.Exit(code=1)
    
    task_list = tm.get_tasks(states=states, limit=limit)
    
    if not task_list:
        console.print("[dim]No tasks found.[/dim]")
        raise typer.Exit(code=0)
    
    table = Table(title=f"{Icons.PLAN} Recent Tasks", box=box.ROUNDED)
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
            console.print(f"[yellow]{Icons.WARNING} Multiple tasks match '{task_id}':[/yellow]")
            for t in matching[:5]:
                console.print(f"  {t.id}")
            raise typer.Exit(code=1)
        else:
            console.print(f"[red]{Icons.ERROR} Task not found:[/red] {task_id}")
            raise typer.Exit(code=1)
    
    # Task details
    console.print()
    console.print(Panel(
        f"[bold]{task.request}[/bold]\n\n"
        f"[dim]ID:[/dim] {task.id}\n"
        f"[dim]State:[/dim] {task.state.value}\n"
        f"[dim]Created:[/dim] {task.created_at}\n"
        f"[dim]Workspace:[/dim] {task.workspace_path or 'N/A'}",
        title=f"{Icons.FOLDER} Task Details",
        border_style="cyan",
    ))
    
    # Plan steps
    if task.plan:
        console.print()
        tree = Tree("[bold]Execution Plan[/bold]", guide_style="dim")
        for step in task.plan.get("steps", []):
            result = task.step_results.get(step["id"], {})
            step_status = result.get("status", "pending")
            status_icons = {
                "success": (Icons.SUCCESS, "green"),
                "error": (Icons.ERROR, "red"),
                "skipped": (Icons.SKIPPED, "dim"),
            }
            icon, color = status_icons.get(step_status, (Icons.PENDING, "yellow"))
            tree.add(f"[{color}]{icon}[/{color}] {step['id']} — {step.get('description', step['action'])}")
        console.print(tree)
    
    # Summary
    if task.summary:
        console.print()
        console.print(Panel(task.summary, title=f"{Icons.SUCCESS} Summary", border_style="green"))
    
    # Error
    if task.error:
        console.print()
        console.print(Panel(task.error, title=f"{Icons.ERROR} Error", border_style="red"))


def approve(
    task_id: str = typer.Argument(..., help="Task ID to approve"),
):
    """Approve a pending task for execution."""
    from agent.orchestrator.task_manager import get_task_manager, TaskState
    
    tm = get_task_manager()
    task = tm.get_task(task_id)
    
    if not task:
        console.print(f"[red]{Icons.ERROR} Task not found:[/red] {task_id}")
        raise typer.Exit(code=1)
    
    if task.state != TaskState.AWAITING_APPROVAL:
        console.print(f"[yellow]{Icons.WARNING} Task is not awaiting approval (current state: {task.state.value})[/yellow]")
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
        console.print(f"[green]{Icons.SUCCESS} Task approved:[/green] {task_id}")
        console.print("[dim]Task will be executed on next server request.[/dim]")
    else:
        tm.update_state(task_id, TaskState.REJECTED)
        console.print(f"[red]{Icons.ERROR} Task rejected:[/red] {task_id}")


def reject(
    task_id: str = typer.Argument(..., help="Task ID to reject"),
    reason: str = typer.Option(None, "--reason", "-r", help="Reason for rejection"),
):
    """Reject a pending task."""
    from agent.orchestrator.task_manager import get_task_manager, TaskState
    
    tm = get_task_manager()
    task = tm.get_task(task_id)
    
    if not task:
        console.print(f"[red]{Icons.ERROR} Task not found:[/red] {task_id}")
        raise typer.Exit(code=1)
    
    if task.state not in (TaskState.AWAITING_APPROVAL, TaskState.PENDING, TaskState.PLANNING):
        console.print(f"[yellow]{Icons.WARNING} Task cannot be rejected (current state: {task.state.value})[/yellow]")
        raise typer.Exit(code=1)
    
    tm.update_state(task_id, TaskState.REJECTED, reason)
    console.print(f"[red]{Icons.ERROR} Task rejected:[/red] {task_id}")
    if reason:
        console.print(f"[dim]Reason: {reason}[/dim]")
