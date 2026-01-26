"""Pure agentic loop - unified handler for questions AND tasks."""

import sys
from rich.panel import Panel
from rich.live import Live
from rich import box

from agent.cli.console import console, Icons, print_error


def run_agent(model_override: str = None):
    """Main agent loop - handles everything autonomously."""
    from agent.llm.client import check_ollama_health, LLMError
    from agent.config import settings
    
    # Check connection
    with console.status("[cyan]Connecting...[/cyan]", spinner="dots"):
        healthy, error = check_ollama_health()
    
    if not healthy:
        print_error("Cannot connect to Ollama", error)
        console.print("\n[dim]Start Ollama: [cyan]ollama serve[/cyan][/dim]")
        raise SystemExit(1)
    
    model = model_override or settings.ollama_model
    
    # Always interactive mode
    _interactive_loop(model)


def _interactive_loop(model: str):
    """Interactive agent loop."""
    console.clear()
    _show_welcome(model)
    
    history = []
    
    while True:
        try:
            user_input = _get_input()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("/quit", "/q", "/exit", "quit", "exit"):
                console.print("\n  [dim]Goodbye! 👋[/dim]\n")
                break
            
            if user_input.lower() in ("/help", "/h"):
                _show_help()
                continue
            
            if user_input.lower() == "/clear":
                history.clear()
                console.print("  [dim]✓ Cleared[/dim]\n")
                continue
            
            _process_input(user_input, model, history)
            
        except KeyboardInterrupt:
            console.print("\n  [dim]Ctrl+C. Type 'quit' to exit.[/dim]\n")
        except EOFError:
            console.print("\n  [dim]Goodbye! 👋[/dim]\n")
            break


def _process_input(user_input: str, model: str, history: list):
    """Process input - agent decides whether to use tools or just respond."""
    from agent.orchestrator.planner import generate_plan
    from agent.orchestrator.executor import Executor
    from agent.orchestrator.deps import get_tool_registry, get_sandbox
    from agent.llm.client import LLMError
    
    console.print()
    
    try:
        # Let the planner decide what to do
        with console.status("  [cyan]Thinking...[/cyan]", spinner="dots"):
            plan = generate_plan(user_input)
        
        # Check if it's a pure chat response (no tools needed)
        is_chat = len(plan.steps) == 1 and plan.steps[0].action == "chat_op"
        
        if is_chat:
            # Just show the chat response
            _show_response(plan.steps[0].args.get("response", ""), model)
        else:
            # Show plan and ask for approval
            if _show_plan_and_approve(plan):
                _execute_plan(plan, user_input, model)
            else:
                console.print("  [dim]Cancelled.[/dim]")
            
    except LLMError as e:
        print_error("AI Error", str(e))
    except Exception as e:
        print_error("Error", str(e))
    
    console.print()


def _show_plan_and_approve(plan) -> bool:
    """Show the plan and ask for user approval."""
    console.print(f"  [bold cyan]📋 Plan[/bold cyan] ({len(plan.steps)} steps)")
    console.print()
    
    for i, step in enumerate(plan.steps, 1):
        action = step.action
        desc = step.description or ""
        
        # Show action with icon
        action_icons = {
            "shell_op": "💻",
            "read_file": "📄",
            "write_file": "✏️",
            "list_dir": "📁",
            "fetch_url": "🌐",
            "run_python": "🐍",
            "search_files": "🔍",
        }
        icon = action_icons.get(action, "•")
        
        console.print(f"  {i}. {icon} [cyan]{action}[/cyan]")
        if desc:
            console.print(f"     [dim]{desc[:60]}{'...' if len(desc) > 60 else ''}[/dim]")
        
        # Show key args
        if step.args:
            for key, val in list(step.args.items())[:2]:
                if val and key not in ("response",):
                    val_str = str(val)[:40]
                    console.print(f"     [dim]{key}: {val_str}{'...' if len(str(val)) > 40 else ''}[/dim]")
    
    console.print()
    
    # Ask for approval
    try:
        response = console.input("  [yellow]Execute?[/yellow] [dim](y/n)[/dim] ").strip().lower()
        return response in ("y", "yes", "")
    except (KeyboardInterrupt, EOFError):
        return False


def _execute_plan(plan, request: str, model: str):
    """Execute a plan and show results with live progress."""
    import asyncio
    from rich.live import Live
    from rich.table import Table
    from rich import box
    from agent.orchestrator.executor import Executor
    from agent.orchestrator.planner import summarize_results
    from agent.orchestrator.deps import get_tool_registry, get_sandbox
    
    tool_registry = get_tool_registry()
    sandbox = get_sandbox()
    
    console.print()
    console.print(f"  [bold cyan]⚡ Executing[/bold cyan]")
    console.print()
    
    step_status = {}
    step_errors = {}
    
    def build_progress_table():
        """Build a live-updating progress table."""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("", width=3)
        table.add_column("Step", width=50)
        table.add_column("Status", width=10)
        
        for step in plan.steps:
            status = step_status.get(step.id, "pending")
            desc = (step.description or step.action)[:45]
            
            icons = {
                "pending": ("○", "dim"),
                "starting": ("●", "yellow"),
                "success": ("✓", "green"),
                "error": ("✗", "red"),
                "skipped": ("◌", "dim"),
            }
            icon, color = icons.get(status, ("●", "yellow"))
            
            table.add_row(
                f"  [{color}]{icon}[/{color}]",
                f"[{color}]{desc}[/{color}]",
                f"[{color}]{status}[/{color}]"
            )
        
        return table
    
    def on_progress(step_id: str, status: str, current: int, total: int):
        step_status[step_id] = status
    
    executor = Executor(
        plan=plan,
        tool_registry=tool_registry,
        sandbox=sandbox,
        on_progress=on_progress,
        parallel=True,
    )
    
    # Run with live progress display
    with Live(build_progress_table(), console=console, refresh_per_second=4) as live:
        async def run_with_updates():
            async def updater():
                while len([s for s in step_status.values() if s in ("success", "error", "skipped")]) < len(plan.steps):
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
        
        results = asyncio.run(run_with_updates())
        live.update(build_progress_table())
    
    console.print()
    
    # Show results summary
    success_count = sum(1 for r in results.values() if r.status == "success")
    failed_count = sum(1 for r in results.values() if r.status in ("error", "skipped"))
    
    if failed_count == 0:
        console.print(f"  [green]✓[/green] All {success_count} steps completed")
    else:
        console.print(f"  [yellow]⚠[/yellow] {success_count} succeeded, {failed_count} failed")
        # Show errors
        for step_id, result in results.items():
            if result.error:
                console.print(f"    [red]✗[/red] {step_id}: [dim]{result.error[:60]}[/dim]")
    
    console.print()
    
    # Generate and show summary
    with console.status("  [cyan]Summarizing...[/cyan]", spinner="dots"):
        summary = summarize_results(request, results)
    
    _show_response(summary, model)


def _show_response(text: str, model: str):
    """Display agent response in a nice box."""
    console.print(f"  [dim]╭─ {Icons.ROBOT} [/dim][cyan]{model}[/cyan]")
    for line in text.split("\n"):
        console.print(f"  [dim]│[/dim] {line}")
    console.print(f"  [dim]╰─────[/dim]")


def _show_welcome(model: str):
    """Show welcome screen."""
    console.print()
    console.print(f"  [bold cyan]╭{'─' * 40}╮[/bold cyan]")
    console.print(f"  [bold cyan]│[/bold cyan] {Icons.ROBOT} [bold]LocalCowork[/bold]                       [bold cyan]│[/bold cyan]")
    console.print(f"  [bold cyan]│[/bold cyan] [dim]Your local AI agent[/dim]                  [bold cyan]│[/bold cyan]")
    console.print(f"  [bold cyan]╰{'─' * 40}╯[/bold cyan]")
    console.print()
    console.print(f"  [dim]Model:[/dim] [cyan]{model}[/cyan]")
    console.print(f"  [dim]Just type. I'll figure out what to do.[/dim]")
    console.print()


def _get_input() -> str:
    """Get user input with styled prompt."""
    console.print(f"  [green]╭─────────────────────────────────────────────╮[/green]")
    console.print(f"  [green]│[/green] ", end="")
    
    lines = []
    try:
        while True:
            line = input()
            if line.endswith("\\"):
                lines.append(line[:-1])
                console.print(f"  [green]│[/green] ", end="")
            else:
                lines.append(line)
                break
    except (KeyboardInterrupt, EOFError):
        console.print()
        console.print(f"  [green]╰─────────────────────────────────────────────╯[/green]")
        raise
    
    console.print(f"  [green]╰─────────────────────────────────────────────╯[/green]")
    return "\n".join(lines).strip()


def _show_help():
    """Show minimal help."""
    console.print()
    console.print("  [bold]Just type what you need:[/bold]")
    console.print()
    console.print("    [cyan]list files in ~/Downloads[/cyan]")
    console.print("    [cyan]what is machine learning?[/cyan]")
    console.print("    [cyan]summarize report.pdf[/cyan]")
    console.print("    [cyan]download example.com/file.txt[/cyan]")
    console.print()
    console.print("  [dim]/clear[/dim] - reset  [dim]/quit[/dim] - exit")
    console.print()
