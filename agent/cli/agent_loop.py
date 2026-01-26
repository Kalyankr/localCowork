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
            # Execute the plan with tools
            _execute_plan(plan, user_input, model)
            
    except LLMError as e:
        print_error("AI Error", str(e))
    except Exception as e:
        print_error("Error", str(e))
    
    console.print()


def _execute_plan(plan, request: str, model: str):
    """Execute a plan and show results."""
    import asyncio
    from agent.orchestrator.executor import Executor
    from agent.orchestrator.planner import summarize_results
    from agent.orchestrator.deps import get_tool_registry, get_sandbox
    
    tool_registry = get_tool_registry()
    sandbox = get_sandbox()
    
    # Show what we're doing
    console.print(f"  [dim]╭─ {Icons.ROBOT} Executing {len(plan.steps)} step(s)[/dim]")
    
    step_status = {}
    
    def on_progress(step_id: str, status: str, current: int, total: int):
        step_status[step_id] = status
        icon = {"starting": "○", "success": "✓", "error": "✗", "skipped": "◌"}.get(status, "●")
        color = {"success": "green", "error": "red", "skipped": "dim"}.get(status, "yellow")
        # Find step description
        for step in plan.steps:
            if step.id == step_id:
                desc = (step.description or step.action)[:40]
                console.print(f"  [dim]│[/dim] [{color}]{icon}[/{color}] {desc}")
                break
    
    executor = Executor(
        plan=plan,
        tool_registry=tool_registry,
        sandbox=sandbox,
        on_progress=on_progress,
        parallel=True,
    )
    
    results = asyncio.run(executor.run())
    
    console.print(f"  [dim]╰─────[/dim]")
    console.print()
    
    # Summarize results
    success_count = sum(1 for r in results.values() if r.status == "success")
    failed_count = sum(1 for r in results.values() if r.status in ("error", "skipped"))
    
    if failed_count == 0:
        console.print(f"  [green]{Icons.SUCCESS}[/green] All {success_count} steps completed")
    else:
        console.print(f"  [yellow]⚠[/yellow] {success_count} succeeded, {failed_count} failed")
    
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
