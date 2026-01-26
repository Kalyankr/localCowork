"""Pure agentic loop - ReAct-based autonomous agent."""

import asyncio
from typing import Optional
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
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
                console.clear()
                _show_welcome(model)
                continue
            
            _process_input_agentic(user_input, model)
            
        except KeyboardInterrupt:
            console.print("\n  [dim]Ctrl+C. Type 'quit' to exit.[/dim]\n")
        except EOFError:
            console.print("\n  [dim]Goodbye! 👋[/dim]\n")
            break


def _process_input_agentic(user_input: str, model: str):
    """Process input using the ReAct agentic loop."""
    from agent.orchestrator.react_agent import ReActAgent
    from agent.orchestrator.deps import get_tool_registry, get_sandbox
    from agent.llm.client import LLMError
    
    console.print()
    
    tool_registry = get_tool_registry()
    sandbox = get_sandbox()
    
    # State for live display
    current_state = {
        "iteration": 0,
        "thought": "",
        "action": "",
        "status": "thinking",
        "steps": []  # List of (iteration, action, status, thought_preview)
    }
    
    def build_agent_display():
        """Build live display showing agent's reasoning and actions."""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
        table.add_column("", width=60)
        
        # Current thinking
        if current_state["status"] == "thinking":
            table.add_row(f"  [yellow]🤔 Thinking...[/yellow]")
        elif current_state["thought"]:
            thought_preview = current_state["thought"][:80]
            if len(current_state["thought"]) > 80:
                thought_preview += "..."
            table.add_row(f"  [cyan]💭 {thought_preview}[/cyan]")
        
        # Current action
        if current_state["action"]:
            table.add_row(f"  [green]⚡ {current_state['action']}[/green]")
        
        # Previous steps summary
        if current_state["steps"]:
            table.add_row("")
            for step in current_state["steps"][-5:]:  # Last 5 steps
                iter_num, action, status, _ = step
                icon = "✓" if status == "success" else "✗" if status == "error" else "○"
                color = "green" if status == "success" else "red" if status == "error" else "dim"
                table.add_row(f"  [{color}]{icon} Step {iter_num}: {action}[/{color}]")
        
        return table
    
    def on_progress(iteration: int, status: str, thought: str, action: Optional[str]):
        """Callback for agent progress updates."""
        current_state["iteration"] = iteration
        current_state["status"] = status
        current_state["thought"] = thought
        current_state["action"] = action or ""
        
        if status in ("success", "error") and action:
            current_state["steps"].append((iteration, action, status, thought[:50]))
    
    try:
        agent = ReActAgent(
            tool_registry=tool_registry,
            sandbox=sandbox,
            on_progress=on_progress,
            max_iterations=15
        )
        
        console.print(f"  [bold cyan]🤖 Working on your request...[/bold cyan]")
        console.print()
        
        # Run agent with live display
        with Live(build_agent_display(), console=console, refresh_per_second=4) as live:
            async def run_with_display():
                async def updater():
                    while True:
                        live.update(build_agent_display())
                        await asyncio.sleep(0.2)
                
                update_task = asyncio.create_task(updater())
                try:
                    return await agent.run(user_input)
                finally:
                    update_task.cancel()
                    try:
                        await update_task
                    except asyncio.CancelledError:
                        pass
            
            state = asyncio.run(run_with_display())
            live.update(build_agent_display())
        
        console.print()
        
        # Show final result
        if state.status == "completed":
            _show_agent_result(state, model)
        elif state.status == "failed":
            console.print(f"  [red]✗ Failed: {state.error}[/red]")
        elif state.status == "max_iterations":
            console.print(f"  [yellow]⚠ Reached max iterations without completing[/yellow]")
            if state.steps:
                # Show what was accomplished
                _show_agent_result(state, model)
        
    except LLMError as e:
        print_error("AI Error", str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print_error("Error", str(e))
    
    console.print()


def _show_agent_result(state, model: str):
    """Display the agent's final result with context."""
    from agent.llm.client import call_llm
    
    # Build summary from agent's work
    if state.final_answer:
        summary = state.final_answer
    else:
        # Generate summary from context
        context_summary = "\n".join([
            f"- {k}: {str(v)[:100]}..." if len(str(v)) > 100 else f"- {k}: {v}"
            for k, v in list(state.context.items())[:10]
        ])
        
        prompt = f"""Summarize what was accomplished for this goal in 1-3 friendly sentences.

Goal: {state.goal}

Data gathered:
{context_summary}

Steps taken: {len(state.steps)}
Final status: {state.status}

Be concise and conversational. Focus on what was achieved."""
        
        summary = call_llm(prompt)
    
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
    console.print(f"  [bold cyan]│[/bold cyan] [dim]Agentic AI assistant[/dim]                [bold cyan]│[/bold cyan]")
    console.print(f"  [bold cyan]╰{'─' * 40}╯[/bold cyan]")
    console.print()
    console.print(f"  [dim]Model:[/dim] [cyan]{model}[/cyan]")
    console.print(f"  [dim]I'll think step-by-step and adapt as I work.[/dim]")
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
    console.print("  [bold]Just describe what you want:[/bold]")
    console.print()
    console.print("    [cyan]organize my downloads by file type[/cyan]")
    console.print("    [cyan]find all PDFs and summarize them[/cyan]")
    console.print("    [cyan]search the web for Python tutorials[/cyan]")
    console.print("    [cyan]create a report from this data[/cyan]")
    console.print()
    console.print("  [dim]I'll figure out the steps and adapt as I work.[/dim]")
    console.print()
    console.print("  [dim]/clear[/dim] - reset  [dim]/quit[/dim] - exit")
    console.print()
