"""Pure agentic loop - ReAct-based autonomous agent."""

import asyncio
import shutil
from typing import Optional
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box

from agent.cli.console import console, Icons, print_error


# Get terminal width for proper formatting
def _get_width() -> int:
    """Get terminal width, with a reasonable default."""
    return min(shutil.get_terminal_size().columns - 4, 100)


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
        width = _get_width()
        
        lines = []
        
        # Current thinking
        if current_state["status"] == "thinking":
            lines.append(Text("  🤔 Thinking...", style="yellow"))
        elif current_state["thought"]:
            thought_text = current_state["thought"]
            if len(thought_text) > width - 10:
                thought_text = thought_text[:width - 13] + "..."
            lines.append(Text(f"  💭 {thought_text}", style="cyan"))
        
        # Current action
        if current_state["action"]:
            action_text = current_state["action"]
            if len(action_text) > width - 10:
                action_text = action_text[:width - 13] + "..."
            lines.append(Text(f"  ⚡ {action_text}", style="green"))
        
        # Previous steps summary
        if current_state["steps"]:
            lines.append(Text(""))
            for step in current_state["steps"][-5:]:
                iter_num, action, status, _ = step
                icon = "✓" if status == "success" else "✗" if status == "error" else "○"
                color = "green" if status == "success" else "red" if status == "error" else "dim"
                step_text = f"  {icon} Step {iter_num}: {action}"
                if len(step_text) > width - 4:
                    step_text = step_text[:width - 7] + "..."
                lines.append(Text(step_text, style=color))
        
        # Build table with lines
        table = Table(box=None, show_header=False, padding=(0, 0), expand=False)
        table.add_column("", width=width)
        for line in lines:
            table.add_row(line)
        
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
    
    # Show actual data from context if there's meaningful output
    _show_context_data(state.context)


def _show_context_data(context: dict):
    """Display actual data collected by the agent."""
    if not context:
        return
    
    width = _get_width()
    
    # Filter out non-displayable or error context
    displayable = {}
    for key, value in context.items():
        if value is None or value == "" or value == []:
            continue
        # Skip error outputs or code that failed
        if isinstance(value, str):
            if "SyntaxError" in value or "Traceback" in value or "Error:" in value:
                continue
            if value.startswith("import ") or value.startswith("def "):
                continue
        displayable[key] = value
    
    if not displayable:
        return
    
    # Build output lines
    output_lines = []
    max_lines = 50  # Total max lines to show
    
    for key, value in displayable.items():
        if len(output_lines) >= max_lines:
            output_lines.append("[dim]... output truncated ...[/dim]")
            break
            
        if isinstance(value, list):
            # Show list items
            items_to_show = min(len(value), max_lines - len(output_lines))
            for item in value[:items_to_show]:
                item_str = str(item)
                if len(item_str) > width - 8:
                    item_str = item_str[:width - 11] + "..."
                output_lines.append(f"  • {item_str}")
            if len(value) > items_to_show:
                output_lines.append(f"  [dim]... and {len(value) - items_to_show} more[/dim]")
                
        elif isinstance(value, dict):
            import json
            formatted = json.dumps(value, indent=2, default=str)
            for line in formatted.split("\n")[:20]:
                if len(line) > width - 4:
                    line = line[:width - 7] + "..."
                output_lines.append(f"  {line}")
                
        elif isinstance(value, str):
            lines = value.strip().split("\n")
            lines_to_show = min(len(lines), max_lines - len(output_lines))
            for line in lines[:lines_to_show]:
                if len(line) > width - 4:
                    line = line[:width - 7] + "..."
                output_lines.append(f"  {line}")
            if len(lines) > lines_to_show:
                output_lines.append(f"  [dim]... {len(lines) - lines_to_show} more lines[/dim]")
        else:
            output_lines.append(f"  {value}")
    
    if output_lines:
        panel = Panel(
            "\n".join(output_lines),
            title="[dim]Output[/dim]",
            title_align="left",
            border_style="dim",
            padding=(0, 1),
            width=width,
        )
        console.print(panel)


def _show_response(text: str, model: str):
    """Display agent response in a nice panel."""
    width = _get_width()
    
    # Clean and wrap the text properly
    lines = []
    for line in text.split("\n"):
        if len(line) > width - 6:
            # Word wrap long lines
            words = line.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 > width - 6:
                    lines.append(current)
                    current = word
                else:
                    current = f"{current} {word}" if current else word
            if current:
                lines.append(current)
        else:
            lines.append(line)
    
    wrapped_text = "\n".join(lines)
    
    panel = Panel(
        wrapped_text,
        title=f"[cyan]{model}[/cyan]",
        title_align="left",
        border_style="dim",
        padding=(0, 1),
        width=width,
    )
    console.print(panel)


def _show_welcome(model: str):
    """Show welcome screen."""
    width = _get_width()
    
    welcome_text = f"""[bold cyan]LocalCowork[/bold cyan]
[dim]Agentic AI Assistant[/dim]

Model: [cyan]{model}[/cyan]
Type your request and I'll work through it step by step.

[dim]Commands: /help • /clear • /quit[/dim]"""
    
    panel = Panel(
        welcome_text,
        box=box.ROUNDED,
        border_style="cyan",
        padding=(1, 2),
        width=width,
    )
    console.print()
    console.print(panel)
    console.print()


def _get_input() -> str:
    """Get user input with styled prompt."""
    try:
        console.print()
        user_input = console.input("[green]❯[/green] ")
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        raise


def _show_help():
    """Show minimal help."""
    width = _get_width()
    
    help_text = """[bold]Examples[/bold]

  [cyan]list files in my home directory[/cyan]
  [cyan]find all python files in this project[/cyan]
  [cyan]what's the weather like today[/cyan]
  [cyan]summarize this PDF[/cyan]

[dim]I'll figure out how to accomplish your request step by step.[/dim]

[bold]Commands[/bold]
  [dim]/clear[/dim]  Reset the screen
  [dim]/quit[/dim]   Exit"""
    
    panel = Panel(
        help_text,
        title="[cyan]Help[/cyan]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
        width=width,
    )
    console.print()
    console.print(panel)
    console.print()
