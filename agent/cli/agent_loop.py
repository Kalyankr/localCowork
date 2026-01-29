"""Pure agentic loop - ReAct-based autonomous agent."""

import asyncio
import re
import shutil
import time
from typing import Optional
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from agent.cli.console import (
    console,
    print_error,
    print_padding,
    format_duration,
)
from agent.version import __version__


# Get terminal width for proper formatting
def _get_width() -> int:
    """Get terminal width, with a reasonable default."""
    return min(shutil.get_terminal_size().columns - 4, 100)


def _is_directory_listing(text: str) -> bool:
    """Check if text looks like a directory listing (ls -la output)."""
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return False

    # Check for ls -la style output (drwxr-xr-x, -rw-r--r--, etc.)
    permission_pattern = re.compile(r"^[drwxlst-]{10}")
    matching_lines = sum(1 for line in lines if permission_pattern.match(line.strip()))

    # If more than 50% of lines look like permission strings, it's a directory listing
    if matching_lines > len(lines) * 0.5:
        return True

    # Also check for common hidden files that appear in ls output
    hidden_file_indicators = [
        ".DS_Store",
        ".localized",
        ".Trash",
        ".CFUserTextEncoding",
    ]
    for indicator in hidden_file_indicators:
        if indicator in text:
            return True

    return False


def run_agent(model_override: str = None):
    """Main agent loop - handles everything autonomously."""
    from agent.llm.client import check_ollama_health
    from agent.config import settings as app_settings

    global settings
    settings = app_settings

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
    """Interactive agent loop with conversation memory."""
    console.clear()
    _show_welcome(model)

    # Conversation history for context
    conversation_history = []

    while True:
        try:
            user_input = _get_input()

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/q", "/exit", "quit", "exit"):
                _show_goodbye()
                break

            if user_input.lower() in ("/help", "/h"):
                _show_help()
                continue

            if user_input.lower() == "/clear":
                console.clear()
                _show_welcome(model)
                conversation_history.clear()  # Also clear history
                continue

            if user_input.lower() == "/status":
                _show_status(model)
                continue

            if user_input.lower() == "/history":
                _show_history(conversation_history)
                continue

            if user_input.lower().startswith("/model "):
                new_model = user_input[7:].strip()
                if new_model:
                    model = new_model
                    console.print(
                        f"  [green]✓[/green] Switched to model: [cyan]{model}[/cyan]"
                    )
                    console.print()
                continue

            result = _process_input_agentic(user_input, model, conversation_history)

            # Add to conversation history
            if result:
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": result})
                # Keep history manageable
                max_history = settings.max_history_messages
                if len(conversation_history) > max_history:
                    conversation_history = conversation_history[-max_history:]

        except KeyboardInterrupt:
            console.print()
            console.print(
                "  [dim]Interrupted. Type [white]/quit[/white] to exit.[/dim]"
            )
            print_padding(1)
        except EOFError:
            _show_goodbye()
            break


def _process_input_agentic(
    user_input: str, model: str, conversation_history: list = None
) -> Optional[str]:
    """Process input using the ReAct agentic loop.

    Returns the assistant's response for conversation history.
    """
    from agent.orchestrator.react_agent import ReActAgent
    from agent.orchestrator.deps import get_sandbox
    from agent.llm.client import LLMError

    sandbox = get_sandbox()

    # State for live display
    current_state = {
        "iteration": 0,
        "thought": "",
        "action": "",
        "status": "thinking",
        "steps": [],  # List of (iteration, action, status, thought_preview)
    }

    # Nice spinner animation frames
    spinner_frames = ["◐", "◓", "◑", "◒"]
    frame_idx = [0]  # Use list to mutate in closure

    def build_agent_display():
        """Build minimal spinner display with hidden details."""
        frame_idx[0] = (frame_idx[0] + 1) % len(spinner_frames)
        spinner = spinner_frames[frame_idx[0]]

        iteration = current_state["iteration"]
        status = current_state["status"]
        steps = current_state["steps"]

        # Build single-line spinner display
        line = Text()
        line.append("  ")

        if status == "thinking":
            line.append(f"{spinner} ", style="bold yellow")
            line.append("Thinking", style="bold yellow")
        elif status == "executing":
            line.append(f"{spinner} ", style="bold cyan")
            line.append("Executing", style="bold cyan")

        # Show step count
        if iteration > 0:
            line.append("  ", style="dim")
            line.append(f"step {iteration}", style="dim")

        # Show completed steps as dots
        if steps:
            line.append("  ", style="dim")
            for step in steps[-5:]:
                _, _, step_status, _ = step
                if step_status == "success":
                    line.append("●", style="green")
                elif step_status == "error":
                    line.append("●", style="red")
                else:
                    line.append("○", style="dim")

        # Current action hint (truncated)
        if current_state["action"]:
            action_text = current_state["action"]
            if action_text.startswith("shell:"):
                action_text = action_text[6:].strip()
            elif action_text.startswith("python:"):
                action_text = "py"
            if len(action_text) > 40:
                action_text = action_text[:37] + "..."
            line.append(f"  {action_text}", style="dim")

        return line

    def on_progress(iteration: int, status: str, thought: str, action: Optional[str]):
        """Callback for agent progress updates."""
        current_state["iteration"] = iteration
        current_state["thought"] = thought
        current_state["action"] = action or ""

        # Map status to display status
        if status in ("thinking", "planning"):
            current_state["status"] = "thinking"
        elif status in ("acting", "running", "executing"):
            current_state["status"] = "executing"
        else:
            current_state["status"] = status

        if status in ("success", "error") and action:
            current_state["steps"].append(
                (iteration, action, status, thought[:50] if thought else "")
            )

    async def on_confirm(command: str, reason: str, message: str) -> bool:
        """Prompt user for confirmation on dangerous operations."""
        from rich.prompt import Confirm

        # Stop live display temporarily to show confirmation
        width = _get_width()
        console.print()
        console.print(f"  [red]╭{'─' * (width - 8)}╮[/red]")
        console.print("  [red]│[/red] [bold red]⚠ Confirmation Required[/bold red]")
        console.print("  [red]│[/red]")
        # Wrap message lines
        for line in message.split("\n"):
            if len(line) > width - 12:
                line = line[: width - 15] + "..."
            console.print(f"  [red]│[/red]  [dim]{line}[/dim]")
        console.print(f"  [red]╰{'─' * (width - 8)}╯[/red]")

        try:
            return Confirm.ask("  [bold]Proceed?[/bold]", default=False)
        except (KeyboardInterrupt, EOFError):
            return False

    try:
        agent = ReActAgent(
            sandbox=sandbox,
            on_progress=on_progress,
            on_confirm=on_confirm,
            max_iterations=settings.max_agent_iterations,
            conversation_history=conversation_history or [],
        )

        start_time = time.time()

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

        elapsed = time.time() - start_time
        console.print(f"  [dim]✓ Done in {format_duration(elapsed)}[/dim]")

        # Get the response for history
        response_text = None

        # Show final result
        if state.status == "completed":
            response_text = _show_agent_result(state, model)
        elif state.status == "failed":
            console.print(f"  [red]✗ Failed: {state.error}[/red]")
            response_text = f"Failed: {state.error}"
        elif state.status == "max_iterations":
            console.print(
                "  [yellow]⚠ Reached max iterations without completing[/yellow]"
            )
            if state.steps:
                # Show what was accomplished
                response_text = _show_agent_result(state, model)

        return response_text

    except LLMError as e:
        print_error("AI Error", str(e))
        return None
    except Exception as e:
        import traceback

        traceback.print_exc()
        print_error("Error", str(e))
        return None

    console.print()


def _show_agent_result(state, model: str) -> str:
    """Display the agent's final result with context.

    Returns the summary text for conversation history.
    """
    from agent.llm.client import call_llm

    # Build summary from agent's work
    if state.final_answer:
        summary = state.final_answer
    else:
        # Generate summary from context
        context_summary = "\n".join(
            [
                f"- {k}: {str(v)[:100]}..." if len(str(v)) > 100 else f"- {k}: {v}"
                for k, v in list(state.context.items())[:10]
            ]
        )

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

    return summary


def _show_context_data(context: dict):
    """Display actual data collected by the agent."""
    if not context:
        return

    width = _get_width()

    # Filter out non-displayable or uninteresting context
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
            # Skip raw directory listings (ls -la output, file permissions, etc.)
            if _is_directory_listing(value):
                continue
            # Skip very short outputs (likely just confirmations)
            if len(value.strip()) < 10:
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
                    item_str = item_str[: width - 11] + "..."
                output_lines.append(f"  • {item_str}")
            if len(value) > items_to_show:
                output_lines.append(
                    f"  [dim]... and {len(value) - items_to_show} more[/dim]"
                )

        elif isinstance(value, dict):
            import json

            formatted = json.dumps(value, indent=2, default=str)
            for line in formatted.split("\n")[:20]:
                if len(line) > width - 4:
                    line = line[: width - 7] + "..."
                output_lines.append(f"  {line}")

        elif isinstance(value, str):
            lines = value.strip().split("\n")
            lines_to_show = min(len(lines), max_lines - len(output_lines))
            for line in lines[:lines_to_show]:
                if len(line) > width - 4:
                    line = line[: width - 7] + "..."
                output_lines.append(f"  {line}")
            if len(lines) > lines_to_show:
                output_lines.append(
                    f"  [dim]... {len(lines) - lines_to_show} more lines[/dim]"
                )
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
    """Display agent response with clean formatting."""
    width = _get_width()

    console.print("  [bold cyan]◆[/bold cyan] [bold white]LocalCowork[/bold white]")

    # Clean and wrap the text properly
    for line in text.split("\n"):
        if len(line) > width - 8:
            # Word wrap long lines
            words = line.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 > width - 8:
                    console.print(f"    {current}")
                    current = word
                else:
                    current = f"{current} {word}" if current else word
            if current:
                console.print(f"    {current}")
        else:
            console.print(f"    {line}")

    console.print()


def _show_welcome(model: str):
    """Show welcome screen - compact with ASCII art."""
    width = _get_width()

    console.print()
    console.print(
        f"  [bold cyan]██╗[/bold cyan]  [bold white]LocalCowork[/bold white] [dim]v{__version__}[/dim]  [green]●[/green] [dim]{model}[/dim]"
    )
    console.print(
        "  [bold cyan]███████╗[/bold cyan]  [dim]Type a request or /help[/dim]"
    )
    console.print("  [bright_black]" + "─" * (width - 6) + "[/bright_black]")


def _get_input() -> str:
    """Get user input with box prompt."""
    width = _get_width()
    try:
        console.print()
        console.print(f"  [bright_black]╭{'─' * (width - 8)}╮[/bright_black]")
        user_input = console.input("  [bright_black]│[/bright_black] ")
        console.print(f"  [bright_black]╰{'─' * (width - 8)}╯[/bright_black]")
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        raise


def _show_help():
    """Show compact help."""
    console.print()
    console.print(
        "  [bold]Examples:[/bold] [dim]list files in home[/dim] · [dim]find python files > 1MB[/dim] · [dim]analyze this CSV[/dim]"
    )
    console.print(
        "  [bold]Commands:[/bold] [cyan]/clear[/cyan] [cyan]/status[/cyan] [cyan]/history[/cyan] [cyan]/model X[/cyan] [cyan]/help[/cyan] [cyan]/quit[/cyan]"
    )
    console.print()


def _show_status(model: str):
    """Show current status and settings."""
    import os

    console.print()
    console.print("  [bold cyan]◆[/bold cyan] [bold white]Status[/bold white]")
    console.print()
    console.print(f"    [dim]Version:[/dim]     [white]{__version__}[/white]")
    console.print(f"    [dim]Model:[/dim]       [cyan]{model}[/cyan]")
    console.print(f"    [dim]Working Dir:[/dim] [white]{os.getcwd()}[/white]")
    console.print(
        f"    [dim]Max Steps:[/dim]   [white]{settings.max_agent_iterations}[/white]"
    )
    console.print(f"    [dim]Ollama URL:[/dim]  [white]{settings.ollama_url}[/white]")
    console.print()


def _show_history(conversation_history: list):
    """Show conversation history."""
    console.print()
    console.print(
        "  [bold cyan]◆[/bold cyan] [bold white]Conversation History[/bold white]"
    )
    console.print()

    if not conversation_history:
        console.print("    [dim]No conversation history yet.[/dim]")
        console.print()
        return

    width = _get_width()

    for i, msg in enumerate(conversation_history):
        role = msg["role"]
        content = msg["content"]

        # Truncate long messages
        if len(content) > width - 20:
            content = content[: width - 23] + "..."

        if role == "user":
            console.print(f"    [bold green]You:[/bold green] {content}")
        else:
            console.print(f"    [bold cyan]AI:[/bold cyan] {content}")

    console.print()
    console.print(f"    [dim]{len(conversation_history)} messages in history[/dim]")
    console.print()


def _show_goodbye():
    """Show a clean goodbye message."""
    console.print()
    console.print("  [dim]Session ended. Goodbye![/dim]")
    console.print()
