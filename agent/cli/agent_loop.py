"""Pure agentic loop - ReAct-based autonomous agent."""

import asyncio
import contextlib
import shutil
import time

from rich.live import Live
from rich.text import Text

from agent.cli.console import (
    console,
    format_duration,
    print_error,
    print_padding,
)
from agent.version import __version__


# Get terminal width for proper formatting
def _get_width() -> int:
    """Get terminal width, with a reasonable default."""
    return min(shutil.get_terminal_size().columns - 4, 100)


def run_agent(model_override: str = None):
    """Main agent loop - handles everything autonomously."""
    from agent.config import settings as app_settings
    from agent.llm.client import check_ollama_health

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
) -> str | None:
    """Process input using the ReAct agentic loop.

    Returns the assistant's response for conversation history.
    """
    from agent.llm.client import LLMError
    from agent.orchestrator.deps import get_sandbox
    from agent.orchestrator.react_agent import ReActAgent

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
        """Build spinner display showing current activity."""
        frame_idx[0] = (frame_idx[0] + 1) % len(spinner_frames)
        spinner = spinner_frames[frame_idx[0]]

        iteration = current_state["iteration"]
        status = current_state["status"]
        steps = current_state["steps"]
        action = current_state["action"]

        # Build display
        line = Text()
        line.append("  ")

        if status == "thinking":
            line.append(f"{spinner} ", style="bold yellow")
            line.append("Thinking...", style="yellow")
        elif status == "executing":
            line.append(f"{spinner} ", style="bold cyan")
            # Show what's being executed
            if action:
                if action.startswith("shell:"):
                    cmd = action[6:].strip()
                    if len(cmd) > 50:
                        cmd = cmd[:47] + "..."
                    line.append(f"$ {cmd}", style="cyan")
                elif action.startswith("python:"):
                    line.append("Running Python...", style="cyan")
                else:
                    line.append("Executing...", style="cyan")
            else:
                line.append("Executing...", style="cyan")

        # Show step count and progress dots
        if iteration > 0 or steps:
            line.append("  ", style="dim")
            if steps:
                for step in steps[-5:]:
                    _, _, step_status, _ = step
                    if step_status == "success":
                        line.append("●", style="green")
                    elif step_status == "error":
                        line.append("●", style="red")
                    else:
                        line.append("○", style="dim")

        return line

    def on_progress(iteration: int, status: str, thought: str, action: str | None):
        """Callback for agent progress updates."""
        current_state["iteration"] = iteration
        current_state["thought"] = thought
        current_state["action"] = action or ""

        # Map status to display status
        if status in ("thinking", "planning"):
            current_state["status"] = "thinking"
        else:
            # Any other status (acting, running, executing, success, error) shows as executing
            current_state["status"] = "executing"

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
                    with contextlib.suppress(asyncio.CancelledError):
                        await update_task

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

    return summary


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
    """Get user input with blue rectangle box."""
    width = _get_width()
    inner_width = width - 8
    try:
        console.print()
        console.print()
        # Top of box
        console.print(f"  [blue]╭{'─' * inner_width}╮[/blue]")
        # Input line with left border, user types, then we close
        user_input = console.input("  [blue]│[/blue] [dim]>[/dim] ")
        # Bottom of box
        console.print(f"  [blue]╰{'─' * inner_width}╯[/blue]")
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

    for _i, msg in enumerate(conversation_history):
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
