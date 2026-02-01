"""Pure agentic loop - ReAct-based autonomous agent."""

import asyncio
import contextlib
import io
import os
import shutil
import sys
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


@contextlib.contextmanager
def suppress_stderr():
    """Suppress stderr output to prevent shell warnings from corrupting CLI display.

    This captures any stderr output (from subprocesses, libraries, etc.)
    during agent execution to maintain clean CLI output.
    """
    # Save original stderr
    original_stderr = sys.stderr
    original_fd = os.dup(2)  # Duplicate the file descriptor

    try:
        # Create a null device to absorb stderr
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)  # Redirect fd 2 (stderr) to devnull
        os.close(devnull)

        # Also redirect Python's stderr object
        sys.stderr = io.StringIO()

        yield
    finally:
        # Restore original stderr
        os.dup2(original_fd, 2)
        os.close(original_fd)
        sys.stderr = original_stderr


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
    # Add top margin so content isn't at very top of terminal
    print_padding(1)
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
                print_padding(1)  # Top margin
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
        "last_error": None,  # Track last error for retry message
        "parallel_subtasks": [],  # List of subtask descriptions for parallel mode
        "parallel_completed": 0,  # Count of completed subtasks
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
        parallel_subtasks = current_state["parallel_subtasks"]

        # Build display
        line = Text()
        line.append("  ")

        # Handle parallel sub-agent mode
        if status == "parallel" and parallel_subtasks:
            line.append("⚡ ", style="bold blue")
            line.append(
                f"Running {len(parallel_subtasks)} subtasks in parallel", style="blue"
            )
            line.append("\n")
            for i, subtask in enumerate(parallel_subtasks):
                line.append("    ")
                if i < current_state["parallel_completed"]:
                    line.append("├─ ✓ ", style="green")
                    line.append(subtask[:40], style="green")
                else:
                    line.append(f"├─ {spinner} ", style="cyan")
                    line.append(subtask[:40], style="cyan")
                if i < len(parallel_subtasks) - 1:
                    line.append("\n")
            return line

        if status == "retrying":
            line.append(f"{spinner} ", style="bold yellow")
            line.append(
                "Hmm, that didn't work. Let me try another approach...", style="yellow"
            )
        elif status == "thinking":
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
                elif action.startswith("web_search:"):
                    line.append("Searching the web...", style="cyan")
                elif action.startswith("fetch_webpage:"):
                    line.append("Fetching webpage...", style="cyan")
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

        # Handle parallel sub-agent mode
        if status == "parallel":
            current_state["status"] = "parallel"
            # Parse subtask descriptions from action field
            if action:
                subtasks = [s.strip() for s in action.split(",")]
                current_state["parallel_subtasks"] = subtasks
            return

        # Handle completion of parallel subtasks
        if status in ("completed", "partial") and current_state["parallel_subtasks"]:
            current_state["parallel_completed"] = len(
                current_state["parallel_subtasks"]
            )
            return

        # Map status to display status
        if status in ("thinking", "planning"):
            # Check if last step was an error - show retry message
            if current_state["last_error"]:
                current_state["status"] = "retrying"
                current_state["last_error"] = None  # Clear after showing
            else:
                current_state["status"] = "thinking"
        elif status == "error":
            current_state["last_error"] = True
            current_state["status"] = "executing"  # Still show as executing
        else:
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
        inner_width = width - 8  # Account for "  ╭" and "╮" on edges
        console.print()
        console.print(f"  [red]╭{'─' * inner_width}╮[/red]")

        # Header line with proper padding
        header = "⚠ Confirmation Required"
        header_padding = inner_width - len(header) - 2  # -2 for spaces around text
        console.print(
            f"  [red]│[/red] [bold red]{header}[/bold red]"
            f"{' ' * header_padding}[red]│[/red]"
        )
        console.print(f"  [red]│[/red]{' ' * inner_width}[red]│[/red]")

        # Wrap message lines with proper right border
        for line in message.split("\n"):
            # Truncate if too long
            max_line_len = inner_width - 4  # -4 for "  " padding on each side
            if len(line) > max_line_len:
                line = line[: max_line_len - 3] + "..."
            # Pad to align right border
            line_padding = inner_width - len(line) - 4
            console.print(
                f"  [red]│[/red]  [dim]{line}[/dim]{' ' * line_padding}  [red]│[/red]"
            )

        console.print(f"  [red]╰{'─' * inner_width}╯[/red]")

        try:
            result = Confirm.ask("  [bold]Proceed?[/bold]", default=False)
            # Show user's decision clearly
            console.print()
            if result:
                console.print(
                    "  [green]✓[/green] [bold green]Approved[/bold green] - "
                    "Continuing with operation..."
                )
            else:
                console.print(
                    "  [red]✗[/red] [bold red]Denied[/bold red] - "
                    "Operation cancelled by user"
                )
            console.print()
            return result
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print(
                "  [red]✗[/red] [bold red]Cancelled[/bold red] - Operation interrupted"
            )
            console.print()
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

        # Run agent with live display, suppressing stderr to prevent
        # shell warnings from corrupting the spinner display
        with (
            suppress_stderr(),
            Live(build_agent_display(), console=console, refresh_per_second=4) as live,
        ):

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
            console.print()  # Padding after error
            response_text = f"Failed: {state.error}"
        elif state.status == "max_iterations":
            console.print(
                "  [yellow]⚠ Reached max iterations without completing[/yellow]"
            )
            if state.steps:
                # Show what was accomplished
                response_text = _show_agent_result(state, model)
            else:
                console.print()  # Padding if no results to show

        return response_text

    except LLMError as e:
        print_error("AI Error", str(e))
        console.print()  # Padding after error
        return None
    except Exception as e:
        import traceback

        traceback.print_exc()
        print_error("Error", str(e))
        console.print()  # Padding after error
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

    console.print()
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

    # Add trailing padding for visual separation and to keep away from terminal bottom
    print_padding(1)


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
    # Bottom margin to keep content away from terminal edge
    print_padding(1)


def _get_input() -> str:
    """Get user input with complete blue rectangle box.

    Shows the full box (top, sides, bottom) before typing,
    with cursor positioned inside. For long text, the box
    expands to show multiple lines.
    """
    width = _get_width()
    inner_width = width - 8
    max_input_len = inner_width - 5  # Space for "│ > " and " │"

    try:
        # Print initial box with prompt
        console.print(f"  [blue]╭{'─' * inner_width}╮[/blue]")
        console.print(
            f"  [blue]│[/blue] [dim]>[/dim] {' ' * (inner_width - 5)}[blue]│[/blue]"
        )
        console.print(f"  [blue]╰{'─' * inner_width}╯[/blue]")

        # Print empty lines BELOW the box to create scroll buffer
        console.print()
        console.print()

        # Move cursor up 4 lines (2 empty + bottom border + to middle line)
        # Then right to input position (past "  │ > ")
        sys.stdout.write("\033[4A\033[7C")
        sys.stdout.flush()

        # Get input
        user_input = input()

        # After input, redraw the box with the full text properly wrapped
        if len(user_input) > max_input_len:
            # Move back up and clear the old box
            # Go up 1 line (we're on middle), clear from here down
            sys.stdout.write("\033[1A")  # Up to top border
            sys.stdout.write("\033[J")  # Clear from cursor to end of screen
            sys.stdout.flush()

            # Redraw box with wrapped text
            console.print(f"  [blue]╭{'─' * inner_width}╮[/blue]")

            # Split input into lines that fit
            remaining = user_input
            first_line = True
            while remaining:
                chunk = remaining[:max_input_len]
                remaining = remaining[max_input_len:]
                padding = max_input_len - len(chunk)

                if first_line:
                    console.print(
                        f"  [blue]│[/blue] [dim]>[/dim] {chunk}"
                        f"{' ' * padding} [blue]│[/blue]"
                    )
                    first_line = False
                else:
                    console.print(
                        f"  [blue]│[/blue]   {chunk}{' ' * padding} [blue]│[/blue]"
                    )

            console.print(f"  [blue]╰{'─' * inner_width}╯[/blue]")
            console.print()
            console.print()
        else:
            # Short input - just move cursor down past the box
            sys.stdout.write("\033[4B\r")
            sys.stdout.flush()

        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        # Clean up cursor position on interrupt
        sys.stdout.write("\033[4B\r")
        sys.stdout.flush()
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
    print_padding(1)


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
    print_padding(1)


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
    print_padding(1)


def _show_goodbye():
    """Show a clean goodbye message."""
    console.print()
    console.print("  [dim]Session ended. Goodbye![/dim]")
    print_padding(1)
