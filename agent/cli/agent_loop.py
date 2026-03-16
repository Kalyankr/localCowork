"""Pure agentic loop - ReAct-based autonomous agent."""

import asyncio
import contextlib
import io
import os
import re as _re
import readline  # noqa: F401 — enables arrow-key history in input()
import shutil
import sys
import time

from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.cli.console import (
    console,
    format_duration,
    print_error,
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
    from agent.llm.client import check_model_exists, check_ollama_health

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

    # Verify model is pulled
    with console.status(f"[cyan]Checking model {model}...[/cyan]", spinner="dots"):
        model_ok = check_model_exists(model)

    if not model_ok:
        print_error(
            f"Model '{model}' not found",
            "The model is not pulled in Ollama.",
        )
        console.print(f"\n[dim]Pull it with: [cyan]ollama pull {model}[/cyan][/dim]")
        raise SystemExit(1)

    # Always interactive mode
    _interactive_loop(model)


def _interactive_loop(model: str):
    """Interactive agent loop with conversation memory."""
    # Clear screen and move cursor to top without leaving whitespace
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()
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
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
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
                continue

            # Catch unknown slash commands
            if user_input.startswith("/"):
                known = [
                    "/quit",
                    "/q",
                    "/exit",
                    "/help",
                    "/h",
                    "/clear",
                    "/status",
                    "/history",
                    "/model",
                ]
                cmd = user_input.split()[0].lower()
                if cmd not in known:
                    console.print(
                        f"  [yellow]⚠[/yellow] Unknown command: "
                        f"[white]{cmd}[/white] — type [cyan]/help[/cyan] for commands"
                    )
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

            # Visual separator between conversations
            width = _get_width()
            console.print("  [bright_black]" + "─" * (width - 6) + "[/bright_black]")

        except KeyboardInterrupt:
            console.print()
            console.print(
                "  [dim]Interrupted. Type [white]/quit[/white] to exit.[/dim]"
            )
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
    display_start_time = time.time()

    def build_agent_display():
        """Build spinner display showing current activity."""
        frame_idx[0] = (frame_idx[0] + 1) % len(spinner_frames)
        spinner = spinner_frames[frame_idx[0]]
        elapsed = format_duration(time.time() - display_start_time)

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
            # Calculate max subtask width (terminal width - indent - prefix - margin)
            term_width = _get_width()
            max_subtask_width = max(term_width - 15, 40)
            for i, subtask in enumerate(parallel_subtasks):
                line.append("    ")
                # Truncate with ellipsis if too long
                display_subtask = (
                    subtask
                    if len(subtask) <= max_subtask_width
                    else subtask[: max_subtask_width - 3] + "..."
                )
                if i < current_state["parallel_completed"]:
                    line.append("├─ ✓ ", style="green")
                    line.append(display_subtask, style="green")
                else:
                    line.append(f"├─ {spinner} ", style="cyan")
                    line.append(display_subtask, style="cyan")
                if i < len(parallel_subtasks) - 1:
                    line.append("\n")
            return line

        if status == "retrying":
            line.append(f"{spinner} ", style="bold yellow")
            line.append(
                "Hmm, that didn't work. Let me try another approach...", style="yellow"
            )
            line.append(f"  [{elapsed}]", style="dim")
        elif status == "steering":
            line.append("↪ ", style="bold magenta")
            thought = current_state.get("thought", "")
            if thought:
                line.append(f"Adjusting: {thought[:50]}", style="magenta")
            else:
                line.append("Received your update, adjusting...", style="magenta")
            line.append(f"  [{elapsed}]", style="dim")
        elif status == "thinking":
            line.append(f"{spinner} ", style="bold yellow")
            line.append("Thinking...", style="yellow")
            line.append(f"  [{elapsed}]", style="dim")
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
            line.append(f"  [{elapsed}]", style="dim")

        # Show step count and progress dots
        if iteration > 0 or steps:
            line.append("  ", style="dim")
            if steps:
                for step in steps[-6:]:
                    _, _, step_status, _ = step
                    if step_status == "success":
                        line.append("●", style="green")
                    elif step_status == "error":
                        line.append("●", style="red")
                    else:
                        line.append("○", style="dim")
                line.append(f" Step {len(steps)}", style="dim")

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

        # Handle steering status
        if status == "steering":
            current_state["status"] = "steering"
            current_state["thought"] = thought
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

        panel = Panel(
            f"[dim]{message}[/dim]",
            title="[bold red]⚠ Confirmation Required[/bold red]",
            border_style="red",
            box=box.HEAVY,
            padding=(1, 2),
            width=min(_get_width(), 60),
        )
        console.print(panel)

        try:
            result = Confirm.ask("  [bold]Proceed?[/bold]", default=False)
            if result:
                console.print("  [green]✓[/green] Approved — continuing")
            else:
                console.print("  [red]✗[/red] Denied — operation cancelled")
            console.print()
            return result
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print("  [red]✗[/red] Cancelled — operation interrupted")
            console.print()
            return False

    try:
        # Create steering queue for mid-task corrections
        steering_queue = asyncio.Queue()

        agent = ReActAgent(
            sandbox=sandbox,
            on_progress=on_progress,
            on_confirm=on_confirm,
            max_iterations=settings.max_agent_iterations,
            conversation_history=conversation_history or [],
            steering_queue=steering_queue,
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

                async def steering_listener():
                    """Listen for user steering input while agent runs."""
                    import select
                    import sys

                    while True:
                        await asyncio.sleep(0.1)
                        # Check if stdin has data (non-blocking)
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            try:
                                line = sys.stdin.readline().strip()
                                if line:
                                    # User typed something - add to steering queue
                                    await steering_queue.put(line)
                                    # Update display to show steering received
                                    current_state["status"] = "steering"
                                    current_state["thought"] = f"User: {line[:40]}..."
                            except Exception:
                                pass

                update_task = asyncio.create_task(updater())
                steering_task = asyncio.create_task(steering_listener())
                try:
                    return await agent.run(user_input)
                finally:
                    update_task.cancel()
                    steering_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await update_task
                    with contextlib.suppress(asyncio.CancelledError):
                        await steering_task

            state = asyncio.run(run_with_display())
            live.update(build_agent_display())

        elapsed = time.time() - start_time

        # Show steps if enabled
        if settings.show_steps and state.steps:
            _show_execution_steps(state.steps)

        # Get the response for history
        response_text = None

        # Show final result with appropriate status indicator
        if state.status == "completed":
            console.print(f"  [dim]✓ Done in {format_duration(elapsed)}[/dim]")
            response_text = _show_agent_result(state, model)
        elif state.status == "failed":
            console.print(f"  [red]✗ Failed after {format_duration(elapsed)}[/red]")
            error_msg = state.error or "Unknown error"
            _show_status_box(
                "red",
                "Something went wrong",
                [error_msg, "", "Please try again or rephrase your request"],
            )
            response_text = f"Failed: {state.error}"
        elif state.status == "max_iterations":
            console.print(
                f"  [yellow]⚠ Stopped after {format_duration(elapsed)}[/yellow]"
            )
            _show_status_box(
                "yellow",
                "Could not complete the task",
                [
                    "Reached maximum attempts without success.",
                    "Try breaking down your request into",
                    "smaller, more specific tasks.",
                ],
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


def _show_status_box(color: str, title: str, lines: list[str]):
    """Display a status box using Rich Panel for clean layout."""
    body = "\n".join(line for line in lines if line)
    panel = Panel(
        f"[dim]{body}[/dim]",
        title=f"[bold {color}]{title}[/bold {color}]",
        border_style=color,
        box=box.ROUNDED,
        padding=(1, 2),
        width=min(_get_width(), 60),
    )
    console.print(panel)


def _show_execution_steps(steps: list):
    """Display detailed execution steps after task completion."""
    steps_tbl = Table(box=None, show_header=False, padding=(0, 1))
    steps_tbl.add_column("icon", width=3)
    steps_tbl.add_column("num", style="dim", width=8)
    steps_tbl.add_column("detail")

    for i, step in enumerate(steps, 1):
        thought = step.thought
        action = step.action

        # Determine status icon
        if step.result and step.result.status == "success":
            icon = "[green]✓[/green]"
        elif step.result and step.result.status == "error":
            icon = "[red]✗[/red]"
        else:
            icon = "[dim]○[/dim]"

        # Format thought (truncate if too long)
        thought_preview = (
            thought.reasoning[:60] if thought and thought.reasoning else ""
        )
        if thought and thought.reasoning and len(thought.reasoning) > 60:
            thought_preview += "..."

        # Format action
        action_str = ""
        if action:
            if action.tool == "shell":
                cmd = action.args.get("command", "")[:40]
                if len(action.args.get("command", "")) > 40:
                    cmd += "..."
                action_str = f"[cyan]$ {cmd}[/cyan]"
            elif action.tool == "python":
                code_preview = action.args.get("code", "")[:30].replace("\n", " ")
                if len(action.args.get("code", "")) > 30:
                    code_preview += "..."
                action_str = f"[yellow]▸ {code_preview}[/yellow]"
            elif action.tool == "web_search":
                query = action.args.get("query", "")[:30]
                action_str = f"[blue]🔍 {query}[/blue]"
            elif action.tool == "done":
                action_str = "[green]✓ Done[/green]"
            else:
                action_str = f"[dim]{action.tool}[/dim]"

        detail = thought_preview
        if action_str:
            detail += f"\n       {action_str}"
        steps_tbl.add_row(icon, f"Step {i}", detail)

    panel = Panel(
        steps_tbl,
        title="[bold white]Steps[/bold white]",
        border_style="bright_black",
        box=box.ROUNDED,
        padding=(1, 2),
        width=min(_get_width(), 72),
    )
    console.print(panel)


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
    """Display agent response with clean formatting and markdown support."""
    from rich.markdown import Markdown
    from rich.padding import Padding

    console.print()
    console.print("  [cyan]▍[/cyan] [bold]LocalCowork[/bold]")
    console.print()

    # Render as Markdown if the text contains markdown indicators
    has_markdown = any(
        marker in text for marker in ["```", "**", "##", "- ", "* ", "1. ", "> ", "| "]
    )

    if has_markdown:
        md = Markdown(text)
        console.print(Padding(md, (0, 4)))
    else:
        width = _get_width()
        for line in text.split("\n"):
            if len(line) > width - 8:
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

    # Detect and display image file paths mentioned in the response
    _show_images_in_response(text)

    console.print()


_IMAGE_PATH_RE = _re.compile(
    r"(?:^|\s)((?:/|\.{1,2}/)?[\w./_ -]+\.(?:png|jpg|jpeg|gif|webp|svg|bmp))\b",
    _re.IGNORECASE,
)


def _show_images_in_response(text: str):
    """Detect image file paths in the response and show clickable links."""
    seen = set()
    for match in _IMAGE_PATH_RE.finditer(text):
        path = match.group(1).strip()
        abs_path = os.path.abspath(path)
        if abs_path in seen or not os.path.isfile(abs_path):
            continue
        seen.add(abs_path)
        file_uri = f"file://{abs_path}"
        console.print(
            f"    [bold green]🖼  Image:[/bold green] [link={file_uri}]{path}[/link] "
            f"[dim](click to open)[/dim]"
        )


def _show_welcome(model: str):
    """Show welcome screen — clean, minimal, professional."""
    width = _get_width()

    # Build the welcome content
    content = Text()
    content.append("Your local AI coding assistant\n\n", style="dim")
    content.append("Model  ", style="dim")
    content.append(f"{model}\n", style="cyan")
    content.append("Ready  ", style="dim")
    content.append("●", style="green")
    content.append(" Connected\n\n", style="dim")
    content.append("Try: ", style="dim")
    content.append("list files", style="white")
    content.append(" · ", style="bright_black")
    content.append("find large files", style="white")
    content.append(" · ", style="bright_black")
    content.append("explain this error\n", style="white")
    content.append("Cmds: ", style="dim")
    content.append("/help", style="cyan")
    content.append("  ", style="dim")
    content.append("/status", style="cyan")
    content.append("  ", style="dim")
    content.append("/clear", style="cyan")
    content.append("  ", style="dim")
    content.append("/model", style="cyan")
    content.append("  ", style="dim")
    content.append("/quit", style="cyan")

    panel = Panel(
        content,
        title="[bold white]LocalCowork[/bold white]",
        subtitle=f"[dim]v{__version__}[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        width=min(width, 64),
    )
    console.print(panel)


def _get_input() -> str:
    """Get user input with a blue rectangle box.

    Uses readline-aware prompt so backspace/arrow keys respect boundaries.
    After input, redraws the complete box with the entered text.
    """
    width = _get_width()
    inner_width = width - 8
    max_input_len = inner_width - 5  # Space for "│ > " and " │"

    try:
        # Draw full box frame: top, empty input line, bottom
        console.print(f"  [blue]╭{'─' * inner_width}╮[/blue]")
        console.print(
            f"  [blue]│[/blue] [dim]>[/dim] {' ' * (inner_width - 5)}[blue]│[/blue]"
        )
        console.print(f"  [blue]╰{'─' * inner_width}╯[/blue]")

        # Move cursor up 2 lines (to the input row) and position after "> "
        sys.stdout.write(
            "\033[2A"
        )  # Up 2 lines (bottom border + input row → input row)
        sys.stdout.write("\033[6G")  # Column 6: past "  │ > "
        sys.stdout.flush()

        # Build a readline-safe prompt with zero visible width
        # \001/\002 delimit non-printing sequences so readline knows
        # the cursor hasn't moved (we already positioned it above)
        prompt = "\001\002"

        user_input = input(prompt)

        # Move up to erase the full 3-line box + input, then redraw cleanly
        sys.stdout.write("\033[3A")  # Up 3 lines (top border, input, bottom)
        sys.stdout.write("\033[J")  # Clear from cursor to end of screen
        sys.stdout.flush()

        # Redraw complete box with entered text
        console.print(f"  [blue]╭{'─' * inner_width}╮[/blue]")

        if not user_input.strip():
            # Empty input — show empty box
            console.print(
                f"  [blue]│[/blue] [dim]>[/dim] {' ' * (inner_width - 5)}[blue]│[/blue]"
            )
        else:
            # Render text, wrapping if needed
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

        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        raise


def _show_help():
    """Show compact help using Rich panels and tables."""
    # Examples table
    examples = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    examples.add_column("prompt", style="white", ratio=3)
    examples.add_column("desc", style="dim", ratio=2)
    examples.add_row("list files in home", "Run shell commands")
    examples.add_row("find python files > 1MB", "Complex file searches")
    examples.add_row("analyze this CSV", "Data analysis")
    examples.add_row("search web for FastAPI tips", "Web search")

    # Commands table
    cmds = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    cmds.add_column("cmd", style="cyan", width=12)
    cmds.add_column("desc", style="dim")
    cmds.add_row("/clear", "Reset conversation")
    cmds.add_row("/status", "Connection & settings")
    cmds.add_row("/history", "Conversation history")
    cmds.add_row("/model X", "Switch to model X")
    cmds.add_row("/quit", "Exit LocalCowork")

    content = Text()
    content.append("Examples\n", style="bold white")

    help_group = Table.grid(padding=(0, 0))
    help_group.add_row(Text("Examples", style="bold white"))
    help_group.add_row(examples)
    help_group.add_row(Text(""))
    help_group.add_row(Text("Commands", style="bold white"))
    help_group.add_row(cmds)
    help_group.add_row(Text(""))
    help_group.add_row(
        Text("↑/↓ arrow keys cycle through previous inputs", style="dim")
    )

    panel = Panel(
        help_group,
        title="[bold white]Quick Guide[/bold white]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        width=min(_get_width(), 60),
    )
    console.print(panel)


def _show_status(model: str):
    """Show current status and settings with live health check."""
    import os

    from agent.llm.client import check_ollama_health

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

    # Live connection check
    healthy, error = check_ollama_health()
    if healthy:
        console.print("    [dim]Connection:[/dim]  [green]● Connected[/green]")
    else:
        console.print(
            f"    [dim]Connection:[/dim]  [red]● Disconnected[/red] [dim]({error})[/dim]"
        )

    console.print()


def _show_history(conversation_history: list):
    """Show conversation history with numbered exchanges."""
    if not conversation_history:
        panel = Panel(
            "[dim]No conversation history yet.[/dim]",
            title="[bold white]History[/bold white]",
            border_style="bright_black",
            box=box.ROUNDED,
            padding=(1, 2),
            width=min(_get_width(), 64),
        )
        console.print(panel)
        return

    tbl = Table(box=None, show_header=True, padding=(0, 1), expand=True)
    tbl.add_column("#", style="dim", width=3, justify="right")
    tbl.add_column("Role", width=5)
    tbl.add_column("Message")

    max_len = min(_get_width(), 64) - 22
    exchange_num = 0

    for msg in conversation_history:
        role = msg["role"]
        content = msg["content"].replace("\n", " ")
        if len(content) > max_len:
            content = content[: max_len - 1] + "…"

        if role == "user":
            exchange_num += 1
            tbl.add_row(str(exchange_num), "[green]You[/green]", content)
        else:
            tbl.add_row("", "[cyan]AI[/cyan]", f"[dim]{content}[/dim]")

    panel = Panel(
        tbl,
        title="[bold white]History[/bold white]",
        subtitle=f"[dim]{exchange_num} exchange{'s' if exchange_num != 1 else ''}[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        width=min(_get_width(), 64),
    )
    console.print(panel)


def _show_goodbye():
    """Show a clean goodbye message."""
    console.print()
    console.print("  [dim]Session ended — see you next time.[/dim]")
    console.print()
