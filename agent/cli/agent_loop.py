"""Pure agentic loop - ReAct-based autonomous agent."""

import asyncio
import shutil
import time
from typing import Optional
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

from agent.cli.console import (
    console,
    print_error,
    print_padding,
    format_duration,
)


# Get terminal width for proper formatting
def _get_width() -> int:
    """Get terminal width, with a reasonable default."""
    return min(shutil.get_terminal_size().columns - 4, 100)


def run_agent(model_override: str = None):
    """Main agent loop - handles everything autonomously."""
    from agent.llm.client import check_ollama_health
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

            result = _process_input_agentic(user_input, model, conversation_history)

            # Add to conversation history
            if result:
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": result})
                # Keep history manageable (last 10 exchanges)
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

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
        "steps": [],  # List of (iteration, action, status, thought_preview)
    }

    # Spinner frames for animation
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    frame_idx = [0]  # Use list to mutate in closure

    def build_agent_display():
        """Build live display showing agent's reasoning and actions."""
        width = _get_width()
        frame_idx[0] = (frame_idx[0] + 1) % len(spinner_frames)
        spinner = spinner_frames[frame_idx[0]]

        lines = []

        # Status bar with iteration count
        iteration = current_state["iteration"]
        status = current_state["status"]

        if status == "thinking":
            status_line = Text()
            status_line.append(f"  {spinner} ", style="bold yellow")
            status_line.append("Thinking", style="bold yellow")
            if iteration > 0:
                status_line.append(f" (step {iteration})", style="dim")
            lines.append(status_line)
        elif status == "executing":
            status_line = Text()
            status_line.append(f"  {spinner} ", style="bold cyan")
            status_line.append("Executing", style="bold cyan")
            if iteration > 0:
                status_line.append(f" (step {iteration})", style="dim")
            lines.append(status_line)

        # Current thought (truncated elegantly)
        if current_state["thought"] and status != "thinking":
            thought_text = current_state["thought"]
            if len(thought_text) > width - 12:
                thought_text = thought_text[: width - 15] + "..."
            thought_line = Text()
            thought_line.append("     💭 ", style="dim")
            thought_line.append(thought_text, style="italic dim")
            lines.append(thought_line)

        # Current action with better formatting
        if current_state["action"]:
            action_text = current_state["action"]
            # Prettify common actions
            if action_text.startswith("shell:"):
                action_text = "⚡ " + action_text[6:].strip()
                action_style = "bold green"
            elif action_text.startswith("python:"):
                action_text = "🐍 " + action_text[7:].strip()
                action_style = "bold blue"
            else:
                action_style = "cyan"

            if len(action_text) > width - 12:
                action_text = action_text[: width - 15] + "..."
            action_line = Text()
            action_line.append("     ", style="dim")
            action_line.append(action_text, style=action_style)
            lines.append(action_line)

        lines.append(Text(""))  # Spacing before steps

        # Previous steps with better visual
        if current_state["steps"]:
            # Show header if we have steps
            if len(current_state["steps"]) > 0:
                lines.append(Text("  ─── Progress ───", style="dim"))
                lines.append(Text(""))

            for step in current_state["steps"][-5:]:
                iter_num, action, step_status, thought_preview = step
                if step_status == "success":
                    icon, color = "✓", "green"
                elif step_status == "error":
                    icon, color = "✗", "red"
                else:
                    icon, color = "○", "dim"

                step_line = Text()
                step_line.append(f"  {icon} ", style=color)
                step_line.append(f"Step {iter_num}: ", style="bold " + color)

                # Shorten action for display
                display_action = action
                if len(display_action) > width - 20:
                    display_action = display_action[: width - 23] + "..."
                step_line.append(display_action, style=color)
                lines.append(step_line)

        lines.append(Text(""))  # Bottom padding

        # Build panel with lines
        content = Text()
        for i, line in enumerate(lines):
            if isinstance(line, Text):
                content.append_text(line)
            else:
                content.append(str(line))
            if i < len(lines) - 1:
                content.append("\n")

        return Panel(
            content,
            border_style="dim",
            box=box.SIMPLE,
            padding=(0, 1),
            width=width,
        )

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

    try:
        agent = ReActAgent(
            tool_registry=tool_registry,
            sandbox=sandbox,
            on_progress=on_progress,
            max_iterations=15,
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

        console.print()
        console.print(f"  [dim]✓ Done in {format_duration(elapsed)}[/dim]")
        console.print()

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

    console.print()
    console.print("  [bold cyan]◆[/bold cyan] [bold white]LocalCowork[/bold white]")
    console.print()

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
    """Show welcome screen - clean and minimal."""
    width = _get_width()

    console.print()
    console.print()
    # Clean logo
    console.print(
        "  [bold cyan]██╗     [/bold cyan][bold white]LocalCowork[/bold white] [dim]v0.3[/dim]"
    )
    console.print(
        "  [bold cyan]██║     [/bold cyan][dim]Pure Agentic AI Assistant[/dim]"
    )
    console.print("  [bold cyan]███████╗[/bold cyan]")
    console.print()

    # Status bar
    console.print(f"  [green]●[/green] [dim]Connected to[/dim] [cyan]{model}[/cyan]")
    console.print()
    console.print("  [bright_black]" + "─" * (width - 6) + "[/bright_black]")
    console.print()
    console.print("  [white]How can I help you today?[/white]")
    console.print("  [dim]Type a request, or /help for commands.[/dim]")
    console.print()


def _get_input() -> str:
    """Get user input with complete box visible before typing."""
    width = _get_width()
    inner = width - 6

    try:
        console.print()
        # Print the complete box first
        console.print(f"  [cyan]┌{'─' * inner}┐[/cyan]")
        console.print("  [cyan]│[/cyan]" + " " * inner + "[cyan]│[/cyan]")
        console.print(f"  [cyan]└{'─' * inner}┘[/cyan]")

        # Move cursor up 2 lines and to the input position
        print("\033[2A\033[5C", end="", flush=True)

        user_input = console.input("[bold magenta]>[/bold magenta] ")

        # Move cursor down to after the box
        print("\033[2B", end="", flush=True)
        return user_input.strip()
    except (KeyboardInterrupt, EOFError):
        print("\033[2B", end="", flush=True)  # Move down on interrupt too
        console.print()
        raise


def _show_help():
    """Show minimal help."""
    console.print()
    console.print("  [bold cyan]◆[/bold cyan] [bold white]Help[/bold white]")
    console.print()

    console.print("  [bold]Examples[/bold]")
    console.print("    [dim]•[/dim] list files in my home directory")
    console.print("    [dim]•[/dim] find all python files larger than 1MB")
    console.print("    [dim]•[/dim] create a backup of my documents folder")
    console.print("    [dim]•[/dim] analyze this CSV and show statistics")
    console.print()

    console.print("  [bold]Commands[/bold]")
    console.print("    [cyan]/clear[/cyan]  [dim]Reset screen and conversation[/dim]")
    console.print("    [cyan]/help[/cyan]   [dim]Show this help[/dim]")
    console.print("    [cyan]/quit[/cyan]   [dim]Exit[/dim]")
    console.print()

    console.print("  [bold]Tips[/bold]")
    console.print("    [dim]• Be specific about what you want[/dim]")
    console.print("    [dim]• Mention file paths when relevant[/dim]")
    console.print("    [dim]• I remember context from this session[/dim]")
    console.print()


def _show_goodbye():
    """Show a clean goodbye message."""
    console.print()
    console.print("  [dim]Session ended. Goodbye![/dim]")
    console.print()
