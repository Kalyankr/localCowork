"""Ask command - interactive question mode with robust features."""

import typer
from typing import Optional
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich import box

from agent.cli.console import console, Icons, print_error, print_warning, print_info


def ask(
    question: Optional[str] = typer.Argument(None, help="Question to ask (optional, enters interactive mode if omitted)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use (overrides default)"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response in real-time"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Enter interactive chat mode"),
):
    """Ask a question directly (quick chat mode).
    
    Examples:
        localcowork ask "What is Python?"
        localcowork ask -i                    # Interactive mode
        localcowork ask -m llama3 "Explain async"
    """
    from agent.llm.client import check_ollama_health, list_models, LLMError
    from agent.config import settings
    
    # Health check first
    if not _check_connection():
        raise typer.Exit(code=1)
    
    # Use provided model or default
    active_model = model or settings.ollama_model
    
    # Verify model exists
    if model:
        available = list_models()
        if model not in available:
            print_error(f"Model '{model}' not found")
            console.print(f"[dim]Available models: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}[/dim]")
            raise typer.Exit(code=1)
    
    # Interactive mode or single question
    if interactive or question is None:
        _interactive_mode(active_model, stream)
    else:
        _single_question(question, active_model, stream)


def _check_connection() -> bool:
    """Check Ollama connection with helpful error."""
    from agent.llm.client import check_ollama_health
    
    with console.status("[cyan]Checking Ollama connection...[/cyan]", spinner="dots"):
        healthy, error = check_ollama_health()
    
    if not healthy:
        print_error("Cannot connect to Ollama", error)
        console.print()
        console.print("[dim]Troubleshooting:[/dim]")
        console.print("  1. Is Ollama running? [cyan]ollama serve[/cyan]")
        console.print("  2. Check URL in config: [cyan]~/.localcowork/config.yaml[/cyan]")
        console.print("  3. Test connection: [cyan]curl http://localhost:11434/api/tags[/cyan]")
        return False
    
    return True


def _single_question(question: str, model: str, stream: bool):
    """Handle a single question."""
    from agent.llm.client import call_llm_chat, call_llm_chat_stream, LLMError
    
    # Validate input
    question = question.strip()
    if not question:
        print_warning("Empty question. Please provide a question.")
        raise typer.Exit(code=1)
    
    if len(question) > 10000:
        print_warning("Question too long (max 10,000 characters)")
        raise typer.Exit(code=1)
    
    console.print()
    console.print(f"[dim]{Icons.ROBOT} {model}[/dim]")
    console.print()
    
    messages = [{"role": "user", "content": question}]
    
    try:
        if stream:
            _stream_response(messages, model)
        else:
            with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                response = call_llm_chat(messages, model=model)
            console.print(Panel(Markdown(response), border_style="cyan", box=box.ROUNDED))
    except LLMError as e:
        print_error("AI Error", str(e))
        raise typer.Exit(code=1)


def _stream_response(messages: list, model: str):
    """Stream response with live display."""
    from agent.llm.client import call_llm_chat_stream
    
    response_text = ""
    console.print(Panel.fit(f"[cyan]{Icons.ROBOT}[/cyan]", box=box.MINIMAL))
    
    for chunk in call_llm_chat_stream(messages, model=model):
        response_text += chunk
        console.print(chunk, end="")
    
    console.print()  # Final newline
    console.print()


def _interactive_mode(model: str, stream: bool):
    """Run interactive chat session - Claude CLI style with nice UI."""
    from agent.llm.client import call_llm_chat, call_llm_chat_stream, list_models, LLMError
    
    # Clear screen and show welcome
    console.clear()
    _show_welcome(model)
    
    history: list[dict] = []
    current_model = model
    
    while True:
        try:
            # Nice input prompt with box
            user_input = _get_styled_input()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                result = _handle_command(user_input, current_model, history, list_models)
                if result == "quit":
                    break
                elif result and result.startswith("model:"):
                    current_model = result.split(":", 1)[1]
                continue
            
            # Add to history
            history.append({"role": "user", "content": user_input})
            
            # Show response area
            console.print()
            console.print(f"  [dim]╭─ {Icons.ROBOT} [/dim][cyan]{current_model}[/cyan]")
            console.print(f"  [dim]│[/dim]")
            
            try:
                if stream:
                    response_text = ""
                    console.print(f"  [dim]│[/dim]  ", end="")
                    line_chars = 0
                    for chunk in call_llm_chat_stream(history, model=current_model):
                        response_text += chunk
                        # Handle newlines in response
                        if "\n" in chunk:
                            parts = chunk.split("\n")
                            for i, part in enumerate(parts):
                                if i > 0:
                                    console.print()
                                    console.print(f"  [dim]│[/dim]  ", end="")
                                    line_chars = 0
                                console.print(part, end="")
                                line_chars += len(part)
                        else:
                            console.print(chunk, end="")
                            line_chars += len(chunk)
                    console.print()
                else:
                    with console.status("  [dim]│[/dim]  [cyan]...[/cyan]", spinner="dots"):
                        response_text = call_llm_chat(history, model=current_model)
                    for line in response_text.split("\n"):
                        console.print(f"  [dim]│[/dim]  {line}")
                
                console.print(f"  [dim]╰─────[/dim]")
                
                # Add response to history
                history.append({"role": "assistant", "content": response_text})
                
            except LLMError as e:
                console.print(f"  [dim]│[/dim]  [red]{Icons.ERROR} {e}[/red]")
                console.print(f"  [dim]╰─────[/dim]")
                history.pop()  # Remove failed user message
            
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[dim]  Ctrl+C pressed. Type /quit to exit.[/dim]\n")
        except EOFError:
            _show_goodbye()
            break


def _show_welcome(model: str):
    """Show welcome screen."""
    console.print()
    console.print(f"  [bold cyan]╭{'─' * 50}╮[/bold cyan]")
    console.print(f"  [bold cyan]│[/bold cyan]  {Icons.ROBOT} [bold]LocalCowork[/bold]                                   [bold cyan]│[/bold cyan]")
    console.print(f"  [bold cyan]│[/bold cyan]  [dim]Your local AI assistant[/dim]                         [bold cyan]│[/bold cyan]")
    console.print(f"  [bold cyan]╰{'─' * 50}╯[/bold cyan]")
    console.print()
    console.print(f"  [dim]Model:[/dim] [cyan]{model}[/cyan]")
    console.print(f"  [dim]Type a question or /help for commands[/dim]")
    console.print()


def _show_goodbye():
    """Show goodbye message."""
    console.print()
    console.print(f"  [dim]Goodbye! 👋[/dim]")
    console.print()


def _get_styled_input() -> str:
    """Get input with styled prompt box."""
    # Draw input box
    console.print(f"  [green]╭─ Ask a question ─────────────────────────────────╮[/green]")
    console.print(f"  [green]│[/green] ", end="")
    
    # Multi-line input support
    lines = []
    try:
        while True:
            if lines:
                console.print(f"  [green]│[/green] ", end="")
            line = input()
            if line.endswith("\\"):
                lines.append(line[:-1])
            else:
                lines.append(line)
                break
    except (KeyboardInterrupt, EOFError):
        console.print()
        console.print(f"  [green]╰{'─' * 51}╯[/green]")
        raise
    
    console.print(f"  [green]╰{'─' * 51}╯[/green]")
    
    return "\n".join(lines).strip()


def _handle_command(user_input: str, current_model: str, history: list, list_models) -> str:
    """Handle slash commands. Returns 'quit', 'model:name', or None."""
    cmd_parts = user_input.split(maxsplit=1)
    cmd = cmd_parts[0].lower()
    args = cmd_parts[1] if len(cmd_parts) > 1 else ""
    
    if cmd in ("/quit", "/q", "/exit"):
        _show_goodbye()
        return "quit"
        
    elif cmd in ("/help", "/h", "/?"):
        _show_help()
        return None
        
    elif cmd in ("/clear", "/c"):
        history.clear()
        console.print()
        console.print(f"  [green]{Icons.SUCCESS}[/green] Chat history cleared")
        console.print()
        return None
        
    elif cmd in ("/model", "/m"):
        if args:
            available = list_models()
            if args in available:
                console.print()
                console.print(f"  [green]{Icons.SUCCESS}[/green] Now using: [cyan]{args}[/cyan]")
                console.print()
                return f"model:{args}"
            else:
                console.print()
                print_warning(f"Model '{args}' not found")
                console.print(f"  [dim]Available: {', '.join(available[:5])}[/dim]")
                console.print()
        else:
            available = list_models()
            console.print()
            console.print(f"  [dim]Current:[/dim] [cyan]{current_model}[/cyan]")
            console.print(f"  [dim]Available:[/dim] {', '.join(available)}")
            console.print()
        return None
    
    elif cmd == "/run":
        if args:
            console.print()
            console.print(f"  [dim]Executing: {args}[/dim]")
            console.print()
            import subprocess
            import sys
            subprocess.run(
                [sys.executable, "-m", "agent.cli", "run", args, "-y"],
                cwd="."
            )
            console.print()
        else:
            console.print()
            console.print("  [dim]Usage: /run <task description>[/dim]")
            console.print("  [dim]Example: /run summarize README.md[/dim]")
            console.print()
        return None
        
    elif cmd == "/history":
        console.print()
        if history:
            console.print(f"  [dim]{len(history)} messages[/dim]")
            for msg in history[-6:]:
                role_icon = "[green]>[/green]" if msg["role"] == "user" else f"[cyan]{Icons.ROBOT}[/cyan]"
                preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                console.print(f"    {role_icon} {preview}")
        else:
            console.print("  [dim]No history yet[/dim]")
        console.print()
        return None
        
    else:
        console.print()
        print_warning(f"Unknown: {cmd}. Type /help for commands")
        console.print()
        return None


def _show_help():
    """Show interactive mode help."""
    console.print()
    console.print(f"  [bold]Commands:[/bold]")
    console.print()
    console.print(f"    [cyan]/run[/cyan] <task>   Execute a task with tools (files, web, code)")
    console.print(f"    [cyan]/model[/cyan] [name] Switch AI model")
    console.print(f"    [cyan]/clear[/cyan]        Clear chat history")
    console.print(f"    [cyan]/history[/cyan]      Show recent messages")
    console.print(f"    [cyan]/help[/cyan]         Show this help")
    console.print(f"    [cyan]/quit[/cyan]         Exit")
    console.print()
    console.print(f"  [bold]Tips:[/bold]")
    console.print(f"    • End a line with \\\\ to continue on next line")
    console.print(f"    • Ctrl+C cancels current generation")
    console.print()
