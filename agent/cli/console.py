"""Console utilities and theming for LocalCowork CLI."""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Custom theme for consistent styling
THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "dim": "dim",
        "highlight": "bold cyan",
        "muted": "bright_black",
        "accent": "magenta",
        "step.pending": "dim",
        "step.running": "bold yellow",
        "step.success": "green",
        "step.error": "red",
        "step.skipped": "dim",
        "input.border": "bright_black",
        "input.prompt": "bold green",
        "thinking": "bold yellow",
        "executing": "bold cyan",
    }
)

# Global console instance
console = Console(theme=THEME)


# Status icons
class Icons:
    """Consistent icons across the CLI."""

    PENDING = "○"
    RUNNING = "●"
    SUCCESS = "✓"
    ERROR = "✗"
    SKIPPED = "◌"
    STAR = "★"
    ARROW = "→"
    ROBOT = "🤖"
    PLAN = "📋"
    ROCKET = "🚀"
    FOLDER = "📁"
    WARNING = "⚠"


def print_header(title: str, subtitle: str | None = None):
    """Print a styled header."""
    content = f"[bold]{title}[/bold]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(content, box=box.ROUNDED, border_style="cyan", padding=(0, 2)))


def print_success(message: str):
    """Print a success message."""
    console.print(f"[success]{Icons.SUCCESS}[/success] {message}")


def print_error(message: str, detail: str | None = None):
    """Print an error message."""
    console.print(f"[error]{Icons.ERROR}[/error] {message}")
    if detail:
        console.print(f"  [dim]{detail}[/dim]")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[warning]{Icons.WARNING}[/warning] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[info]{Icons.ARROW}[/info] {message}")


def print_padding(lines: int = 2):
    """Add vertical padding at bottom of terminal."""
    for _ in range(lines):
        console.print()


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"


def create_input_panel(prompt: str = "❯", placeholder: str = "") -> Panel:
    """Create a styled input panel."""
    return Panel(
        Text(placeholder, style="dim") if placeholder else Text(""),
        title=f"[input.prompt]{prompt}[/input.prompt]",
        title_align="left",
        border_style="input.border",
        padding=(0, 1),
        box=box.ROUNDED,
    )


def create_status_table(show_header: bool = True) -> Table:
    """Create a table for displaying step status."""
    table = Table(
        box=box.SIMPLE,
        show_header=show_header,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("Status", width=10, justify="center")
    table.add_column("Step", style="cyan", width=20)
    table.add_column("Description", style="dim", max_width=40)
    return table


def format_status(status: str) -> str:
    """Format a status with icon and color."""
    styles = {
        "pending": (Icons.PENDING, "step.pending"),
        "starting": (Icons.RUNNING, "step.running"),
        "running": (Icons.RUNNING, "step.running"),
        "thinking": (Icons.RUNNING, "step.running"),
        "success": (Icons.SUCCESS, "step.success"),
        "done": (Icons.SUCCESS, "step.success"),
        "completed": (Icons.STAR, "step.success"),
        "error": (Icons.ERROR, "step.error"),
        "failed": (Icons.ERROR, "step.error"),
        "skipped": (Icons.SKIPPED, "step.skipped"),
    }
    icon, style = styles.get(status, (Icons.PENDING, "dim"))
    return f"[{style}]{icon} {status}[/{style}]"


def friendly_error(error: str) -> tuple[str, str]:
    """Convert raw Python errors to user-friendly messages.

    Returns: (friendly_message, technical_detail)
    """
    error_lower = error.lower()

    # Error mappings
    mappings = [
        # File/path errors
        (["filenotfounderror", "no such file"], "File not found"),
        (["permissionerror", "permission denied"], "Permission denied"),
        (["isadirectoryerror"], "Expected file, got directory"),
        # Network errors
        (["connectionerror", "connection refused"], "Connection failed"),
        (["timeouterror", "timed out"], "Request timed out"),
        # Docker/sandbox errors
        (["docker"], "Docker error"),
        (["container"], "Sandbox error"),
        # JSON/parsing errors
        (["jsondecodeerror", "json"], "Invalid data format"),
        # Python runtime errors
        (["nameerror"], "Code error - undefined variable"),
        (["typeerror"], "Type mismatch"),
        (["valueerror"], "Invalid value"),
        (["keyerror"], "Missing key"),
        (["indexerror"], "Index out of range"),
        (["attributeerror"], "Missing attribute"),
        (["importerror", "modulenotfounderror"], "Missing dependency"),
        (["zerodivisionerror"], "Division by zero"),
        (["memoryerror"], "Out of memory"),
        # Dependency errors
        (["dependency failed"], "Skipped - dependency failed"),
        # LLM errors
        (["ollama", "llm", "cannot connect"], "AI service error"),
    ]

    for keywords, friendly_msg in mappings:
        if any(kw in error_lower for kw in keywords):
            detail = error.split(":")[-1].strip() if ":" in error else error
            return friendly_msg, detail[:80]

    # Generic fallback
    if len(error) > 80:
        if ":" in error:
            return "Error", error.split(":")[-1].strip()[:60]
        return "Error", error[:60] + "…"

    return "Error", error
