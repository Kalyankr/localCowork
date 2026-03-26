"""LocalCowork doctor — diagnose setup issues in one command."""

from __future__ import annotations

import platform
import shutil
import sqlite3
import sys
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table

from agent.cli.console import console
from agent.config import settings
from agent.version import __version__


def _check_python() -> tuple[bool, str]:
    """Check Python version >= 3.12."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = (v.major, v.minor) >= (3, 12)
    return ok, version_str


def _check_ollama() -> tuple[bool, str]:
    """Check Ollama is reachable."""
    try:
        from agent.llm.client import check_ollama_health

        healthy, error = check_ollama_health()
        if healthy:
            return True, "Connected"
        # Strip verbose error prefixes
        msg = error or "Unreachable"
        for prefix in ("Unknown error: ", "Connection error: "):
            if msg.startswith(prefix):
                msg = msg[len(prefix) :]
        return False, msg[:100]
    except Exception as e:
        return False, str(e)[:80]


def _check_model() -> tuple[bool, str]:
    """Check configured model is pulled."""
    model = settings.ollama_model
    try:
        from agent.llm.client import check_model_exists, list_models

        if check_model_exists(model):
            return True, model
        available = list_models()
        if available:
            hint = f"not found (available: {', '.join(available[:5])})"
        else:
            hint = "not found (no models pulled)"
        return False, f"{model} — {hint}"
    except Exception:
        return False, f"{model} — cannot check (Ollama down?)"


def _check_docker() -> tuple[bool | None, str]:
    """Check Docker availability (only relevant if use_docker=True).

    Returns None as the bool if Docker is not required.
    """
    if not settings.use_docker:
        return None, "Not required (use_docker=False)"

    docker_path = shutil.which("docker")
    if not docker_path:
        return False, "Not found in PATH"

    import subprocess

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, "Running"
        stderr = result.stderr.decode(errors="replace").strip()
        return False, stderr[:80] if stderr else "Not running"
    except FileNotFoundError:
        return False, "Binary not found"
    except subprocess.TimeoutExpired:
        return False, "Timed out connecting"
    except Exception as e:
        return False, str(e)[:80]


def _check_database() -> tuple[bool, str]:
    """Check database is writable."""
    db_path = settings.db_path
    try:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        # Quick write test
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _doctor_check (id INTEGER PRIMARY KEY)"
        )
        conn.execute("DROP TABLE IF EXISTS _doctor_check")
        conn.commit()

        # Count memories
        try:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            mem_count = row[0] if row else 0
        except sqlite3.OperationalError:
            mem_count = 0

        # Count tasks
        try:
            row = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
            task_count = row[0] if row else 0
        except sqlite3.OperationalError:
            task_count = 0

        conn.close()
        return True, f"OK ({task_count} tasks, {mem_count} memories)"
    except Exception as e:
        return False, str(e)[:80]


def _check_disk_space() -> tuple[bool, str]:
    """Check disk space on the home directory."""
    try:
        home = Path.home()
        usage = shutil.disk_usage(home)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        pct_free = (usage.free / usage.total) * 100
        ok = free_gb >= 1.0  # Warn if less than 1 GB free
        return ok, f"{free_gb:.1f} GB free / {total_gb:.1f} GB ({pct_free:.0f}%)"
    except Exception as e:
        return False, str(e)[:80]


def _check_workspace() -> tuple[bool, str]:
    """Check workspace directory is writable."""
    ws_path = settings.workspace_path
    try:
        path = Path(ws_path)
        path.mkdir(parents=True, exist_ok=True)
        # Quick write test
        test_file = path / ".doctor_check"
        test_file.write_text("ok")
        test_file.unlink()
        return True, ws_path
    except Exception as e:
        return False, str(e)[:80]


def run_doctor() -> int:
    """Run all diagnostic checks and display results.

    Returns 0 if all critical checks pass, 1 otherwise.
    """
    import logging

    # Suppress library warnings during diagnostics — we report status visually
    logging.getLogger("agent.llm").setLevel(logging.CRITICAL)

    console.print()
    console.print(f"  [bold cyan]LocalCowork Doctor[/bold cyan] v{__version__}")
    console.print(
        f"  [dim]Python {platform.python_version()} on "
        f"{platform.system()} {platform.release()}[/dim]"
    )
    console.print()

    # Run checks
    checks: list[tuple[str, bool | None, str]] = []

    ok, detail = _check_python()
    checks.append(("Python >= 3.12", ok, detail))

    ok, detail = _check_ollama()
    checks.append(("Ollama", ok, detail))

    ok, detail = _check_model()
    checks.append(("Model", ok, detail))

    ok_or_none, detail = _check_docker()
    checks.append(("Docker", ok_or_none, detail))

    ok, detail = _check_database()
    checks.append(("Database", ok, detail))

    ok, detail = _check_workspace()
    checks.append(("Workspace", ok, detail))

    ok, detail = _check_disk_space()
    checks.append(("Disk space", ok, detail))

    # Build table
    table = Table(box=box.SIMPLE, padding=(0, 2), show_header=True)
    table.add_column("Check", style="white", min_width=16)
    table.add_column("Status", min_width=6)
    table.add_column("Details", style="dim")

    failures = 0
    for name, passed, detail in checks:
        if passed is True:
            icon = "[green]✓ pass[/green]"
        elif passed is None:
            icon = "[dim]— skip[/dim]"
        else:
            icon = "[red]✗ FAIL[/red]"
            failures += 1
        table.add_row(name, icon, detail)

    console.print(
        Panel(table, title="[bold]Diagnostics[/bold]", border_style="bright_black")
    )

    # Config summary
    cfg_table = Table(box=box.SIMPLE, padding=(0, 2), show_header=False)
    cfg_table.add_column("Key", style="cyan", min_width=20)
    cfg_table.add_column("Value", style="white")
    cfg_table.add_row("Model", settings.ollama_model)
    cfg_table.add_row("Ollama URL", settings.ollama_url)
    cfg_table.add_row("Safety profile", settings.safety_profile)
    cfg_table.add_row("Max iterations", str(settings.max_agent_iterations))
    cfg_table.add_row("Docker sandbox", str(settings.use_docker))
    cfg_table.add_row("DB path", settings.db_path)
    cfg_table.add_row("Workspace", settings.workspace_path)

    console.print(
        Panel(
            cfg_table,
            title="[bold]Configuration[/bold]",
            border_style="bright_black",
        )
    )

    # Remediation hints
    if failures > 0:
        console.print()
        for name, passed, _detail in checks:
            if passed is False:
                if name == "Ollama":
                    console.print(
                        "  [yellow]→[/yellow] Start Ollama: [cyan]ollama serve[/cyan]"
                    )
                elif name == "Model":
                    console.print(
                        f"  [yellow]→[/yellow] Pull model: "
                        f"[cyan]ollama pull {settings.ollama_model}[/cyan]"
                    )
                elif name == "Docker":
                    console.print(
                        "  [yellow]→[/yellow] Start Docker: "
                        "[cyan]sudo systemctl start docker[/cyan]"
                    )
                elif name == "Disk space":
                    console.print(
                        "  [yellow]→[/yellow] Free up disk space (< 1 GB remaining)"
                    )
        console.print()
        console.print(f"  [red]✗[/red] {failures} check(s) failed")
    else:
        console.print()
        console.print("  [green]✓[/green] All checks passed — ready to go!")

    console.print()
    return 1 if failures > 0 else 0
