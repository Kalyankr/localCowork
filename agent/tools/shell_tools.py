"""Shell operations: run safe shell commands."""

import subprocess
import shlex
from pathlib import Path
from typing import Optional

# Allowed commands (whitelist for safety)
ALLOWED_COMMANDS = {
    # File inspection
    "ls",
    "find",
    "du",
    "df",
    "wc",
    "head",
    "tail",
    "cat",
    "file",
    "stat",
    # Text processing
    "grep",
    "awk",
    "sed",
    "sort",
    "uniq",
    "cut",
    "tr",
    "diff",
    # System info
    "date",
    "whoami",
    "hostname",
    "uname",
    "uptime",
    "pwd",
    "which",
    "echo",
    # Other safe utilities
    "tree",
    "basename",
    "dirname",
    "realpath",
}

# Explicitly blocked (even if in a pipeline)
BLOCKED_COMMANDS = {
    "rm",
    "rmdir",
    "sudo",
    "su",
    "chmod",
    "chown",
    "kill",
    "pkill",
    "shutdown",
    "reboot",
    "dd",
    "mkfs",
    "mount",
    "umount",
    "fdisk",
    "passwd",
    "useradd",
    "userdel",
    "curl",
    "wget",  # use web_op instead
}


def is_command_safe(cmd: str) -> tuple[bool, str]:
    """
    Check if a command is safe to execute.
    Returns (is_safe, reason).
    """
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return False, f"Invalid command syntax: {e}"

    if not parts:
        return False, "Empty command"

    base_cmd = Path(parts[0]).name  # Handle full paths like /bin/ls

    # Check for blocked commands anywhere in the command
    all_words = set(cmd.split())
    for blocked in BLOCKED_COMMANDS:
        if blocked in all_words:
            return False, f"Command '{blocked}' is not allowed for safety"

    # Check if base command is in whitelist
    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"Command '{base_cmd}' is not in the allowed list"

    # Check for dangerous patterns
    dangerous_patterns = [
        (">", "Output redirection"),
        (">>", "Append redirection"),
        ("$(", "Command substitution"),
        ("`", "Backtick substitution"),
        ("&&", "Command chaining"),
        ("||", "Command chaining"),
        (";", "Command separator"),
    ]

    for pattern, reason in dangerous_patterns:
        if pattern in cmd:
            return False, f"{reason} is not allowed"

    # Validate pipe commands - each command in pipeline must be allowed
    if "|" in cmd:
        is_valid, error = _validate_pipeline(cmd)
        if not is_valid:
            return False, error

    return True, "OK"


def _validate_pipeline(cmd: str) -> tuple[bool, str]:
    """Validate that all commands in a pipeline are allowed.
    
    Args:
        cmd: Command string containing pipes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    pipe_parts = cmd.split("|")
    
    for part in pipe_parts:
        part = part.strip()
        if not part:
            continue
            
        try:
            sub_parts = shlex.split(part)
        except ValueError:
            return False, f"Invalid syntax in pipeline: {part}"
            
        if not sub_parts:
            continue
            
        sub_cmd = Path(sub_parts[0]).name
        
        # Check if piped command is blocked
        for blocked in BLOCKED_COMMANDS:
            if blocked in part.split():
                return False, f"Command '{blocked}' is not allowed in pipeline"
        
        # Check if piped command is allowed
        if sub_cmd not in ALLOWED_COMMANDS:
            return False, f"Command '{sub_cmd}' in pipeline is not in the allowed list"
    
    return True, "OK"


def run_command(
    cmd: str,
    cwd: Optional[str] = None,
    timeout: int = 30,
    check_safety: bool = True,
) -> dict:
    """
    Run a shell command safely.
    Returns dict with 'stdout', 'stderr', 'returncode'.
    """
    if check_safety:
        is_safe, reason = is_command_safe(cmd)
        if not is_safe:
            return {
                "error": reason,
                "allowed_commands": sorted(ALLOWED_COMMANDS),
            }

    work_dir = Path(cwd).expanduser() if cwd else None

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "stdout": result.stdout[:50000],  # Limit output size
            "stderr": result.stderr[:5000],
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def get_system_info() -> dict:
    """Get basic system information."""
    import platform

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }


def dispatch(op: str, **kwargs) -> dict:
    """Dispatch shell operations."""
    if op == "run":
        return run_command(
            kwargs["cmd"],
            kwargs.get("cwd"),
            kwargs.get("timeout", 30),
            kwargs.get("check_safety", True),
        )
    if op == "sysinfo":
        return get_system_info()
    if op == "allowed":
        return {"allowed_commands": sorted(ALLOWED_COMMANDS)}
    raise ValueError(f"Unsupported shell op: {op}")
