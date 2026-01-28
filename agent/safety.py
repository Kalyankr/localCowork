"""Dangerous operation detection and confirmation.

This module provides utilities to detect destructive operations
and require user confirmation before execution.
"""

import re
import shlex
from typing import Optional, Tuple, List
from enum import Enum


class DangerLevel(str, Enum):
    """Level of danger for an operation."""
    SAFE = "safe"
    WARNING = "warning"  # Potentially dangerous, warn user
    DANGEROUS = "dangerous"  # Destructive, require confirmation
    BLOCKED = "blocked"  # Never allowed


# Commands that delete or destroy data
DESTRUCTIVE_COMMANDS = {
    # File deletion
    "rm": DangerLevel.DANGEROUS,
    "rmdir": DangerLevel.DANGEROUS,
    "unlink": DangerLevel.DANGEROUS,
    "shred": DangerLevel.DANGEROUS,
    
    # Dangerous file operations
    "dd": DangerLevel.BLOCKED,
    "mkfs": DangerLevel.BLOCKED,
    
    # System commands - always blocked
    "sudo": DangerLevel.BLOCKED,
    "su": DangerLevel.BLOCKED,
    "chmod": DangerLevel.WARNING,
    "chown": DangerLevel.WARNING,
    "kill": DangerLevel.WARNING,
    "pkill": DangerLevel.WARNING,
    "killall": DangerLevel.WARNING,
    "shutdown": DangerLevel.BLOCKED,
    "reboot": DangerLevel.BLOCKED,
    "halt": DangerLevel.BLOCKED,
    "poweroff": DangerLevel.BLOCKED,
    
    # Disk operations - blocked
    "fdisk": DangerLevel.BLOCKED,
    "parted": DangerLevel.BLOCKED,
    "mount": DangerLevel.BLOCKED,
    "umount": DangerLevel.BLOCKED,
    
    # User management - blocked
    "passwd": DangerLevel.BLOCKED,
    "useradd": DangerLevel.BLOCKED,
    "userdel": DangerLevel.BLOCKED,
    "usermod": DangerLevel.BLOCKED,
}

# Dangerous flags that make commands more destructive
DANGEROUS_FLAGS = {
    "rm": ["-r", "-rf", "-fr", "--recursive", "-R"],
    "chmod": ["-R", "--recursive"],
    "chown": ["-R", "--recursive"],
}

# Patterns that indicate dangerous operations
DANGEROUS_PATTERNS = [
    (r"rm\s+.*-[rf]", "Recursive or forced file deletion"),
    (r"rm\s+.*\*", "Wildcard deletion"),
    (r"rm\s+.*~", "Home directory deletion"),
    (r"rm\s+.*/", "Path deletion"),
    (r">\s*/dev/", "Writing to device files"),
    (r">\s*~", "Overwriting home directory files"),
    (r"\|\s*sh", "Piping to shell"),
    (r"\|\s*bash", "Piping to bash"),
    (r"curl.*\|\s*sh", "Downloading and executing scripts"),
    (r"wget.*\|\s*sh", "Downloading and executing scripts"),
]


def analyze_command(command: str) -> Tuple[DangerLevel, Optional[str]]:
    """
    Analyze a shell command for dangerous operations.
    
    Args:
        command: The shell command to analyze
        
    Returns:
        Tuple of (danger_level, reason)
    """
    if not command or not command.strip():
        return DangerLevel.SAFE, None
    
    command = command.strip()
    
    # Try to parse the command
    try:
        parts = shlex.split(command)
    except ValueError:
        # If we can't parse it, be cautious
        return DangerLevel.WARNING, "Could not parse command safely"
    
    if not parts:
        return DangerLevel.SAFE, None
    
    # Get base command (handle paths like /bin/rm)
    base_cmd = parts[0].split("/")[-1]
    
    # Check if base command is in our list
    if base_cmd in DESTRUCTIVE_COMMANDS:
        level = DESTRUCTIVE_COMMANDS[base_cmd]
        
        if level == DangerLevel.BLOCKED:
            return DangerLevel.BLOCKED, f"Command '{base_cmd}' is not allowed"
        
        # Check for dangerous flags
        if base_cmd in DANGEROUS_FLAGS:
            for flag in DANGEROUS_FLAGS[base_cmd]:
                if flag in parts:
                    return DangerLevel.DANGEROUS, f"Command '{base_cmd}' with '{flag}' flag"
        
        return level, f"Command '{base_cmd}' can modify or delete data"
    
    # Check for dangerous patterns in the full command
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return DangerLevel.DANGEROUS, reason
    
    # Check for piped commands
    if "|" in command:
        # Split by pipe and check each part
        pipe_parts = command.split("|")
        for part in pipe_parts:
            level, reason = analyze_command(part.strip())
            if level in (DangerLevel.DANGEROUS, DangerLevel.BLOCKED):
                return level, f"Pipeline contains dangerous command: {reason}"
    
    return DangerLevel.SAFE, None


def analyze_python_code(code: str) -> Tuple[DangerLevel, Optional[str]]:
    """
    Analyze Python code for dangerous operations.
    
    Args:
        code: The Python code to analyze
        
    Returns:
        Tuple of (danger_level, reason)
    """
    if not code:
        return DangerLevel.SAFE, None
    
    # Patterns that indicate file deletion in Python
    dangerous_python_patterns = [
        (r"os\.remove\s*\(", "File deletion with os.remove()"),
        (r"os\.unlink\s*\(", "File deletion with os.unlink()"),
        (r"os\.rmdir\s*\(", "Directory deletion with os.rmdir()"),
        (r"shutil\.rmtree\s*\(", "Recursive directory deletion with shutil.rmtree()"),
        (r"pathlib\.Path.*\.unlink\s*\(", "File deletion with Path.unlink()"),
        (r"\.unlink\s*\(\s*\)", "File deletion with unlink()"),
        (r"pathlib\.Path.*\.rmdir\s*\(", "Directory deletion with Path.rmdir()"),
        (r"\.rmdir\s*\(\s*\)", "Directory deletion with rmdir()"),
        (r"send2trash", "Moving files to trash"),
    ]
    
    for pattern, reason in dangerous_python_patterns:
        if re.search(pattern, code):
            return DangerLevel.DANGEROUS, reason
    
    # Check for subprocess with dangerous commands
    if "subprocess" in code:
        for cmd in DESTRUCTIVE_COMMANDS:
            if cmd in code:
                level = DESTRUCTIVE_COMMANDS[cmd]
                if level in (DangerLevel.DANGEROUS, DangerLevel.BLOCKED):
                    return level, f"Subprocess executing '{cmd}'"
    
    return DangerLevel.SAFE, None


def get_affected_paths(command: str) -> List[str]:
    """
    Extract file/directory paths that would be affected by a command.
    
    Args:
        command: The shell command
        
    Returns:
        List of paths that would be affected
    """
    paths = []
    
    try:
        parts = shlex.split(command)
    except ValueError:
        return paths
    
    if not parts:
        return paths
    
    base_cmd = parts[0].split("/")[-1]
    
    if base_cmd in ("rm", "rmdir", "unlink"):
        # Skip flags, get file arguments
        for part in parts[1:]:
            if not part.startswith("-"):
                paths.append(part)
    
    return paths


def format_confirmation_message(
    command: str,
    danger_level: DangerLevel,
    reason: str,
    affected_paths: Optional[List[str]] = None
) -> str:
    """
    Format a user-friendly confirmation message.
    
    Args:
        command: The command being executed
        danger_level: The danger level
        reason: The reason for the warning
        affected_paths: Optional list of affected paths
        
    Returns:
        Formatted confirmation message
    """
    if danger_level == DangerLevel.BLOCKED:
        return f"🚫 BLOCKED: {reason}\nCommand: {command}"
    
    emoji = "⚠️" if danger_level == DangerLevel.WARNING else "🗑️"
    
    lines = [
        f"{emoji} CONFIRMATION REQUIRED",
        f"",
        f"Reason: {reason}",
        f"Command: {command}",
    ]
    
    if affected_paths:
        lines.append("")
        lines.append("Files/directories that will be affected:")
        for path in affected_paths[:10]:  # Limit to 10 paths
            lines.append(f"  • {path}")
        if len(affected_paths) > 10:
            lines.append(f"  ... and {len(affected_paths) - 10} more")
    
    lines.append("")
    lines.append("Do you want to proceed? (y/N)")
    
    return "\n".join(lines)
