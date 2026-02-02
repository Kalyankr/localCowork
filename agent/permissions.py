"""
File Permission Management for LocalCowork.

This module provides path validation and access control to ensure
the agent only accesses files/folders the user has explicitly allowed.
"""

import fnmatch
import logging
import re
from enum import Enum
from pathlib import Path

from agent.config import get_settings

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access level for a path."""

    ALLOWED = "allowed"  # Path is explicitly allowed
    DENIED = "denied"  # Path is explicitly denied (blocked)
    NEEDS_CONFIRMATION = "needs_confirmation"  # Path requires user confirmation
    SENSITIVE = "sensitive"  # Path is sensitive (always denied)


class PermissionError(Exception):
    """Raised when access to a path is denied."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Access denied to '{path}': {reason}")


# Sensitive paths that are always blocked regardless of settings
ALWAYS_BLOCKED_PATTERNS = [
    "~/.ssh/*",
    "~/.gnupg/*",
    "~/.aws/*",
    "~/.config/gcloud/*",
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/*.pem",
    "**/.env",
    "**/.env.*",
    "**/secrets.*",
    "**/credentials*",
]


def _expand_path(path: str) -> str:
    """Expand ~ and resolve to absolute path."""
    return str(Path(path).expanduser().resolve())


def _parse_path_list(path_str: str) -> list[str]:
    """Parse comma-separated path list, expanding each path."""
    if not path_str or not path_str.strip():
        return []
    paths = [p.strip() for p in path_str.split(",") if p.strip()]
    return [_expand_path(p) for p in paths]


def _matches_pattern(path: str, pattern: str) -> bool:
    """Check if path matches a glob pattern."""
    # Expand pattern
    expanded_pattern = _expand_path(pattern) if "~" in pattern else pattern

    # Handle ** for recursive matching
    if "**" in expanded_pattern:
        # Convert glob to regex - handle ** before * to avoid conflicts
        # First, escape regex special chars except * and **
        regex_pattern = re.escape(expanded_pattern)
        # Unescape our glob patterns
        regex_pattern = regex_pattern.replace(r"\*\*", "<<DOUBLE_STAR>>")
        regex_pattern = regex_pattern.replace(r"\*", "[^/]*")
        regex_pattern = regex_pattern.replace("<<DOUBLE_STAR>>", ".*")
        regex_pattern = f"^{regex_pattern}$"
        return bool(re.match(regex_pattern, path))

    # Use fnmatch for simple patterns
    return fnmatch.fnmatch(path, expanded_pattern)


def _is_subpath(path: str, parent: str) -> bool:
    """Check if path is under parent directory."""
    try:
        path_resolved = Path(path).resolve()
        parent_resolved = Path(parent).resolve()
        return (
            path_resolved == parent_resolved or parent_resolved in path_resolved.parents
        )
    except (OSError, ValueError):
        return False


def check_path_access(path: str) -> AccessLevel:
    """
    Check access level for a given path.

    Args:
        path: The file or directory path to check

    Returns:
        AccessLevel indicating whether access is allowed, denied, or needs confirmation
    """
    settings = get_settings()
    expanded_path = _expand_path(path)

    # First check: Always blocked sensitive patterns
    for pattern in ALWAYS_BLOCKED_PATTERNS:
        if _matches_pattern(expanded_path, pattern):
            logger.warning(f"Path '{path}' matches sensitive pattern '{pattern}'")
            return AccessLevel.SENSITIVE

    # Second check: User-configured denied paths
    denied_paths = _parse_path_list(settings.denied_paths)
    for denied in denied_paths:
        if _is_subpath(expanded_path, denied) or _matches_pattern(
            expanded_path, denied
        ):
            logger.info(f"Path '{path}' is in denied list")
            return AccessLevel.DENIED

    # Third check: User-configured allowed paths
    allowed_paths = _parse_path_list(settings.allowed_paths)
    if allowed_paths:
        for allowed in allowed_paths:
            if _is_subpath(expanded_path, allowed):
                logger.debug(f"Path '{path}' is under allowed path '{allowed}'")
                return AccessLevel.ALLOWED

        # Path is not under any allowed path
        if settings.require_path_confirmation:
            return AccessLevel.NEEDS_CONFIRMATION
        else:
            return AccessLevel.ALLOWED

    # No allowed_paths configured - allow all (with confirmation for sensitive)
    if settings.require_path_confirmation:
        # Check if it's outside common safe directories
        safe_dirs = [
            _expand_path("~"),
            "/tmp",
            "/var/tmp",
        ]
        for safe_dir in safe_dirs:
            if _is_subpath(expanded_path, safe_dir):
                return AccessLevel.ALLOWED

        return AccessLevel.NEEDS_CONFIRMATION

    return AccessLevel.ALLOWED


def validate_command_paths(command: str) -> tuple[AccessLevel, list[str]]:
    """
    Extract and validate paths from a shell command.

    Args:
        command: Shell command to analyze

    Returns:
        Tuple of (worst access level, list of paths that need attention)
    """
    # Common path patterns in commands
    path_patterns = [
        r"(?:^|\s)(/[^\s;|&><]+)",  # Absolute paths
        r"(?:^|\s)(~/[^\s;|&><]+)",  # Home-relative paths
        r"(?:^|\s)(\./[^\s;|&><]+)",  # Current-relative paths
        r"(?:^|\s)(\.\./[^\s;|&><]+)",  # Parent-relative paths
        r">>\s*([^\s;|&]+)",  # Redirect append
        r">\s*([^\s;|&]+)",  # Redirect output
        r"<\s*([^\s;|&]+)",  # Redirect input
    ]

    paths_found = set()
    for pattern in path_patterns:
        matches = re.findall(pattern, command)
        paths_found.update(matches)

    # Filter out common non-path arguments and special files
    non_paths = {"-", "--", "-r", "-f", "-rf", "-v", "-a", "-l", "-la"}
    # /dev/null is a standard output discard, always safe
    safe_special_paths = {"/dev/null", "/dev/zero", "/dev/urandom", "/dev/random"}
    paths_found = {
        p
        for p in paths_found
        if p not in non_paths and p not in safe_special_paths and len(p) > 1
    }

    if not paths_found:
        return AccessLevel.ALLOWED, []

    worst_level = AccessLevel.ALLOWED
    attention_paths = []

    for path in paths_found:
        level = check_path_access(path)
        if (
            level in (AccessLevel.DENIED, AccessLevel.SENSITIVE)
            or level == AccessLevel.NEEDS_CONFIRMATION
            and worst_level == AccessLevel.ALLOWED
        ):
            worst_level = level
            attention_paths.append(path)

    return worst_level, attention_paths


def get_permission_error_message(path: str, level: AccessLevel) -> str:
    """Get a user-friendly error message for a permission denial."""
    if level == AccessLevel.SENSITIVE:
        return (
            f"Access to '{path}' is blocked for security reasons. "
            "This path contains sensitive data (SSH keys, credentials, etc.)."
        )
    elif level == AccessLevel.DENIED:
        return (
            f"Access to '{path}' is denied. "
            "This path is in your denied_paths configuration."
        )
    else:
        return f"Access to '{path}' requires confirmation."


def format_allowed_paths_info() -> str:
    """Get a formatted string showing current permission settings."""
    settings = get_settings()
    allowed = _parse_path_list(settings.allowed_paths)
    denied = _parse_path_list(settings.denied_paths)

    lines = ["Current permission settings:"]
    if allowed:
        lines.append(f"  Allowed paths: {', '.join(allowed)}")
    else:
        lines.append("  Allowed paths: (all paths with confirmation)")
    if denied:
        lines.append(f"  Denied paths: {', '.join(denied)}")
    lines.append(f"  Require confirmation: {settings.require_path_confirmation}")

    return "\n".join(lines)
