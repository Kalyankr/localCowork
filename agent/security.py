"""Security utilities for input validation and path sanitization.

This module provides security functions to prevent:
- Path traversal attacks (../../etc/passwd)
- Symlink attacks
- Command injection
- Invalid input data
"""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Exception raised for security violations."""

    pass


class PathTraversalError(SecurityError):
    """Exception raised when path traversal is detected."""

    pass


class InputValidationError(SecurityError):
    """Exception raised for invalid input."""

    pass


# Configurable allowed base directories (expand as needed)
# Empty means all paths are allowed (for development)
ALLOWED_BASE_DIRS: list[Path] = []

# Dangerous path patterns
DANGEROUS_PATTERNS = [
    r"\.\.",  # Parent directory traversal
    r"^/",  # Absolute path starting with /
    r"^[a-zA-Z]:",  # Windows absolute path
    r"\x00",  # Null byte injection
]

# Sensitive paths that should never be accessed
SENSITIVE_PATHS = [
    "/etc/passwd",
    "/etc/shadow",
    "/etc/hosts",
    "/etc/ssh",
    "/root",
    "/proc",
    "/sys",
    "/dev",
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.config",
    "/var/log",
    "/boot",
]

# Maximum path length
MAX_PATH_LENGTH = 4096

# Maximum filename length
MAX_FILENAME_LENGTH = 255


def is_path_safe(path: str | Path, base_dir: Path | None = None) -> bool:
    """
    Check if a path is safe (no traversal, not sensitive).

    Args:
        path: Path to check
        base_dir: Optional base directory the path must be within

    Returns:
        True if path is safe, False otherwise
    """
    try:
        validate_path(path, base_dir)
        return True
    except SecurityError:
        return False


def validate_path(
    path: str | Path,
    base_dir: Path | None = None,
    must_exist: bool = False,
    allow_symlinks: bool = False,
) -> Path:
    """
    Validate and sanitize a file path.

    Args:
        path: Path to validate
        base_dir: If provided, path must resolve within this directory
        must_exist: If True, path must exist
        allow_symlinks: If False, reject symlinks (prevent symlink attacks)

    Returns:
        Resolved, validated Path object

    Raises:
        PathTraversalError: If path traversal is detected
        InputValidationError: If path is otherwise invalid
    """
    if not path:
        raise InputValidationError("Path cannot be empty")

    path_str = str(path)

    # Check length
    if len(path_str) > MAX_PATH_LENGTH:
        raise InputValidationError(
            f"Path too long: {len(path_str)} > {MAX_PATH_LENGTH}"
        )

    # Check for null bytes (injection attack)
    if "\x00" in path_str:
        raise PathTraversalError("Null byte detected in path")

    # Expand user home directory
    expanded = Path(path_str).expanduser()

    # Resolve to absolute path (this resolves .. and symlinks)
    try:
        resolved = expanded.resolve()
    except (OSError, RuntimeError) as e:
        raise InputValidationError(f"Cannot resolve path: {e}")

    # Check against sensitive paths
    resolved_str = str(resolved)
    for sensitive in SENSITIVE_PATHS:
        sensitive_expanded = str(Path(sensitive).expanduser())
        if resolved_str.startswith(sensitive_expanded):
            logger.warning(f"Blocked access to sensitive path: {resolved}")
            raise PathTraversalError("Access denied: sensitive path")

    # Check if path is within allowed base directory
    if base_dir:
        base_resolved = base_dir.expanduser().resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            logger.warning(f"Path traversal attempt: {path} escapes {base_dir}")
            raise PathTraversalError(f"Path escapes base directory: {path}")

    # Check global allowed directories if configured
    if ALLOWED_BASE_DIRS:
        allowed = False
        for allowed_base in ALLOWED_BASE_DIRS:
            try:
                resolved.relative_to(allowed_base.expanduser().resolve())
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise PathTraversalError(f"Path not in allowed directories: {path}")

    # Check for symlinks if not allowed
    if not allow_symlinks and expanded.is_symlink():
        raise PathTraversalError(f"Symlinks not allowed: {path}")

    # Check if path exists when required
    if must_exist and not resolved.exists():
        raise InputValidationError(f"Path does not exist: {path}")

    return resolved


def validate_filename(filename: str) -> str:
    """
    Validate and sanitize a filename (no directory components).

    Args:
        filename: Filename to validate

    Returns:
        Sanitized filename

    Raises:
        InputValidationError: If filename is invalid
    """
    if not filename:
        raise InputValidationError("Filename cannot be empty")

    # Check length
    if len(filename) > MAX_FILENAME_LENGTH:
        raise InputValidationError(
            f"Filename too long: {len(filename)} > {MAX_FILENAME_LENGTH}"
        )

    # Check for null bytes
    if "\x00" in filename:
        raise InputValidationError("Null byte in filename")

    # Check for directory separators
    if "/" in filename or "\\" in filename:
        raise InputValidationError("Filename cannot contain path separators")

    # Check for path traversal
    if filename in (".", "..") or filename.startswith(".."):
        raise PathTraversalError("Invalid filename: path traversal detected")

    # Remove potentially dangerous characters
    # Keep alphanumeric, dots, dashes, underscores, spaces
    sanitized = re.sub(r"[^\w\s\-\.]", "_", filename)

    return sanitized


def validate_string(
    value: str,
    name: str = "value",
    min_length: int = 0,
    max_length: int = 10000,
    pattern: str | None = None,
    allow_empty: bool = False,
) -> str:
    """
    Validate a string input.

    Args:
        value: String to validate
        name: Name of the field (for error messages)
        min_length: Minimum length
        max_length: Maximum length
        pattern: Optional regex pattern to match
        allow_empty: If False, empty strings raise error

    Returns:
        Validated string (stripped of leading/trailing whitespace)

    Raises:
        InputValidationError: If validation fails
    """
    if value is None:
        if allow_empty:
            return ""
        raise InputValidationError(f"{name} cannot be None")

    if not isinstance(value, str):
        raise InputValidationError(
            f"{name} must be a string, got {type(value).__name__}"
        )

    # Strip whitespace
    value = value.strip()

    if not value and not allow_empty:
        raise InputValidationError(f"{name} cannot be empty")

    if len(value) < min_length:
        raise InputValidationError(f"{name} too short: {len(value)} < {min_length}")

    if len(value) > max_length:
        raise InputValidationError(f"{name} too long: {len(value)} > {max_length}")

    # Check for null bytes
    if "\x00" in value:
        raise InputValidationError(f"Null byte in {name}")

    # Check pattern if provided
    if pattern and not re.match(pattern, value):
        raise InputValidationError(f"{name} does not match required pattern")

    return value


def validate_integer(
    value: Any,
    name: str = "value",
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """
    Validate an integer input.

    Args:
        value: Value to validate
        name: Name of the field (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Validated integer

    Raises:
        InputValidationError: If validation fails
    """
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise InputValidationError(f"{name} must be an integer")

    if min_value is not None and result < min_value:
        raise InputValidationError(f"{name} must be >= {min_value}")

    if max_value is not None and result > max_value:
        raise InputValidationError(f"{name} must be <= {max_value}")

    return result


def validate_list(
    value: Any,
    name: str = "value",
    min_items: int = 0,
    max_items: int = 1000,
    item_validator: Callable | None = None,
) -> list:
    """
    Validate a list input.

    Args:
        value: Value to validate
        name: Name of the field
        min_items: Minimum number of items
        max_items: Maximum number of items
        item_validator: Optional function to validate each item

    Returns:
        Validated list

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(value, (list, tuple)):
        raise InputValidationError(f"{name} must be a list")

    result = list(value)

    if len(result) < min_items:
        raise InputValidationError(f"{name} must have at least {min_items} items")

    if len(result) > max_items:
        raise InputValidationError(f"{name} exceeds maximum of {max_items} items")

    if item_validator:
        result = [item_validator(item) for item in result]

    return result


def sanitize_shell_arg(arg: str) -> str:
    """
    Sanitize an argument for shell commands.

    WARNING: Prefer using subprocess with list args instead of shell=True.
    This is a fallback for cases where shell escaping is needed.

    Args:
        arg: Argument to sanitize

    Returns:
        Sanitized argument safe for shell use
    """
    if not arg:
        return "''"

    # Check for null bytes
    if "\x00" in arg:
        raise InputValidationError("Null byte in shell argument")

    # Simple approach: single-quote the entire string
    # and escape any single quotes within
    return "'" + arg.replace("'", "'\"'\"'") + "'"


def check_path_traversal_in_archive(_archive_path: str, member_name: str) -> bool:
    """
    Check if an archive member name attempts path traversal.

    Use this when extracting archives to prevent zip slip attacks.

    Args:
        _archive_path: Path to the archive being extracted (reserved for future use)
        member_name: Name of the member/entry in the archive

    Returns:
        True if safe, False if path traversal detected
    """
    # Normalize the member name
    member_name = member_name.replace("\\", "/")

    # Check for absolute paths
    if member_name.startswith("/"):
        return False

    # Check for path traversal
    if ".." in member_name.split("/"):
        return False

    # Check for drive letters (Windows)
    return not (len(member_name) >= 2 and member_name[1] == ":")
