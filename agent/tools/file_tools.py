from pathlib import Path
import shutil
import logging
from typing import Any

logger = logging.getLogger(__name__)


class FileOperationError(Exception):
    """Exception raised for file operation errors."""
    pass


def _to_path(val: str | dict) -> Path:
    """Convert string or dict to Path, expanding user home."""
    if isinstance(val, dict) and "path" in val:
        return Path(val["path"]).expanduser()
    return Path(str(val)).expanduser()


def _validate_path(path: Path, must_exist: bool = False, must_be_dir: bool = False) -> None:
    """Validate a path, raising FileOperationError if invalid."""
    if must_exist and not path.exists():
        raise FileOperationError(f"Path does not exist: {path}")
    if must_be_dir and path.exists() and not path.is_dir():
        raise FileOperationError(f"Path is not a directory: {path}")


def list_files(path: str | dict, raise_on_missing: bool = False, raise_on_missing: bool = False) -> list[dict]:
    """List files in a directory with metadata.
    
    Args:
        path: Directory path to list
        raise_on_missing: If True, raise error for missing path; else return empty list
        
    Returns:
        List of file info dicts with path, name, size, mtime, is_dir
        
    Raises:
        FileOperationError: If raise_on_missing=True and path doesn't exist
    """
    p = _to_path(path)
    
    if not p.exists():
        if raise_on_missing:
            raise FileOperationError(f"Path does not exist: {p}")
        logger.warning(f"Path does not exist: {p}")
        return []
    
    if not p.is_dir():
        if raise_on_missing:
            raise FileOperationError(f"Path is not a directory: {p}")
        logger.warning(f"Path is not a directory: {p}")
        return []
    
    results = []
    for x in p.iterdir():
        try:
            stat = x.stat()
            results.append({
                "path": str(x),
                "name": x.name,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "is_dir": x.is_dir()
            })
        except PermissionError:
            logger.debug(f"Permission denied: {x}")
        except OSError as e:
            logger.debug(f"Cannot stat {x}: {e}")
    return results


def move_file(src: str | dict | list[str | dict], dest: str | dict) -> str:
    """Move file(s) to destination.
    
    Args:
        src: Source file/path or list of sources
        dest: Destination path
        
    Returns:
        Success message describing the operation
        
    Raises:
        FileOperationError: If source doesn't exist
    """
    if not src:
        return "No files found to move; skipping."
    
    dest_path = _to_path(dest)
    # Create destination if it's a directory and doesn't exist
    if not dest_path.suffix and not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)
    
    if isinstance(src, list):
        moved = []
        errors = []
        for s in src:
            s_path = _to_path(s)
            if not s_path.exists():
                errors.append(f"Source not found: {s_path}")
                continue
            try:
                shutil.move(str(s_path), str(dest_path))
                moved.append(str(s_path))
            except Exception as e:
                errors.append(f"Failed to move {s_path}: {e}")
        
        msg = f"Moved {len(moved)} files to {dest_path}"
        if errors:
            msg += f" ({len(errors)} errors: {'; '.join(errors[:3])})"
        return msg
    else:
        src_path = _to_path(src)
        if not src_path.exists():
            raise FileOperationError(f"Source not found: {src_path}")
        shutil.move(str(src_path), str(dest_path))
        return f"Moved {src_path} → {dest_path}"


def create_dir(path: str | dict) -> str:
    """Create a directory (and parents if needed).
    
    Args:
        path: Directory path to create
        
    Returns:
        Success message
    """
    p = _to_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Created directory {p}"


def rename_file(path: str | dict, new_name: str) -> str:
    """Rename a file or directory.
    
    Args:
        path: Path to rename
        new_name: New name for the file/directory
        
    Returns:
        Success message
        
    Raises:
        FileOperationError: If path doesn't exist
    """
    if not new_name:
        raise FileOperationError("new_name cannot be empty")
    p = _to_path(path)
    if not p.exists():
        raise FileOperationError(f"Path not found: {p}")
    new_path = p.with_name(new_name)
    p.rename(new_path)
    return f"Renamed {p} → {new_path}"


def read_text(path: str | dict) -> str:
    """Read text content from a file.
    
    Args:
        path: Path to read
        
    Returns:
        File content as string
        
    Raises:
        FileOperationError: If path doesn't exist or isn't a file
    """
    p = _to_path(path)
    if not p.exists():
        raise FileOperationError(f"File not found: {p}")
    if not p.is_file():
        raise FileOperationError(f"Not a file: {p}")
    return p.read_text()


def write_text(path: str | dict, content: str) -> str:
    """Write text content to a file.
    
    Args:
        path: Path to write to
        content: Content to write
        
    Returns:
        Success message
    """
    if content is None:
        raise FileOperationError("content cannot be None")
    p = _to_path(path)
    # Create parent directories if needed
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote text to {p}"


def dispatch(op: str, **kwargs) -> str | list[dict]:
    if op == "list":
        return list_files(kwargs["path"])
    if op == "move":
        return move_file(kwargs["src"], kwargs["dest"])
    if op == "mkdir":
        return create_dir(kwargs["path"])
    if op == "rename":
        return rename_file(kwargs["path"], kwargs["new_name"])
    if op == "read":
        return read_text(kwargs["path"])
    if op == "write":
        return write_text(kwargs["path"], kwargs["content"])
    raise ValueError(f"Unsupported file op: {op}")
