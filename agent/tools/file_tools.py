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


def list_files(path: str | dict, raise_on_missing: bool = False, recursive: bool = False, pattern: str | None = None) -> list[dict]:
    """List files in a directory with metadata.
    
    Args:
        path: Directory path to list
        raise_on_missing: If True, raise error for missing path; else return empty list
        recursive: If True, list files recursively
        pattern: Optional glob pattern to filter files (e.g., '*.txt')
        
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
    
    # Choose iteration method based on recursive flag
    if recursive:
        iterator = p.rglob(pattern or "*")
    elif pattern:
        iterator = p.glob(pattern)
    else:
        iterator = p.iterdir()
    
    for x in iterator:
        try:
            stat = x.stat()
            results.append({
                "path": str(x),
                "name": x.name,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "is_dir": x.is_dir(),
                "extension": x.suffix.lower() if x.is_file() else None,
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


def copy_file(src: str | dict | list[str | dict], dest: str | dict) -> str:
    """Copy file(s) to destination.
    
    Args:
        src: Source file/path or list of sources
        dest: Destination path
        
    Returns:
        Success message describing the operation
        
    Raises:
        FileOperationError: If source doesn't exist
    """
    if not src:
        return "No files found to copy; skipping."
    
    dest_path = _to_path(dest)
    
    if isinstance(src, list):
        # Multiple files - dest must be a directory
        if not dest_path.exists():
            dest_path.mkdir(parents=True, exist_ok=True)
        
        copied = []
        errors = []
        for s in src:
            s_path = _to_path(s)
            if not s_path.exists():
                errors.append(f"Source not found: {s_path}")
                continue
            try:
                if s_path.is_dir():
                    shutil.copytree(str(s_path), str(dest_path / s_path.name))
                else:
                    shutil.copy2(str(s_path), str(dest_path))
                copied.append(str(s_path))
            except Exception as e:
                errors.append(f"Failed to copy {s_path}: {e}")
        
        msg = f"Copied {len(copied)} files to {dest_path}"
        if errors:
            msg += f" ({len(errors)} errors: {'; '.join(errors[:3])})"
        return msg
    else:
        src_path = _to_path(src)
        if not src_path.exists():
            raise FileOperationError(f"Source not found: {src_path}")
        
        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dest_path))
        else:
            # Create parent dirs if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_path), str(dest_path))
        return f"Copied {src_path} → {dest_path}"


def delete_file(path: str | dict, recursive: bool = False) -> str:
    """Delete a file or directory.
    
    Args:
        path: Path to delete
        recursive: If True, delete directories recursively
        
    Returns:
        Success message
        
    Raises:
        FileOperationError: If path doesn't exist
    """
    p = _to_path(path)
    if not p.exists():
        raise FileOperationError(f"Path not found: {p}")
    
    if p.is_dir():
        if recursive:
            shutil.rmtree(str(p))
            return f"Deleted directory (recursive): {p}"
        else:
            try:
                p.rmdir()
                return f"Deleted empty directory: {p}"
            except OSError:
                raise FileOperationError(f"Directory not empty (use recursive=True): {p}")
    else:
        p.unlink()
        return f"Deleted file: {p}"


def get_file_info(path: str | dict) -> dict:
    """Get detailed information about a file or directory.
    
    Args:
        path: Path to inspect
        
    Returns:
        Dict with file metadata
        
    Raises:
        FileOperationError: If path doesn't exist
    """
    p = _to_path(path)
    if not p.exists():
        raise FileOperationError(f"Path not found: {p}")
    
    stat = p.stat()
    return {
        "path": str(p),
        "name": p.name,
        "extension": p.suffix.lower() if p.is_file() else None,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "is_symlink": p.is_symlink(),
        "size": stat.st_size,
        "size_human": _format_size(stat.st_size),
        "mtime": stat.st_mtime,
        "atime": stat.st_atime,
        "ctime": stat.st_ctime,
        "mode": oct(stat.st_mode),
    }


def get_dir_size(path: str | dict) -> dict:
    """Calculate total size of a directory.
    
    Args:
        path: Directory path
        
    Returns:
        Dict with size information
    """
    p = _to_path(path)
    if not p.exists():
        raise FileOperationError(f"Path not found: {p}")
    if not p.is_dir():
        raise FileOperationError(f"Not a directory: {p}")
    
    total_size = 0
    file_count = 0
    dir_count = 0
    
    for item in p.rglob("*"):
        try:
            if item.is_file():
                total_size += item.stat().st_size
                file_count += 1
            elif item.is_dir():
                dir_count += 1
        except (PermissionError, OSError):
            pass
    
    return {
        "path": str(p),
        "total_size": total_size,
        "total_size_human": _format_size(total_size),
        "file_count": file_count,
        "dir_count": dir_count,
    }


def find_files(path: str | dict, pattern: str, recursive: bool = True) -> list[dict]:
    """Find files matching a pattern.
    
    Args:
        path: Directory to search in
        pattern: Glob pattern (e.g., '*.py', '**/*.txt')
        recursive: If True, search recursively
        
    Returns:
        List of matching file info dicts
    """
    p = _to_path(path)
    if not p.exists():
        raise FileOperationError(f"Path not found: {p}")
    
    results = []
    iterator = p.rglob(pattern) if recursive else p.glob(pattern)
    
    for x in iterator:
        try:
            stat = x.stat()
            results.append({
                "path": str(x),
                "name": x.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "is_dir": x.is_dir(),
            })
        except (PermissionError, OSError):
            pass
    
    return results


def _format_size(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def dispatch(op: str, **kwargs) -> str | list[dict] | dict:
    """Dispatch file operations.
    
    Supported operations:
        - list: List files in directory
        - move: Move file(s) to destination
        - copy: Copy file(s) to destination
        - delete: Delete file or directory
        - mkdir: Create directory
        - rename: Rename file/directory
        - read: Read text file content
        - write: Write text to file
        - info: Get file/directory info
        - size: Get directory size
        - find: Find files by pattern
    """
    if op == "list":
        return list_files(
            kwargs["path"],
            kwargs.get("raise_on_missing", False),
            kwargs.get("recursive", False),
            kwargs.get("pattern"),
        )
    if op == "move":
        return move_file(kwargs["src"], kwargs["dest"])
    if op == "copy":
        return copy_file(kwargs["src"], kwargs["dest"])
    if op == "delete":
        return delete_file(kwargs["path"], kwargs.get("recursive", False))
    if op == "mkdir":
        return create_dir(kwargs["path"])
    if op == "rename":
        return rename_file(kwargs["path"], kwargs["new_name"])
    if op == "read":
        return read_text(kwargs["path"])
    if op == "write":
        return write_text(kwargs["path"], kwargs["content"])
    if op == "info":
        return get_file_info(kwargs["path"])
    if op == "size":
        return get_dir_size(kwargs["path"])
    if op == "find":
        return find_files(kwargs["path"], kwargs["pattern"], kwargs.get("recursive", True))
    raise ValueError(f"Unsupported file op: {op}")
