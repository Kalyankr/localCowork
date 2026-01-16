from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)


def _to_path(val: str | dict) -> Path:
    """Convert string or dict to Path, expanding user home."""
    if isinstance(val, dict) and "path" in val:
        return Path(val["path"]).expanduser()
    return Path(str(val)).expanduser()


def list_files(path: str | dict) -> list[dict]:
    """List files in a directory with metadata."""
    p = _to_path(path)
    if not p.exists():
        logger.warning(f"Path does not exist: {p}")
        return []
    
    if not p.is_dir():
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


def move_file(src: str | dict | list[str | dict], dest: str | dict):
    if not src:
        return "No files found to move; skipping."
    
    dest_path = _to_path(dest)
    # Create destination if it's a directory and doesn't exist
    if not dest_path.suffix and not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)
    
    if isinstance(src, list):
        moved = []
        for s in src:
            s_path = _to_path(s)
            shutil.move(str(s_path), str(dest_path))
            moved.append(str(s_path))
        return f"Moved {len(moved)} files to {dest_path}"
    else:
        src_path = _to_path(src)
        shutil.move(str(src_path), str(dest_path))
        return f"Moved {src_path} → {dest_path}"


def create_dir(path: str | dict):
    p = _to_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Created directory {p}"


def rename_file(path: str | dict, new_name: str):
    p = _to_path(path)
    new_path = p.with_name(new_name)
    p.rename(new_path)
    return f"Renamed {p} → {new_path}"


def read_text(path: str | dict):
    p = _to_path(path)
    return p.read_text()


def write_text(path: str | dict, content: str):
    p = _to_path(path)
    p.write_text(content)
    return f"Wrote text to {p}"


def dispatch(op: str, **kwargs):
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
