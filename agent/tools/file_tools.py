from pathlib import Path
import shutil


def list_files(path: str):
    p = Path(path).expanduser()
    if not p.exists():
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
        except:
            continue
    return results


def move_file(src: str | list[str], dest: str):
    if not src:
        raise ValueError("Source path(s) for move cannot be empty")
    
    dest = Path(dest).expanduser()
    # Create destination if it's a directory and doesn't exist
    if not dest.suffix and not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
    
    if isinstance(src, list):
        moved = []
        for s in src:
            s_path = Path(s).expanduser()
            shutil.move(str(s_path), str(dest))
            moved.append(str(s_path))
        return f"Moved {len(moved)} files to {dest}"
    else:
        src = Path(src).expanduser()
        shutil.move(str(src), str(dest))
        return f"Moved {src} → {dest}"


def create_dir(path: str):
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return f"Created directory {p}"


def rename_file(path: str, new_name: str):
    p = Path(path).expanduser()
    new_path = p.with_name(new_name)
    p.rename(new_path)
    return f"Renamed {p} → {new_path}"


def read_text(path: str):
    if not path or not str(path).strip():
        raise ValueError("Path for read cannot be empty")
    return Path(path).expanduser().read_text()


def write_text(path: str, content: str):
    Path(path).expanduser().write_text(content)
    return f"Wrote text to {path}"


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
