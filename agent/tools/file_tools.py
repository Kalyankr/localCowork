from pathlib import Path
import shutil


def list_files(path: str):
    p = Path(path).expanduser()
    return [str(x) for x in p.iterdir()] if p.exists() else []


def move_file(src: str, dest: str):
    src = Path(src).expanduser()
    dest = Path(dest).expanduser()
    shutil.move(str(src), str(dest))
    return f"Moved {src} → {dest}"


def rename_file(path: str, new_name: str):
    p = Path(path).expanduser()
    new_path = p.with_name(new_name)
    p.rename(new_path)
    return f"Renamed {p} → {new_path}"


def read_text(path: str):
    return Path(path).expanduser().read_text()


def write_text(path: str, content: str):
    Path(path).expanduser().write_text(content)
    return f"Wrote text to {path}"


def dispatch(op: str, **kwargs):
    if op == "list":
        return list_files(kwargs["path"])
    if op == "move":
        return move_file(kwargs["src"], kwargs["dest"])
    if op == "rename":
        return rename_file(kwargs["path"], kwargs["new_name"])
    if op == "read":
        return read_text(kwargs["path"])
    if op == "write":
        return write_text(kwargs["path"], kwargs["content"])
    raise ValueError(f"Unsupported file op: {op}")
