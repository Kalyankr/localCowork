from pathlib import Path
from typing import List, Dict, Any


def list_files(path: str) -> List[str]:
    p = Path(path).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    return [str(child) for child in p.iterdir()]


def dispatch(op: str, **kwargs) -> Any:
    if op == "list":
        return list_files(kwargs["path"])
    else:
        raise ValueError(f"Unsupported file op: {op}")
