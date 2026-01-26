"""JSON operations: parse, query, transform JSON data."""

import json
from pathlib import Path
from typing import Any, Optional


def read_json(path: str) -> Any:
    """Read and parse a JSON file."""
    p = Path(path).expanduser()
    with open(p, 'r') as f:
        return json.load(f)


def write_json(path: str, data: Any, indent: int = 2) -> str:
    """Write data to a JSON file."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(data, f, indent=indent)
    return f"Wrote JSON to {p}"


def parse_json(text: str) -> Any:
    """Parse a JSON string."""
    return json.loads(text)


def query_json(data: Any, path: str) -> Any:
    """
    Query JSON data using a simple dot-notation path.
    Supports: 
      - "key" - access dict key
      - "key.nested" - nested access
      - "[0]" - array index
      - "key[0].nested" - combined
    
    Example: query_json(data, "users[0].name")
    """
    import re
    
    # Parse path into tokens
    tokens = re.findall(r'(\w+)|\[(\d+)\]', path)
    
    result = data
    for token in tokens:
        key, index = token
        if key:
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return None
        elif index:
            if isinstance(result, list):
                idx = int(index)
                if 0 <= idx < len(result):
                    result = result[idx]
                else:
                    return None
            else:
                return None
    
    return result


def filter_json(data: list, key: str, value: Any) -> list:
    """Filter a list of objects where key == value."""
    if not isinstance(data, list):
        raise ValueError("filter_json expects a list")
    return [item for item in data if isinstance(item, dict) and item.get(key) == value]


def map_json(data: list, keys: list) -> list:
    """
    Extract specific keys from each object in a list.
    Example: map_json([{"a": 1, "b": 2}], ["a"]) -> [{"a": 1}]
    """
    if not isinstance(data, list):
        raise ValueError("map_json expects a list")
    
    result = []
    for item in data:
        if isinstance(item, dict):
            result.append({k: item.get(k) for k in keys})
    return result


def merge_json(*objects) -> dict:
    """Merge multiple JSON objects (later values override earlier)."""
    result = {}
    for obj in objects:
        if isinstance(obj, dict):
            result.update(obj)
    return result


def flatten_json(data: dict, prefix: str = "", sep: str = ".") -> dict:
    """
    Flatten nested JSON into dot-notation keys.
    Example: {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.update(flatten_json(value, new_key, sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_json(item, f"{new_key}[{i}]", sep))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = value
    
    return items


def diff_json(obj1: Any, obj2: Any) -> dict:
    """
    Compare two JSON objects and return differences.
    Returns {"added": [], "removed": [], "changed": []}.
    """
    def _diff(a, b, path=""):
        changes = {"added": [], "removed": [], "changed": []}
        
        if type(a) != type(b):
            changes["changed"].append({"path": path, "from": a, "to": b})
            return changes
            
        if isinstance(a, dict):
            all_keys = set(a.keys()) | set(b.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in a:
                    changes["added"].append({"path": new_path, "value": b[key]})
                elif key not in b:
                    changes["removed"].append({"path": new_path, "value": a[key]})
                else:
                    sub = _diff(a[key], b[key], new_path)
                    changes["added"].extend(sub["added"])
                    changes["removed"].extend(sub["removed"])
                    changes["changed"].extend(sub["changed"])
                    
        elif isinstance(a, list):
            if a != b:
                changes["changed"].append({"path": path, "from": a, "to": b})
        else:
            if a != b:
                changes["changed"].append({"path": path, "from": a, "to": b})
        
        return changes
    
    return _diff(obj1, obj2)


def dispatch(op: str, **kwargs) -> Any:
    """Dispatch JSON operations."""
    if op == "read":
        return read_json(kwargs["path"])
    if op == "write":
        return write_json(kwargs["path"], kwargs["data"], kwargs.get("indent", 2))
    if op == "parse":
        return parse_json(kwargs["text"])
    if op == "query":
        return query_json(kwargs["data"], kwargs["path"])
    if op == "filter":
        return filter_json(kwargs["data"], kwargs["key"], kwargs["value"])
    if op == "map":
        return map_json(kwargs["data"], kwargs["keys"])
    if op == "merge":
        return merge_json(*kwargs.get("objects", []))
    if op == "flatten":
        return flatten_json(kwargs["data"], kwargs.get("prefix", ""), kwargs.get("sep", "."))
    if op == "diff":
        return diff_json(kwargs["obj1"], kwargs["obj2"])
    raise ValueError(f"Unsupported json op: {op}")
