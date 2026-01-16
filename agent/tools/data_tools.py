import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def _to_path(val: str | dict) -> Path:
    """Convert string or dict to Path."""
    if isinstance(val, dict) and "path" in val:
        return Path(val["path"]).expanduser()
    return Path(str(val)).expanduser()


def csv_to_excel(csv_path: str | dict, excel_path: str | dict) -> str:
    """Convert a CSV file to Excel format."""
    src = _to_path(csv_path)
    dest = _to_path(excel_path)
    
    if not src.exists():
        raise FileNotFoundError(f"CSV file not found: {src}")
    
    # Create parent dirs for dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(src)
        df.to_excel(dest, index=False)
        logger.info(f"Converted {src} → {dest}")
        return f"Converted {src} → {dest}"
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {src}")
    except Exception as e:
        raise RuntimeError(f"Failed to convert CSV: {e}")


def dispatch(op: str, **kwargs) -> str:
    """Dispatch data operations."""
    if op == "csv_to_excel":
        csv_path = kwargs.get("csv_path") or kwargs.get("src")
        excel_path = kwargs.get("excel_path") or kwargs.get("dest")
        if not csv_path or not excel_path:
            raise ValueError("csv_path and excel_path are required")
        return csv_to_excel(csv_path, excel_path)
    
    raise ValueError(f"Unsupported data op: {op}")
