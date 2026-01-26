"""Data operations: convert, transform, and analyze data files."""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DataOperationError(Exception):
    """Exception raised for data operation errors."""
    pass


def _to_path(val: str | dict) -> Path:
    """Convert string or dict to Path."""
    if isinstance(val, dict) and "path" in val:
        return Path(val["path"]).expanduser()
    return Path(str(val)).expanduser()


def _validate_file(path: Path, extensions: list[str] | None = None) -> None:
    """Validate file exists and optionally check extension."""
    if not path.exists():
        raise DataOperationError(f"File not found: {path}")
    if extensions and path.suffix.lower() not in extensions:
        raise DataOperationError(
            f"Unsupported file type: {path.suffix}. Expected: {extensions}"
        )


def csv_to_excel(csv_path: str | dict, excel_path: str | dict) -> str:
    """Convert a CSV file to Excel format."""
    src = _to_path(csv_path)
    dest = _to_path(excel_path)
    
    _validate_file(src, ['.csv'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(src)
        df.to_excel(dest, index=False)
        logger.info(f"Converted {src} → {dest}")
        return f"Converted {src} → {dest} ({len(df)} rows)"
    except pd.errors.EmptyDataError:
        raise DataOperationError(f"CSV file is empty: {src}")
    except Exception as e:
        raise DataOperationError(f"Failed to convert CSV: {e}")


def excel_to_csv(excel_path: str | dict, csv_path: str | dict, sheet: str | int = 0) -> str:
    """Convert an Excel file to CSV format.
    
    Args:
        excel_path: Path to Excel file
        csv_path: Output CSV path
        sheet: Sheet name or index to convert
    """
    src = _to_path(excel_path)
    dest = _to_path(csv_path)
    
    _validate_file(src, ['.xlsx', '.xls'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_excel(src, sheet_name=sheet)
        df.to_csv(dest, index=False)
        return f"Converted {src} → {dest} ({len(df)} rows)"
    except Exception as e:
        raise DataOperationError(f"Failed to convert Excel: {e}")


def json_to_csv(json_path: str | dict, csv_path: str | dict) -> str:
    """Convert a JSON file (array of objects) to CSV format."""
    src = _to_path(json_path)
    dest = _to_path(csv_path)
    
    _validate_file(src, ['.json'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(src) as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise DataOperationError("JSON must be an array of objects")
        
        df = pd.DataFrame(data)
        df.to_csv(dest, index=False)
        return f"Converted {src} → {dest} ({len(df)} rows)"
    except json.JSONDecodeError as e:
        raise DataOperationError(f"Invalid JSON: {e}")
    except Exception as e:
        raise DataOperationError(f"Failed to convert JSON: {e}")


def csv_to_json(csv_path: str | dict, json_path: str | dict) -> str:
    """Convert a CSV file to JSON format (array of objects)."""
    src = _to_path(csv_path)
    dest = _to_path(json_path)
    
    _validate_file(src, ['.csv'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(src)
        data = df.to_dict(orient='records')
        
        with open(dest, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return f"Converted {src} → {dest} ({len(df)} rows)"
    except Exception as e:
        raise DataOperationError(f"Failed to convert CSV: {e}")


def preview_data(path: str | dict, rows: int = 10) -> dict:
    """Preview the first N rows of a data file.
    
    Args:
        path: Path to CSV, Excel, or JSON file
        rows: Number of rows to preview
        
    Returns:
        Dict with preview data and metadata
    """
    p = _to_path(path)
    _validate_file(p)
    
    ext = p.suffix.lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(p, nrows=rows)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(p, nrows=rows)
        elif ext == '.json':
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data[:rows])
            else:
                return {
                    "path": str(p),
                    "type": "json_object",
                    "preview": data,
                }
        else:
            raise DataOperationError(f"Unsupported file type: {ext}")
        
        return {
            "path": str(p),
            "type": ext,
            "columns": list(df.columns),
            "shape": list(df.shape),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.to_dict(orient='records'),
        }
        
    except Exception as e:
        raise DataOperationError(f"Failed to preview data: {e}")


def get_stats(path: str | dict) -> dict:
    """Get statistics for numeric columns in a data file.
    
    Args:
        path: Path to CSV or Excel file
        
    Returns:
        Dict with statistical summary
    """
    p = _to_path(path)
    _validate_file(p)
    
    ext = p.suffix.lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(p)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(p)
        else:
            raise DataOperationError(f"Unsupported file type: {ext}")
        
        # Get basic info
        info = {
            "path": str(p),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
        }
        
        # Get stats for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            info["numeric_stats"] = numeric_df.describe().to_dict()
        
        return info
        
    except Exception as e:
        raise DataOperationError(f"Failed to get stats: {e}")


def filter_data(
    path: str | dict,
    output: str | dict,
    column: str,
    operator: str,
    value: Any,
) -> str:
    """Filter rows in a data file based on a condition.
    
    Args:
        path: Path to input file
        output: Path to output file
        column: Column name to filter on
        operator: Comparison operator (eq, ne, gt, lt, ge, le, contains)
        value: Value to compare against
        
    Returns:
        Success message with row count
    """
    p = _to_path(path)
    out_p = _to_path(output)
    _validate_file(p)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    ext = p.suffix.lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(p)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(p)
        else:
            raise DataOperationError(f"Unsupported file type: {ext}")
        
        if column not in df.columns:
            raise DataOperationError(f"Column not found: {column}")
        
        # Apply filter
        if operator == "eq":
            mask = df[column] == value
        elif operator == "ne":
            mask = df[column] != value
        elif operator == "gt":
            mask = df[column] > value
        elif operator == "lt":
            mask = df[column] < value
        elif operator == "ge":
            mask = df[column] >= value
        elif operator == "le":
            mask = df[column] <= value
        elif operator == "contains":
            mask = df[column].astype(str).str.contains(str(value), case=False, na=False)
        else:
            raise DataOperationError(f"Unknown operator: {operator}")
        
        filtered_df = df[mask]
        
        # Save output
        out_ext = out_p.suffix.lower()
        if out_ext == '.csv':
            filtered_df.to_csv(out_p, index=False)
        elif out_ext in ['.xlsx', '.xls']:
            filtered_df.to_excel(out_p, index=False)
        elif out_ext == '.json':
            filtered_df.to_json(out_p, orient='records', indent=2)
        else:
            filtered_df.to_csv(out_p, index=False)
        
        return f"Filtered {len(df)} → {len(filtered_df)} rows, saved to {out_p}"
        
    except Exception as e:
        raise DataOperationError(f"Failed to filter data: {e}")


def dispatch(op: str, **kwargs) -> str | dict:
    """Dispatch data operations.
    
    Supported operations:
        - csv_to_excel: Convert CSV to Excel
        - excel_to_csv: Convert Excel to CSV
        - json_to_csv: Convert JSON to CSV
        - csv_to_json: Convert CSV to JSON
        - preview: Preview first N rows
        - stats: Get statistical summary
        - filter: Filter rows by condition
    """
    if op == "csv_to_excel":
        csv_path = kwargs.get("csv_path") or kwargs.get("src")
        excel_path = kwargs.get("excel_path") or kwargs.get("dest")
        if not csv_path or not excel_path:
            raise DataOperationError("csv_path and excel_path are required")
        return csv_to_excel(csv_path, excel_path)
    
    if op == "excel_to_csv":
        excel_path = kwargs.get("excel_path") or kwargs.get("src")
        csv_path = kwargs.get("csv_path") or kwargs.get("dest")
        if not excel_path or not csv_path:
            raise DataOperationError("excel_path and csv_path are required")
        return excel_to_csv(excel_path, csv_path, kwargs.get("sheet", 0))
    
    if op == "json_to_csv":
        return json_to_csv(kwargs["json_path"], kwargs["csv_path"])
    
    if op == "csv_to_json":
        return csv_to_json(kwargs["csv_path"], kwargs["json_path"])
    
    if op == "preview":
        return preview_data(kwargs["path"], kwargs.get("rows", 10))
    
    if op == "stats":
        return get_stats(kwargs["path"])
    
    if op == "filter":
        return filter_data(
            kwargs["path"],
            kwargs["output"],
            kwargs["column"],
            kwargs["operator"],
            kwargs["value"],
        )
    
    raise ValueError(f"Unsupported data op: {op}")
