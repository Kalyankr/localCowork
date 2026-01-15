import pandas as pd
from pathlib import Path


def csv_to_excel(csv_path: str, excel_path: str):
    df = pd.read_csv(Path(csv_path).expanduser())
    df.to_excel(Path(excel_path).expanduser(), index=False)
    return f"Converted {csv_path} → {excel_path}"


def dispatch(op: str, **kwargs):
    if op == "csv_to_excel":
        return csv_to_excel(kwargs["csv_path"], kwargs["excel_path"])
    raise ValueError(f"Unsupported data op: {op}")
