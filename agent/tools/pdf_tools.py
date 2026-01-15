from pathlib import Path
from pypdf import PdfReader


def extract_metadata(files):
    results = {}
    for f in files:
        path = Path(f).expanduser()
        reader = PdfReader(str(path))
        results[f] = reader.metadata
    return results


def dispatch(op: str = "extract", **kwargs):
    return extract_metadata(kwargs["files"])
