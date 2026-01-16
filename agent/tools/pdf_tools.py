from pathlib import Path
import logging
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


def _to_path(val: str | dict) -> Path:
    """Convert string or dict to Path."""
    if isinstance(val, dict) and "path" in val:
        return Path(val["path"]).expanduser()
    return Path(str(val)).expanduser()


def extract_metadata(files: list) -> dict:
    """Extract metadata from PDF files."""
    if not files:
        return {}
    
    results = {}
    for f in files:
        path = _to_path(f)
        key = str(path)
        
        if not path.exists():
            results[key] = {"error": "File not found"}
            continue
            
        if not path.suffix.lower() == ".pdf":
            results[key] = {"error": "Not a PDF file"}
            continue
        
        try:
            reader = PdfReader(str(path))
            meta = reader.metadata
            results[key] = {
                "title": meta.title if meta else None,
                "author": meta.author if meta else None,
                "pages": len(reader.pages),
            }
        except PdfReadError as e:
            logger.warning(f"Failed to read PDF {path}: {e}")
            results[key] = {"error": f"Invalid PDF: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error reading {path}: {e}")
            results[key] = {"error": str(e)}
    
    return results


def dispatch(op: str = "extract", **kwargs) -> dict:
    """Dispatch PDF operations."""
    files = kwargs.get("files", [])
    
    # Handle single file or list
    if isinstance(files, (str, dict)):
        files = [files]
    
    if op == "extract":
        return extract_metadata(files)
    
    raise ValueError(f"Unsupported pdf op: {op}")
