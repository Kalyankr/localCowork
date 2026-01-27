from pathlib import Path
import logging
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from agent.security import (
    validate_path,
    validate_integer,
    PathTraversalError,
    InputValidationError,
)

logger = logging.getLogger(__name__)


class PDFOperationError(Exception):
    """Exception raised for PDF operation errors."""

    pass


def _to_path(val: str | dict) -> Path:
    """Convert string or dict to Path."""
    if isinstance(val, dict) and "path" in val:
        return Path(val["path"]).expanduser()
    return Path(str(val)).expanduser()


def _safe_path(val: str | dict, must_exist: bool = False) -> Path:
    """Convert to path and validate for security."""
    try:
        raw_path = val.get("path") if isinstance(val, dict) else str(val)
        return validate_path(raw_path, must_exist=must_exist, allow_symlinks=True)
    except PathTraversalError as e:
        raise PDFOperationError(f"Security error: {e}")
    except InputValidationError as e:
        raise PDFOperationError(str(e))


def _validate_pdf(path: Path) -> None:
    """Validate that path exists and is a PDF file."""
    if not path.exists():
        raise PDFOperationError(f"File not found: {path}")
    if not path.suffix.lower() == ".pdf":
        raise PDFOperationError(f"Not a PDF file: {path}")


def extract_metadata(files: list) -> dict:
    """Extract metadata from PDF files."""
    if not files:
        return {}

    results = {}
    for f in files:
        try:
            path = _safe_path(f)
        except PDFOperationError as e:
            results[str(f)] = {"error": str(e)}
            continue

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
                "subject": meta.subject if meta else None,
                "creator": meta.creator if meta else None,
                "pages": len(reader.pages),
            }
        except PdfReadError as e:
            logger.warning(f"Failed to read PDF {path}: {e}")
            results[key] = {"error": f"Invalid PDF: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error reading {path}: {e}")
            results[key] = {"error": str(e)}

    return results


def extract_text(
    path: str | dict,
    page_numbers: list[int] | None = None,
    max_pages: int | None = None,
) -> dict:
    """Extract text content from a PDF file.

    Args:
        path: Path to PDF file
        page_numbers: Optional list of specific page numbers (0-indexed)
        max_pages: Maximum number of pages to extract

    Returns:
        Dict with text content and metadata
    """
    p = _safe_path(path, must_exist=True)
    _validate_pdf(p)

    try:
        reader = PdfReader(str(p))
        total_pages = len(reader.pages)

        # Determine which pages to extract
        if page_numbers:
            pages_to_extract = [i for i in page_numbers if 0 <= i < total_pages]
        else:
            pages_to_extract = list(range(total_pages))

        if max_pages:
            pages_to_extract = pages_to_extract[:max_pages]

        text_content = []
        for page_num in pages_to_extract:
            page = reader.pages[page_num]
            text = page.extract_text() or ""
            text_content.append(
                {
                    "page": page_num + 1,  # 1-indexed for user display
                    "text": text,
                }
            )

        # Combined text for convenience
        full_text = "\n\n".join(
            f"--- Page {item['page']} ---\n{item['text']}" for item in text_content
        )

        return {
            "path": str(p),
            "total_pages": total_pages,
            "extracted_pages": len(text_content),
            "pages": text_content,
            "full_text": full_text[:100000],  # Limit size
        }

    except PdfReadError as e:
        raise PDFOperationError(f"Failed to read PDF: {e}")


def get_page_count(path: str | dict) -> dict:
    """Get the number of pages in a PDF file.

    Args:
        path: Path to PDF file

    Returns:
        Dict with page count info
    """
    p = _safe_path(path, must_exist=True)
    _validate_pdf(p)

    try:
        reader = PdfReader(str(p))
        return {
            "path": str(p),
            "pages": len(reader.pages),
        }
    except PdfReadError as e:
        raise PDFOperationError(f"Failed to read PDF: {e}")


def merge_pdfs(files: list, output: str | dict) -> str:
    """Merge multiple PDF files into one.

    Args:
        files: List of PDF file paths
        output: Output file path

    Returns:
        Success message
    """
    if not files:
        raise PDFOperationError("No files provided to merge")

    output_path = _safe_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()

    for f in files:
        p = _safe_path(f, must_exist=True)
        _validate_pdf(p)

        try:
            reader = PdfReader(str(p))
            for page in reader.pages:
                writer.add_page(page)
        except PdfReadError as e:
            raise PDFOperationError(f"Failed to read {p}: {e}")

    with open(output_path, "wb") as out_file:
        writer.write(out_file)

    return f"Merged {len(files)} PDFs → {output_path}"


def split_pdf(
    path: str | dict,
    output_dir: str | dict,
    pages_per_file: int = 1,
) -> dict:
    """Split a PDF into multiple files.

    Args:
        path: Path to PDF file
        output_dir: Directory for output files
        pages_per_file: Number of pages per output file

    Returns:
        Dict with split info
    """
    # Validate pages_per_file
    try:
        pages_per_file = validate_integer(
            pages_per_file, "pages_per_file", min_value=1, max_value=1000
        )
    except InputValidationError as e:
        raise PDFOperationError(str(e))

    p = _safe_path(path, must_exist=True)
    _validate_pdf(p)

    out_dir = _safe_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        reader = PdfReader(str(p))
        total_pages = len(reader.pages)

        output_files = []
        for i in range(0, total_pages, pages_per_file):
            writer = PdfWriter()
            end_page = min(i + pages_per_file, total_pages)

            for page_num in range(i, end_page):
                writer.add_page(reader.pages[page_num])

            output_name = f"{p.stem}_pages_{i + 1}-{end_page}.pdf"
            output_path = out_dir / output_name

            with open(output_path, "wb") as out_file:
                writer.write(out_file)

            output_files.append(str(output_path))

        return {
            "source": str(p),
            "total_pages": total_pages,
            "files_created": len(output_files),
            "output_files": output_files,
        }

    except PdfReadError as e:
        raise PDFOperationError(f"Failed to read PDF: {e}")


def dispatch(op: str = "extract", **kwargs) -> dict | str:
    """Dispatch PDF operations.

    Supported operations:
        - extract: Extract metadata from PDF(s)
        - text: Extract text content from PDF
        - pages: Get page count
        - merge: Merge multiple PDFs
        - split: Split PDF into multiple files
    """
    if op == "extract":
        files = kwargs.get("files", [])
        if isinstance(files, (str, dict)):
            files = [files]
        return extract_metadata(files)

    if op == "text":
        return extract_text(
            kwargs["path"],
            kwargs.get("page_numbers"),
            kwargs.get("max_pages"),
        )

    if op == "pages":
        return get_page_count(kwargs["path"])

    if op == "merge":
        return merge_pdfs(kwargs["files"], kwargs["output"])

    if op == "split":
        return split_pdf(
            kwargs["path"],
            kwargs["output_dir"],
            kwargs.get("pages_per_file", 1),
        )

    raise ValueError(f"Unsupported pdf op: {op}")
