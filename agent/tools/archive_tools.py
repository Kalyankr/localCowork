"""Archive operations: zip, unzip, tar, extract."""

import zipfile
import tarfile
import shutil
from pathlib import Path

from agent.security import (
    validate_path,
    check_path_traversal_in_archive,
    PathTraversalError,
    InputValidationError,
)


class ArchiveOperationError(Exception):
    """Error during archive operation."""

    pass


def _safe_path(val: str | dict, must_exist: bool = False) -> Path:
    """Convert to path and validate for security."""
    try:
        raw_path = val.get("path") if isinstance(val, dict) else str(val)
        return validate_path(raw_path, must_exist=must_exist, allow_symlinks=True)
    except PathTraversalError as e:
        raise ArchiveOperationError(f"Security error: {e}")
    except InputValidationError as e:
        raise ArchiveOperationError(str(e))


def _validate_archive_entries(entries: list[str], dest_path: Path) -> None:
    """Validate archive entries for zip slip attacks."""
    malicious = check_path_traversal_in_archive(entries, dest_path)
    if malicious:
        raise ArchiveOperationError(
            f"Zip slip attack detected! Malicious entries: {malicious[:5]}"
        )


def create_zip(source: str | list, dest: str, compression: str = "deflate") -> str:
    """
    Create a ZIP archive.
    source: single path or list of paths to include
    dest: output ZIP file path
    compression: "deflate" (default), "store" (no compression), or "bzip2"
    """
    dest_path = _safe_path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate compression option
    compression_map = {
        "deflate": zipfile.ZIP_DEFLATED,
        "store": zipfile.ZIP_STORED,
        "bzip2": zipfile.ZIP_BZIP2,
    }
    if compression not in compression_map:
        raise ArchiveOperationError(
            f"Invalid compression: {compression}. Use: deflate, store, bzip2"
        )
    comp = compression_map[compression]

    sources = [source] if isinstance(source, str) else source

    with zipfile.ZipFile(dest_path, "w", compression=comp) as zf:
        for src in sources:
            src_path = _safe_path(src, must_exist=True)
            if src_path.is_file():
                zf.write(src_path, src_path.name)
            elif src_path.is_dir():
                for file in src_path.rglob("*"):
                    if file.is_file():
                        arcname = file.relative_to(src_path.parent)
                        zf.write(file, arcname)

    return f"Created ZIP archive: {dest_path}"


def extract_zip(source: str, dest: str) -> str:
    """Extract a ZIP archive to destination directory with zip slip protection."""
    src_path = _safe_path(source, must_exist=True)
    dest_path = _safe_path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src_path, "r") as zf:
        # Check for zip slip attack before extracting
        entries = [info.filename for info in zf.infolist()]
        _validate_archive_entries(entries, dest_path)
        zf.extractall(dest_path)

    return f"Extracted ZIP to: {dest_path}"


def list_zip(source: str) -> list:
    """List contents of a ZIP archive."""
    src_path = _safe_path(source, must_exist=True)

    with zipfile.ZipFile(src_path, "r") as zf:
        return [
            {
                "name": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "is_dir": info.is_dir(),
            }
            for info in zf.infolist()
        ]


def create_tar(source: str | list, dest: str, compression: str = "gz") -> str:
    """
    Create a TAR archive.
    source: single path or list of paths
    dest: output file path
    compression: "gz" (gzip), "bz2" (bzip2), "xz", or "" (no compression)
    """
    # Validate compression option
    valid_compressions = {"gz", "bz2", "xz", ""}
    if compression not in valid_compressions:
        raise ArchiveOperationError(
            f"Invalid compression: {compression}. Use: gz, bz2, xz, or empty string"
        )

    dest_path = _safe_path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    mode = f"w:{compression}" if compression else "w"

    sources = [source] if isinstance(source, str) else source

    with tarfile.open(dest_path, mode) as tf:
        for src in sources:
            src_path = _safe_path(src, must_exist=True)
            tf.add(src_path, arcname=src_path.name)

    return f"Created TAR archive: {dest_path}"


def extract_tar(source: str, dest: str) -> str:
    """Extract a TAR archive (auto-detects compression) with path traversal protection."""
    src_path = _safe_path(source, must_exist=True)
    dest_path = _safe_path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    with tarfile.open(src_path, "r:*") as tf:
        # Check for path traversal attack before extracting
        entries = [member.name for member in tf.getmembers()]
        _validate_archive_entries(entries, dest_path)
        tf.extractall(dest_path)

    return f"Extracted TAR to: {dest_path}"


def list_tar(source: str) -> list:
    """List contents of a TAR archive."""
    src_path = _safe_path(source, must_exist=True)

    with tarfile.open(src_path, "r:*") as tf:
        return [
            {
                "name": member.name,
                "size": member.size,
                "is_dir": member.isdir(),
                "is_file": member.isfile(),
            }
            for member in tf.getmembers()
        ]


def extract_auto(source: str, dest: str) -> str:
    """
    Auto-detect archive type and extract.
    Supports: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz
    """
    src_path = _safe_path(source, must_exist=True)
    name = src_path.name.lower()

    if name.endswith(".zip"):
        return extract_zip(source, dest)
    elif any(
        name.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"]
    ):
        return extract_tar(source, dest)
    else:
        # Try shutil.unpack_archive as fallback
        dest_path = _safe_path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(src_path, dest_path)
        return f"Extracted to: {dest_path}"


def dispatch(op: str, **kwargs) -> str | list:
    """Dispatch archive operations."""
    if op == "zip":
        return create_zip(
            kwargs["source"], kwargs["dest"], kwargs.get("compression", "deflate")
        )
    if op == "unzip":
        return extract_zip(kwargs["source"], kwargs["dest"])
    if op == "list_zip":
        return list_zip(kwargs["source"])
    if op == "tar":
        return create_tar(
            kwargs["source"], kwargs["dest"], kwargs.get("compression", "gz")
        )
    if op == "untar":
        return extract_tar(kwargs["source"], kwargs["dest"])
    if op == "list_tar":
        return list_tar(kwargs["source"])
    if op == "extract":
        return extract_auto(kwargs["source"], kwargs["dest"])
    raise ValueError(f"Unsupported archive op: {op}")
