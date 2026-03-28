"""Built-in tool plugins: shell, python, web_search, fetch_webpage, read_file, write_file, edit_file, memory_store, memory_recall.

Each tool class satisfies the ToolPlugin protocol and encapsulates
only the core execution logic.  Safety checking and error sanitisation
remain in the agent layer.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import structlog

from agent.config import settings
from agent.sandbox.sandbox_runner import Sandbox
from agent.web import fetch_webpage as _fetch_webpage
from agent.web import web_search as _web_search

logger = structlog.get_logger(__name__)

# Maximum file size we'll read (10 MB)
_MAX_READ_SIZE = 10 * 1024 * 1024
# Maximum file size we'll write (50 MB)
_MAX_WRITE_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Shell tool
# ---------------------------------------------------------------------------


class ShellTool:
    """Execute bash commands via asyncio subprocess."""

    name = "shell"
    description = "Run bash commands"
    args_schema = {"command": "bash command to run"}

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        command = args.get("command", "")
        cwd = args.get("cwd")

        cwd = os.path.expanduser(cwd) if cwd else os.path.expanduser("~")
        command = command.replace("~/", os.path.expanduser("~") + "/")

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env={**os.environ, "HOME": os.path.expanduser("~")},
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=settings.shell_timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "status": "error",
                "output": None,
                "error": f"Command timed out after {settings.shell_timeout} seconds",
            }

        output = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        if proc.returncode != 0:
            raw_error = (
                f"Exit {proc.returncode}: {stderr}"
                if stderr
                else f"Exit {proc.returncode}"
            )
            return {"status": "error", "output": None, "error": raw_error}

        return {"status": "success", "output": output.strip() or "(no output)"}


# ---------------------------------------------------------------------------
# Python tool
# ---------------------------------------------------------------------------


class PythonTool:
    """Execute Python code inside a sandboxed runner."""

    name = "python"
    description = "Run Python code (pandas, requests, openpyxl, etc. available)"
    args_schema = {"code": "Python code to execute"}

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        code = args.get("code", "")
        result = await self.sandbox.run_python(code)

        if result.get("error"):
            return {"status": "error", "output": None, "error": result["error"]}

        return {"status": "success", "output": result.get("output", "")}


# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------


class WebSearchTool:
    """Search the web using DuckDuckGo."""

    name = "web_search"
    description = "Search the web (DuckDuckGo)"
    args_schema = {"query": "search terms", "max_results": "5"}

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        query = args.get("query", "")
        max_results = args.get("max_results", 5)
        result = _web_search(query, max_results=max_results)

        if result.get("error"):
            return {"status": "error", "output": None, "error": result["error"]}

        return {"status": "success", "output": result}


# ---------------------------------------------------------------------------
# Fetch webpage tool
# ---------------------------------------------------------------------------


class FetchWebpageTool:
    """Fetch and extract text content from a URL."""

    name = "fetch_webpage"
    description = "Fetch and extract text from a URL"
    args_schema = {"url": "https://..."}

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        url = args.get("url", "")
        result = _fetch_webpage(url)

        if result.get("error"):
            return {"status": "error", "output": None, "error": result["error"]}

        return {"status": "success", "output": result}


def register_builtin_tools(sandbox: Sandbox) -> None:
    """Register all built-in tools in the global registry."""
    from agent.tools.registry import tool_registry

    tool_registry.register(ShellTool())
    tool_registry.register(PythonTool(sandbox))
    tool_registry.register(WebSearchTool())
    tool_registry.register(FetchWebpageTool())
    tool_registry.register(ReadFileTool())
    tool_registry.register(WriteFileTool())
    tool_registry.register(EditFileTool())
    tool_registry.register(MemoryStoreTool())
    tool_registry.register(MemoryRecallTool())
    tool_registry.register(ListDirTool())


# ---------------------------------------------------------------------------
# Shared file helpers
# ---------------------------------------------------------------------------


def _is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Detect binary content by checking for null bytes in a sample."""
    return b"\x00" in data[:sample_size]


def _resolve_path(raw_path: str) -> Path:
    """Expand ~, resolve relative paths against CWD, and return absolute."""
    return Path(os.path.expanduser(raw_path)).resolve()


# ---------------------------------------------------------------------------
# Read file tool
# ---------------------------------------------------------------------------


class ListDirTool:
    """List directory contents with metadata (size, type) as structured JSON."""

    name = "list_dir"
    description = "List files and directories at a path. Returns structured JSON with names, types, and sizes."
    args_schema = {
        "path": "directory path (absolute or relative)",
        "pattern": "(optional) glob pattern to filter, e.g. '*.py'",
        "recursive": "(optional) true to recurse into subdirectories",
    }

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        raw_path = args.get("path", ".")
        pattern = args.get("pattern", "")
        recursive = str(args.get("recursive", "false")).lower() in ("true", "1", "yes")

        dirpath = _resolve_path(raw_path)
        if not dirpath.is_dir():
            return {
                "status": "error",
                "output": None,
                "error": f"Not a directory: {dirpath}",
            }

        entries: list[dict[str, Any]] = []
        max_entries = 1000  # Safety cap

        try:
            if pattern:
                glob_method = dirpath.rglob if recursive else dirpath.glob
                items = glob_method(pattern)
            elif recursive:
                items = dirpath.rglob("*")
            else:
                items = dirpath.iterdir()

            for item in items:
                if len(entries) >= max_entries:
                    break
                try:
                    stat = item.stat()
                    entries.append(
                        {
                            "name": str(item.relative_to(dirpath)),
                            "type": "dir" if item.is_dir() else "file",
                            "size": stat.st_size if item.is_file() else None,
                        }
                    )
                except (PermissionError, OSError):
                    entries.append(
                        {
                            "name": str(item.relative_to(dirpath)),
                            "type": "unknown",
                            "size": None,
                        }
                    )

            entries.sort(key=lambda e: (e["type"] != "dir", e["name"]))

        except PermissionError:
            return {
                "status": "error",
                "output": None,
                "error": f"Permission denied: {dirpath}",
            }

        return {
            "status": "success",
            "output": {
                "path": str(dirpath),
                "count": len(entries),
                "truncated": len(entries) >= max_entries,
                "entries": entries,
            },
        }


# ---------------------------------------------------------------------------
# Read file tool
# ---------------------------------------------------------------------------


class ReadFileTool:
    """Read file contents with encoding detection and line-range support."""

    name = "read_file"
    description = "Read a file's contents (text). Supports line ranges for large files."
    args_schema = {
        "path": "file path (absolute or relative)",
        "start_line": "(optional) 1-based start line",
        "end_line": "(optional) 1-based end line (inclusive)",
        "encoding": "(optional) encoding, default utf-8",
    }

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        raw_path = args.get("path", "")
        if not raw_path:
            return {"status": "error", "output": None, "error": "No path provided"}

        filepath = _resolve_path(raw_path)

        if not filepath.is_file():
            return {
                "status": "error",
                "output": None,
                "error": f"File not found: {filepath}",
            }

        # Size guard
        size = filepath.stat().st_size
        if size > _MAX_READ_SIZE:
            return {
                "status": "error",
                "output": None,
                "error": f"File too large ({size:,} bytes, max {_MAX_READ_SIZE:,}). "
                "Use start_line/end_line to read a portion.",
            }

        # Binary detection
        try:
            raw = filepath.read_bytes()
        except PermissionError:
            return {
                "status": "error",
                "output": None,
                "error": f"Permission denied: {filepath}",
            }

        if _is_binary(raw):
            return {
                "status": "success",
                "output": (
                    f"[binary file: {filepath.name}, {size:,} bytes, "
                    f"type={filepath.suffix or 'unknown'}]"
                ),
            }

        # Decode
        encoding = args.get("encoding", "utf-8")
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            try:
                text = raw.decode("latin-1")
            except Exception:
                return {
                    "status": "error",
                    "output": None,
                    "error": f"Cannot decode {filepath} with {encoding} or latin-1",
                }

        # Line-range slicing
        start = args.get("start_line")
        end = args.get("end_line")
        if start is not None or end is not None:
            lines = text.splitlines(keepends=True)
            total = len(lines)
            s = max(1, int(start or 1)) - 1
            e = min(total, int(end or total))
            text = "".join(lines[s:e])
            return {
                "status": "success",
                "output": text,
                "metadata": {"total_lines": total, "showing": f"{s + 1}-{e}"},
            }

        return {"status": "success", "output": text}


# ---------------------------------------------------------------------------
# Write file tool
# ---------------------------------------------------------------------------


class WriteFileTool:
    """Write content to a file atomically. Creates parent directories."""

    name = "write_file"
    description = "Write text to a file (atomic). Creates directories as needed."
    args_schema = {
        "path": "file path (absolute or relative)",
        "content": "text content to write",
        "encoding": "(optional) encoding, default utf-8",
    }

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        raw_path = args.get("path", "")
        content = args.get("content", "")
        if not raw_path:
            return {"status": "error", "output": None, "error": "No path provided"}

        filepath = _resolve_path(raw_path)
        encoding = args.get("encoding", "utf-8")

        # Size guard
        encoded = content.encode(encoding)
        if len(encoded) > _MAX_WRITE_SIZE:
            return {
                "status": "error",
                "output": None,
                "error": f"Content too large ({len(encoded):,} bytes, "
                f"max {_MAX_WRITE_SIZE:,})",
            }

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to temp file in same dir, then rename
            fd, tmp_path = tempfile.mkstemp(dir=str(filepath.parent), suffix=".tmp")
            try:
                os.write(fd, encoded)
                os.fsync(fd)
                os.close(fd)
                os.replace(tmp_path, str(filepath))
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

        except PermissionError:
            return {
                "status": "error",
                "output": None,
                "error": f"Permission denied: {filepath}",
            }
        except OSError as e:
            return {"status": "error", "output": None, "error": str(e)}

        return {
            "status": "success",
            "output": f"Wrote {len(encoded):,} bytes to {filepath}",
        }


# ---------------------------------------------------------------------------
# Edit file tool (surgical string replacement)
# ---------------------------------------------------------------------------


class EditFileTool:
    """Replace a specific string in a file. Safer than shell-based sed."""

    name = "edit_file"
    description = "Replace an exact string in a file (find-and-replace in place)."
    args_schema = {
        "path": "file path",
        "old_string": "exact text to find (must match once)",
        "new_string": "replacement text",
    }

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        raw_path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")

        if not raw_path:
            return {"status": "error", "output": None, "error": "No path provided"}
        if not old_string:
            return {
                "status": "error",
                "output": None,
                "error": "old_string is required",
            }

        filepath = _resolve_path(raw_path)

        if not filepath.is_file():
            return {
                "status": "error",
                "output": None,
                "error": f"File not found: {filepath}",
            }

        try:
            text = filepath.read_text(encoding="utf-8")
        except PermissionError:
            return {
                "status": "error",
                "output": None,
                "error": f"Permission denied: {filepath}",
            }

        count = text.count(old_string)
        if count == 0:
            return {
                "status": "error",
                "output": None,
                "error": "old_string not found in file",
            }
        if count > 1:
            return {
                "status": "error",
                "output": None,
                "error": f"old_string matches {count} times — must match exactly once. "
                "Include more surrounding context to make it unique.",
            }

        new_text = text.replace(old_string, new_string, 1)

        # Atomic write
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(filepath.parent), suffix=".tmp")
            try:
                os.write(fd, new_text.encode("utf-8"))
                os.fsync(fd)
                os.close(fd)
                os.replace(tmp_path, str(filepath))
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except PermissionError:
            return {
                "status": "error",
                "output": None,
                "error": f"Permission denied writing to: {filepath}",
            }

        return {
            "status": "success",
            "output": f"Replaced 1 occurrence in {filepath}",
        }


# ---------------------------------------------------------------------------
# Memory store tool
# ---------------------------------------------------------------------------


class MemoryStoreTool:
    """Store a key/value fact in persistent memory (survives across sessions)."""

    name = "memory_store"
    description = (
        "Remember a fact for future sessions. "
        "Use to store user preferences, project context, or key learnings."
    )
    args_schema = {
        "key": "short unique identifier (e.g. 'project_language')",
        "value": "the fact to remember",
        "category": "(optional) category: general, preference, project, tool (default: general)",
    }

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        key = args.get("key", "").strip()
        value = args.get("value", "").strip()
        category = args.get("category", "general").strip()

        if not key:
            return {"status": "error", "output": None, "error": "key is required"}
        if not value:
            return {"status": "error", "output": None, "error": "value is required"}
        if category not in ("general", "preference", "project", "tool"):
            category = "general"

        from agent.orchestrator.database import get_database

        try:
            db = await get_database()
            await db.store_memory(key, value, category)
        except Exception as e:
            return {"status": "error", "output": None, "error": f"DB error: {e}"}

        return {
            "status": "success",
            "output": f"Remembered [{category}] {key} = {value}",
        }


# ---------------------------------------------------------------------------
# Memory recall tool
# ---------------------------------------------------------------------------


class MemoryRecallTool:
    """Search or list facts from persistent memory."""

    name = "memory_recall"
    description = (
        "Recall stored facts from memory. "
        "Search by keyword or list all memories in a category."
    )
    args_schema = {
        "query": "(optional) search term to find relevant memories",
        "key": "(optional) exact key to retrieve",
        "category": "(optional) filter by category: general, preference, project, tool",
        "limit": "(optional) max results, default 10",
    }

    async def execute(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        query = args.get("query", "").strip()
        key = args.get("key", "").strip()
        category = args.get("category", "").strip() or None
        limit = int(args.get("limit", 10))

        from agent.orchestrator.database import get_database

        try:
            db = await get_database()

            if key:
                mem = await db.get_memory(key)
                if mem:
                    return {
                        "status": "success",
                        "output": f"{mem['key']}: {mem['value']}",
                    }
                return {
                    "status": "success",
                    "output": f"No memory found for key '{key}'",
                }

            if query:
                results = await db.search_memories(query, limit=limit)
            else:
                results = await db.list_memories(category=category, limit=limit)

            if not results:
                return {"status": "success", "output": "No memories found."}

            lines = [f"- [{r['category']}] {r['key']}: {r['value']}" for r in results]
            return {"status": "success", "output": "\n".join(lines)}

        except Exception as e:
            return {"status": "error", "output": None, "error": f"DB error: {e}"}
