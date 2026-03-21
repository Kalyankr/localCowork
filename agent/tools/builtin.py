"""Built-in tool plugins: shell, python, web_search, fetch_webpage.

Each tool class satisfies the ToolPlugin protocol and encapsulates
only the core execution logic.  Safety checking and error sanitisation
remain in the agent layer.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog

from agent.config import settings
from agent.sandbox.sandbox_runner import Sandbox
from agent.web import fetch_webpage as _fetch_webpage
from agent.web import web_search as _web_search

logger = structlog.get_logger(__name__)


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
    """Register the four built-in tools in the global registry."""
    from agent.tools.registry import tool_registry

    tool_registry.register(ShellTool())
    tool_registry.register(PythonTool(sandbox))
    tool_registry.register(WebSearchTool())
    tool_registry.register(FetchWebpageTool())
