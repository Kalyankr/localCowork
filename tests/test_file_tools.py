"""Tests for read_file, write_file, and edit_file tool plugins."""

import os

import pytest

from agent.tools.builtin import (
    EditFileTool,
    ReadFileTool,
    WriteFileTool,
    _is_binary,
    _resolve_path,
)

# ── Helper ────────────────────────────────────────────────────────────


@pytest.fixture
def read_tool():
    return ReadFileTool()


@pytest.fixture
def write_tool():
    return WriteFileTool()


@pytest.fixture
def edit_tool():
    return EditFileTool()


CTX: dict = {}


# ── _is_binary / _resolve_path ────────────────────────────────────────


class TestHelpers:
    def test_binary_detection_true(self):
        assert _is_binary(b"\x89PNG\r\n\x1a\n\x00\x00") is True

    def test_binary_detection_false(self):
        assert _is_binary(b"Hello, world!") is False

    def test_binary_empty(self):
        assert _is_binary(b"") is False

    def test_resolve_path_tilde(self):
        p = _resolve_path("~/test.txt")
        assert str(p).startswith(os.path.expanduser("~"))
        assert p.is_absolute()

    def test_resolve_path_relative(self):
        p = _resolve_path("foo/bar.txt")
        assert p.is_absolute()


# ── ReadFileTool ──────────────────────────────────────────────────────


class TestReadFileTool:
    @pytest.mark.asyncio
    async def test_read_text_file(self, tmp_path, read_tool):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, World!\nLine 2\n")
        result = await read_tool.execute({"path": str(f)}, CTX)
        assert result["status"] == "success"
        assert "Hello, World!" in result["output"]
        assert "Line 2" in result["output"]

    @pytest.mark.asyncio
    async def test_read_with_line_range(self, tmp_path, read_tool):
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = await read_tool.execute(
            {"path": str(f), "start_line": 2, "end_line": 4}, CTX
        )
        assert result["status"] == "success"
        assert "line2" in result["output"]
        assert "line4" in result["output"]
        assert "line1" not in result["output"]
        assert "line5" not in result["output"]
        assert result["metadata"]["total_lines"] == 5

    @pytest.mark.asyncio
    async def test_read_binary_file(self, tmp_path, read_tool):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
        result = await read_tool.execute({"path": str(f)}, CTX)
        assert result["status"] == "success"
        assert "binary file" in result["output"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, read_tool):
        result = await read_tool.execute({"path": "/nonexistent/file.txt"}, CTX)
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_read_no_path(self, read_tool):
        result = await read_tool.execute({}, CTX)
        assert result["status"] == "error"
        assert "No path" in result["error"]

    @pytest.mark.asyncio
    async def test_read_large_file_rejected(self, tmp_path, read_tool):
        """Files over 10 MB should be rejected."""
        f = tmp_path / "big.txt"
        # Create a file stub that reports large size
        f.write_text("x")
        # Patch the size check threshold
        import agent.tools.builtin as mod

        original = mod._MAX_READ_SIZE
        mod._MAX_READ_SIZE = 5  # 5 bytes
        try:
            result = await read_tool.execute({"path": str(f)}, CTX)
            # File is only 1 byte, should succeed
            assert result["status"] == "success"

            f.write_text("toolarge")  # 8 bytes > 5
            result = await read_tool.execute({"path": str(f)}, CTX)
            assert result["status"] == "error"
            assert "too large" in result["error"].lower()
        finally:
            mod._MAX_READ_SIZE = original

    @pytest.mark.asyncio
    async def test_read_latin1_fallback(self, tmp_path, read_tool):
        """Should fall back to latin-1 if utf-8 fails."""
        f = tmp_path / "latin.txt"
        f.write_bytes("café résumé".encode("latin-1"))
        result = await read_tool.execute({"path": str(f)}, CTX)
        assert result["status"] == "success"
        assert "caf" in result["output"]


# ── WriteFileTool ─────────────────────────────────────────────────────


class TestWriteFileTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path, write_tool):
        f = tmp_path / "new.txt"
        result = await write_tool.execute({"path": str(f), "content": "Hello!"}, CTX)
        assert result["status"] == "success"
        assert f.read_text() == "Hello!"
        assert "6" in result["output"]  # 6 bytes

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, tmp_path, write_tool):
        f = tmp_path / "a" / "b" / "c" / "file.txt"
        result = await write_tool.execute({"path": str(f), "content": "deep"}, CTX)
        assert result["status"] == "success"
        assert f.read_text() == "deep"

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, tmp_path, write_tool):
        f = tmp_path / "exist.txt"
        f.write_text("old content")
        result = await write_tool.execute(
            {"path": str(f), "content": "new content"}, CTX
        )
        assert result["status"] == "success"
        assert f.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_no_path(self, write_tool):
        result = await write_tool.execute({"content": "hello"}, CTX)
        assert result["status"] == "error"
        assert "No path" in result["error"]

    @pytest.mark.asyncio
    async def test_write_preserves_encoding(self, tmp_path, write_tool):
        f = tmp_path / "unicode.txt"
        content = "日本語テスト 🎉"
        result = await write_tool.execute({"path": str(f), "content": content}, CTX)
        assert result["status"] == "success"
        assert f.read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_write_atomic_no_partial(self, tmp_path, write_tool):
        """If the file existed before, it should remain intact on size-limit error."""
        f = tmp_path / "safe.txt"
        f.write_text("original")

        import agent.tools.builtin as mod

        original = mod._MAX_WRITE_SIZE
        mod._MAX_WRITE_SIZE = 5  # 5 bytes
        try:
            result = await write_tool.execute(
                {"path": str(f), "content": "toolarge"}, CTX
            )
            assert result["status"] == "error"
            # Original file should be untouched
            assert f.read_text() == "original"
        finally:
            mod._MAX_WRITE_SIZE = original

    @pytest.mark.asyncio
    async def test_write_special_chars_no_shell_expansion(self, tmp_path, write_tool):
        """Content with $, backticks, etc. should be written verbatim."""
        f = tmp_path / "special.sh"
        content = 'echo "$HOME"\nVAR=`whoami`\n'
        result = await write_tool.execute({"path": str(f), "content": content}, CTX)
        assert result["status"] == "success"
        assert f.read_text() == content  # No shell expansion


# ── EditFileTool ──────────────────────────────────────────────────────


class TestEditFileTool:
    @pytest.mark.asyncio
    async def test_edit_single_replacement(self, tmp_path, edit_tool):
        f = tmp_path / "config.yaml"
        f.write_text("host: localhost\nport: 8080\ndebug: true\n")
        result = await edit_tool.execute(
            {"path": str(f), "old_string": "port: 8080", "new_string": "port: 9090"},
            CTX,
        )
        assert result["status"] == "success"
        assert "Replaced 1" in result["output"]
        assert "port: 9090" in f.read_text()
        assert "port: 8080" not in f.read_text()

    @pytest.mark.asyncio
    async def test_edit_not_found(self, tmp_path, edit_tool):
        f = tmp_path / "file.txt"
        f.write_text("hello world")
        result = await edit_tool.execute(
            {"path": str(f), "old_string": "nonexistent", "new_string": "x"}, CTX
        )
        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_multiple_matches_rejected(self, tmp_path, edit_tool):
        f = tmp_path / "file.txt"
        f.write_text("foo bar foo baz foo")
        result = await edit_tool.execute(
            {"path": str(f), "old_string": "foo", "new_string": "qux"}, CTX
        )
        assert result["status"] == "error"
        assert "3 times" in result["error"]
        # File should be unchanged
        assert f.read_text() == "foo bar foo baz foo"

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, edit_tool):
        result = await edit_tool.execute(
            {
                "path": "/nonexistent/file.txt",
                "old_string": "a",
                "new_string": "b",
            },
            CTX,
        )
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_edit_no_path(self, edit_tool):
        result = await edit_tool.execute({"old_string": "a", "new_string": "b"}, CTX)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_edit_empty_old_string(self, tmp_path, edit_tool):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = await edit_tool.execute(
            {"path": str(f), "old_string": "", "new_string": "x"}, CTX
        )
        assert result["status"] == "error"
        assert "old_string" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_preserves_rest_of_file(self, tmp_path, edit_tool):
        f = tmp_path / "code.py"
        original = (
            "def hello():\n    print('hello')\n\ndef goodbye():\n    print('bye')\n"
        )
        f.write_text(original)
        result = await edit_tool.execute(
            {
                "path": str(f),
                "old_string": "    print('hello')",
                "new_string": "    print('hi there')",
            },
            CTX,
        )
        assert result["status"] == "success"
        new_text = f.read_text()
        assert "print('hi there')" in new_text
        assert "def goodbye():" in new_text
        assert "print('bye')" in new_text


# ── Registration test ─────────────────────────────────────────────────


class TestRegistration:
    def test_file_tools_registered(self):
        """read_file, write_file, edit_file should be in the registry after
        register_builtin_tools is called."""
        from unittest.mock import MagicMock

        from agent.tools.builtin import register_builtin_tools
        from agent.tools.registry import ToolRegistry

        reg = ToolRegistry()
        # Temporarily swap the global registry
        import agent.tools.builtin as mod

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod, "tool_registry", reg, raising=False)
            # Need to patch at the import location
            import agent.tools.registry as reg_mod

            mp.setattr(reg_mod, "tool_registry", reg)
            register_builtin_tools(MagicMock())

        # Check directly on reg (we registered into it)
        names = reg.get_tool_names()
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names

    def test_tool_descriptions_include_file_tools(self):
        """The prompt tool descriptions should mention the file tools."""
        from unittest.mock import MagicMock

        from agent.tools.builtin import register_builtin_tools
        from agent.tools.registry import ToolRegistry

        reg = ToolRegistry()
        import agent.tools.builtin as mod
        import agent.tools.registry as reg_mod

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod, "tool_registry", reg, raising=False)
            mp.setattr(reg_mod, "tool_registry", reg)
            register_builtin_tools(MagicMock())

        desc = reg.get_tool_descriptions()
        assert "read_file" in desc
        assert "write_file" in desc
        assert "edit_file" in desc
        assert "atomic" in desc.lower()
