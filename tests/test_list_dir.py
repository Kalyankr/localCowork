"""Tests for the ListDirTool (list_dir)."""

from unittest.mock import MagicMock as MM

import pytest

from agent.tools.builtin import ListDirTool


class TestListDirTool:
    """ListDirTool should list directory contents as structured JSON."""

    @pytest.fixture
    def tool(self):
        return ListDirTool()

    @pytest.fixture
    def tmp_tree(self, tmp_path):
        """Create a small directory tree for testing."""
        (tmp_path / "a.py").write_text("print('a')")
        (tmp_path / "b.txt").write_text("hello")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "c.py").write_text("print('c')")
        (sub / "d.log").write_text("log entry")
        return tmp_path

    @pytest.mark.asyncio
    async def test_basic_listing(self, tool, tmp_tree):
        result = await tool.execute({"path": str(tmp_tree)}, {})
        assert result["status"] == "success"
        output = result["output"]
        names = [e["name"] for e in output["entries"]]
        assert "a.py" in names
        assert "b.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_dirs_sorted_first(self, tool, tmp_tree):
        result = await tool.execute({"path": str(tmp_tree)}, {})
        entries = result["output"]["entries"]
        types = [e["type"] for e in entries]
        dir_indices = [i for i, t in enumerate(types) if t == "dir"]
        file_indices = [i for i, t in enumerate(types) if t == "file"]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)

    @pytest.mark.asyncio
    async def test_glob_pattern(self, tool, tmp_tree):
        result = await tool.execute({"path": str(tmp_tree), "pattern": "*.py"}, {})
        assert result["status"] == "success"
        names = [e["name"] for e in result["output"]["entries"]]
        assert "a.py" in names
        assert "b.txt" not in names

    @pytest.mark.asyncio
    async def test_recursive(self, tool, tmp_tree):
        result = await tool.execute({"path": str(tmp_tree), "recursive": "true"}, {})
        assert result["status"] == "success"
        names = [e["name"] for e in result["output"]["entries"]]
        assert any("c.py" in n for n in names)

    @pytest.mark.asyncio
    async def test_recursive_glob(self, tool, tmp_tree):
        result = await tool.execute(
            {"path": str(tmp_tree), "pattern": "*.py", "recursive": "true"}, {}
        )
        names = [e["name"] for e in result["output"]["entries"]]
        assert any("a.py" in n for n in names)
        assert any("c.py" in n for n in names)
        assert not any(n.endswith(".txt") or n.endswith(".log") for n in names)

    @pytest.mark.asyncio
    async def test_not_a_directory(self, tool, tmp_tree):
        result = await tool.execute({"path": str(tmp_tree / "a.py")}, {})
        assert result["status"] == "error"
        assert "Not a directory" in result["error"]

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tool):
        result = await tool.execute({"path": "/tmp/nonexistent_xyz_abc"}, {})
        assert result["status"] == "error"
        assert "Not a directory" in result["error"]

    @pytest.mark.asyncio
    async def test_entries_have_metadata(self, tool, tmp_tree):
        result = await tool.execute({"path": str(tmp_tree)}, {})
        entries = result["output"]["entries"]
        for e in entries:
            assert "name" in e
            assert "type" in e
            assert "size" in e
            if e["type"] == "file":
                assert isinstance(e["size"], int)
                assert e["size"] >= 0

    @pytest.mark.asyncio
    async def test_count_matches_entries(self, tool, tmp_tree):
        result = await tool.execute({"path": str(tmp_tree)}, {})
        output = result["output"]
        assert output["count"] == len(output["entries"])

    @pytest.mark.asyncio
    async def test_default_path_is_cwd(self, tool):
        result = await tool.execute({}, {})
        assert result["status"] == "success"
        assert result["output"]["count"] > 0

    def test_registered_in_registry(self):
        from agent.tools.builtin import register_builtin_tools
        from agent.tools.registry import tool_registry

        register_builtin_tools(MM())
        assert tool_registry.get("list_dir") is not None
