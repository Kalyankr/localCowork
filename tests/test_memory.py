"""Tests for the persistent memory system (database + tool plugins)."""

import pytest

from agent.orchestrator.database import Database
from agent.tools.builtin import MemoryRecallTool, MemoryStoreTool

CTX: dict = {}


# ── Database memory operations ────────────────────────────────────────


@pytest.fixture
async def db(tmp_path):
    """Create a temporary in-memory-like database."""
    database = Database(db_path=str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


class TestDatabaseMemory:
    @pytest.mark.asyncio
    async def test_store_and_get(self, db):
        await db.store_memory("lang", "Python 3.12", category="project")
        mem = await db.get_memory("lang")
        assert mem is not None
        assert mem["key"] == "lang"
        assert mem["value"] == "Python 3.12"
        assert mem["category"] == "project"

    @pytest.mark.asyncio
    async def test_store_upsert(self, db):
        await db.store_memory("framework", "Django")
        await db.store_memory("framework", "FastAPI")
        mem = await db.get_memory("framework")
        assert mem["value"] == "FastAPI"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, db):
        mem = await db.get_memory("nope")
        assert mem is None

    @pytest.mark.asyncio
    async def test_delete_memory(self, db):
        await db.store_memory("tmp", "gone soon")
        assert await db.delete_memory("tmp") is True
        assert await db.get_memory("tmp") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, db):
        assert await db.delete_memory("nope") is False

    @pytest.mark.asyncio
    async def test_list_memories(self, db):
        await db.store_memory("a", "1", category="general")
        await db.store_memory("b", "2", category="project")
        await db.store_memory("c", "3", category="project")
        all_mems = await db.list_memories()
        assert len(all_mems) == 3

    @pytest.mark.asyncio
    async def test_list_memories_by_category(self, db):
        await db.store_memory("a", "1", category="general")
        await db.store_memory("b", "2", category="project")
        project_mems = await db.list_memories(category="project")
        assert len(project_mems) == 1
        assert project_mems[0]["key"] == "b"

    @pytest.mark.asyncio
    async def test_search_memories(self, db):
        await db.store_memory("test_framework", "uses pytest for testing")
        await db.store_memory("language", "Python 3.12")
        results = await db.search_memories("pytest")
        assert len(results) >= 1
        assert any("pytest" in r["value"] for r in results)

    @pytest.mark.asyncio
    async def test_search_no_results(self, db):
        await db.store_memory("x", "y")
        results = await db.search_memories("nonexistentterm123")
        assert results == []

    @pytest.mark.asyncio
    async def test_fts_updated_on_upsert(self, db):
        """After updating a memory, FTS should return the new value."""
        await db.store_memory("key1", "old value about Django")
        await db.store_memory("key1", "new value about FastAPI")
        # Search for the new term
        results = await db.search_memories("FastAPI")
        assert len(results) == 1
        assert results[0]["value"] == "new value about FastAPI"
        # Old term should not match
        results_old = await db.search_memories("Django")
        assert len(results_old) == 0


# ── Memory tool plugins ──────────────────────────────────────────────


@pytest.fixture
def store_tool():
    return MemoryStoreTool()


@pytest.fixture
def recall_tool():
    return MemoryRecallTool()


class TestMemoryStoreTool:
    @pytest.mark.asyncio
    async def test_store_basic(self, tmp_path, store_tool, monkeypatch):
        """Store a memory via the tool plugin."""
        db = Database(db_path=str(tmp_path / "tool_test.db"))
        await db.initialize()

        async def fake_get_db():
            return db

        monkeypatch.setattr(
            "agent.tools.builtin.get_database", fake_get_db, raising=False
        )
        # Patch at the import location inside the tool
        import agent.orchestrator.database as db_mod

        monkeypatch.setattr(db_mod, "_db", db)

        result = await store_tool.execute(
            {"key": "test_key", "value": "test_value", "category": "project"}, CTX
        )
        assert result["status"] == "success"
        assert "test_key" in result["output"]

        # Verify it was stored
        mem = await db.get_memory("test_key")
        assert mem is not None
        assert mem["value"] == "test_value"
        await db.close()

    @pytest.mark.asyncio
    async def test_store_missing_key(self, store_tool):
        result = await store_tool.execute({"value": "v"}, CTX)
        assert result["status"] == "error"
        assert "key" in result["error"]

    @pytest.mark.asyncio
    async def test_store_missing_value(self, store_tool):
        result = await store_tool.execute({"key": "k"}, CTX)
        assert result["status"] == "error"
        assert "value" in result["error"]

    @pytest.mark.asyncio
    async def test_store_invalid_category_fallback(
        self, tmp_path, store_tool, monkeypatch
    ):
        """Invalid category should fall back to 'general'."""
        db = Database(db_path=str(tmp_path / "cat_test.db"))
        await db.initialize()

        import agent.orchestrator.database as db_mod

        monkeypatch.setattr(db_mod, "_db", db)

        result = await store_tool.execute(
            {"key": "k", "value": "v", "category": "bogus"}, CTX
        )
        assert result["status"] == "success"
        mem = await db.get_memory("k")
        assert mem["category"] == "general"
        await db.close()


class TestMemoryRecallTool:
    @pytest.mark.asyncio
    async def test_recall_by_key(self, tmp_path, recall_tool, monkeypatch):
        db = Database(db_path=str(tmp_path / "recall_test.db"))
        await db.initialize()

        import agent.orchestrator.database as db_mod

        monkeypatch.setattr(db_mod, "_db", db)

        await db.store_memory("lang", "Python")
        result = await recall_tool.execute({"key": "lang"}, CTX)
        assert result["status"] == "success"
        assert "Python" in result["output"]
        await db.close()

    @pytest.mark.asyncio
    async def test_recall_by_key_not_found(self, tmp_path, recall_tool, monkeypatch):
        db = Database(db_path=str(tmp_path / "recall_nf.db"))
        await db.initialize()

        import agent.orchestrator.database as db_mod

        monkeypatch.setattr(db_mod, "_db", db)

        result = await recall_tool.execute({"key": "nope"}, CTX)
        assert result["status"] == "success"
        assert "No memory" in result["output"]
        await db.close()

    @pytest.mark.asyncio
    async def test_recall_search(self, tmp_path, recall_tool, monkeypatch):
        db = Database(db_path=str(tmp_path / "recall_search.db"))
        await db.initialize()

        import agent.orchestrator.database as db_mod

        monkeypatch.setattr(db_mod, "_db", db)

        await db.store_memory("fw", "uses pytest", category="project")
        result = await recall_tool.execute({"query": "pytest"}, CTX)
        assert result["status"] == "success"
        assert "pytest" in result["output"]
        await db.close()

    @pytest.mark.asyncio
    async def test_recall_list_by_category(self, tmp_path, recall_tool, monkeypatch):
        db = Database(db_path=str(tmp_path / "recall_cat.db"))
        await db.initialize()

        import agent.orchestrator.database as db_mod

        monkeypatch.setattr(db_mod, "_db", db)

        await db.store_memory("a", "1", category="general")
        await db.store_memory("b", "2", category="preference")
        result = await recall_tool.execute({"category": "preference"}, CTX)
        assert result["status"] == "success"
        assert "b" in result["output"]
        assert "a" not in result["output"]
        await db.close()

    @pytest.mark.asyncio
    async def test_recall_empty(self, tmp_path, recall_tool, monkeypatch):
        db = Database(db_path=str(tmp_path / "recall_empty.db"))
        await db.initialize()

        import agent.orchestrator.database as db_mod

        monkeypatch.setattr(db_mod, "_db", db)

        result = await recall_tool.execute({}, CTX)
        assert result["status"] == "success"
        assert "No memories" in result["output"]
        await db.close()


# ── Registration ──────────────────────────────────────────────────────


class TestMemoryToolRegistration:
    def test_memory_tools_in_registry(self):
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

        names = reg.get_tool_names()
        assert "memory_store" in names
        assert "memory_recall" in names

    def test_tool_descriptions_mention_memory(self):
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
        assert "memory_store" in desc
        assert "memory_recall" in desc
