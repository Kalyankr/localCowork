"""SQLite persistence layer for tasks and conversation history.

Provides async database access via aiosqlite, replacing JSON file storage
for tasks and in-memory storage for conversations.
"""

import json
import sqlite3
from pathlib import Path

import aiosqlite
import structlog

from agent.config import settings

logger = structlog.get_logger(__name__)

# Schema version for future migrations
SCHEMA_VERSION = 2

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    request TEXT NOT NULL,
    session_id TEXT,
    state TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    plan TEXT,           -- JSON
    step_results TEXT,   -- JSON
    current_step TEXT,
    summary TEXT,
    error TEXT,
    workspace_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);

-- Agent memories: persistent key/value facts across sessions
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
"""

# FTS5 virtual table (created separately — not idempotent with executescript)
_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(key, value, category, content=memories, content_rowid=id);
"""


class Database:
    """Async SQLite database for task and conversation persistence."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.db_path
        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database and create tables if needed."""
        db_file = Path(self._db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent read performance
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA foreign_keys=ON")

        await self._connection.executescript(_SCHEMA_SQL)
        # FTS5 virtual table (must be separate from executescript)
        await self._connection.execute(_FTS_SQL)

        # Check / set schema version
        async with self._connection.execute(
            "SELECT version FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await self._connection.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        await self._connection.commit()
        logger.info("database_initialized", path=self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    @property
    def conn(self) -> aiosqlite.Connection:
        """Get the active connection, raising if not initialized."""
        if self._connection is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._connection

    # ── Task operations ──────────────────────────────────────────────

    async def save_task(self, task_data: dict) -> None:
        """Insert or replace a task row."""
        await self.conn.execute(
            """INSERT OR REPLACE INTO tasks
               (id, request, session_id, state, created_at, updated_at,
                plan, step_results, current_step, summary, error, workspace_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_data["id"],
                task_data["request"],
                task_data.get("session_id"),
                task_data["state"],
                task_data["created_at"],
                task_data["updated_at"],
                json.dumps(task_data.get("plan")) if task_data.get("plan") else None,
                json.dumps(task_data.get("step_results", {})),
                task_data.get("current_step"),
                task_data.get("summary"),
                task_data.get("error"),
                task_data.get("workspace_path"),
            ),
        )
        await self.conn.commit()

    async def load_tasks(self) -> list[dict]:
        """Load all tasks from the database."""
        async with self.conn.execute("SELECT * FROM tasks") as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_task(row) for row in rows]

    async def delete_task(self, task_id: str) -> None:
        """Delete a task by ID."""
        await self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await self.conn.commit()

    @staticmethod
    def _row_to_task(row: aiosqlite.Row) -> dict:
        """Convert a database row to a task dict."""
        data = dict(row)
        # Deserialize JSON fields
        if data.get("plan"):
            data["plan"] = json.loads(data["plan"])
        if data.get("step_results"):
            data["step_results"] = json.loads(data["step_results"])
        else:
            data["step_results"] = {}
        return data

    # ── Conversation operations ──────────────────────────────────────

    async def add_conversation_message(
        self, session_id: str, role: str, content: str, timestamp: float
    ) -> None:
        """Add a message to conversation history."""
        await self.conn.execute(
            """INSERT INTO conversations (session_id, role, content, timestamp)
               VALUES (?, ?, ?, ?)""",
            (session_id, role, content, timestamp),
        )
        await self.conn.commit()

    async def get_conversation_history(
        self, session_id: str, limit: int | None = None
    ) -> list[dict]:
        """Get conversation messages for a session, ordered chronologically."""
        if limit:
            async with self.conn.execute(
                """SELECT role, content FROM conversations
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
            # Reverse to get chronological order
            return [dict(r) for r in reversed(rows)]
        else:
            async with self.conn.execute(
                """SELECT role, content FROM conversations
                   WHERE session_id = ?
                   ORDER BY id ASC""",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_conversation_timestamp(self, session_id: str) -> float | None:
        """Get the latest message timestamp for a session."""
        async with self.conn.execute(
            "SELECT MAX(timestamp) FROM conversations WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row and row[0] is not None:
            return row[0]
        return None

    async def delete_session_conversations(self, session_id: str) -> None:
        """Delete all conversation messages for a session."""
        await self.conn.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        await self.conn.commit()

    async def cleanup_expired_sessions(self, timeout: float) -> list[str]:
        """Delete conversations older than timeout seconds. Returns deleted session IDs."""
        import time

        cutoff = time.time() - timeout
        # Find sessions whose latest message is older than cutoff
        async with self.conn.execute(
            """SELECT session_id FROM conversations
               GROUP BY session_id
               HAVING MAX(timestamp) < ?""",
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()

        expired = [row[0] for row in rows]
        if expired:
            placeholders = ",".join("?" * len(expired))
            await self.conn.execute(
                f"DELETE FROM conversations WHERE session_id IN ({placeholders})",
                expired,
            )
            await self.conn.commit()
        return expired

    async def trim_conversation(self, session_id: str, max_messages: int) -> None:
        """Keep only the most recent max_messages for a session."""
        await self.conn.execute(
            """DELETE FROM conversations WHERE id NOT IN (
                   SELECT id FROM conversations
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?
               ) AND session_id = ?""",
            (session_id, max_messages, session_id),
        )
        await self.conn.commit()

    # ── Memory operations ─────────────────────────────────────────────

    async def store_memory(
        self, key: str, value: str, category: str = "general"
    ) -> None:
        """Store or update a memory fact.

        If a memory with the same key already exists it is updated.
        The FTS index is kept in sync via manual triggers.
        """
        import time as _time

        now = _time.time()
        # Check if key exists
        async with self.conn.execute(
            "SELECT id FROM memories WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()

        if row:
            rid = row[0]
            # Remove old FTS entry, then update row, then re-insert FTS
            await self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (rid,))
            await self.conn.execute(
                "UPDATE memories SET value = ?, category = ?, updated_at = ? WHERE id = ?",
                (value, category, now, rid),
            )
            await self.conn.execute(
                "INSERT INTO memories_fts(rowid, key, value, category) VALUES (?, ?, ?, ?)",
                (rid, key, value, category),
            )
        else:
            await self.conn.execute(
                """INSERT INTO memories (key, value, category, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, value, category, now, now),
            )
            # Get the rowid for FTS
            async with self.conn.execute("SELECT last_insert_rowid()") as cur:
                rid_row = await cur.fetchone()
            rid = rid_row[0]
            await self.conn.execute(
                "INSERT INTO memories_fts(rowid, key, value, category) VALUES (?, ?, ?, ?)",
                (rid, key, value, category),
            )

        await self.conn.commit()
        logger.debug("memory_stored", key=key, category=category)

    async def search_memories(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search over stored memories."""
        async with self.conn.execute(
            """SELECT m.key, m.value, m.category, m.updated_at
               FROM memories_fts f
               JOIN memories m ON f.rowid = m.id
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"key": r[0], "value": r[1], "category": r[2], "updated_at": r[3]}
            for r in rows
        ]

    async def get_memory(self, key: str) -> dict | None:
        """Retrieve a specific memory by exact key."""
        async with self.conn.execute(
            "SELECT key, value, category, updated_at FROM memories WHERE key = ?",
            (key,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return {
                "key": row[0],
                "value": row[1],
                "category": row[2],
                "updated_at": row[3],
            }
        return None

    async def delete_memory(self, key: str) -> bool:
        """Delete a memory by key. Returns True if it existed."""
        async with self.conn.execute(
            "SELECT id FROM memories WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return False
        rid = row[0]
        await self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (rid,))
        await self.conn.execute("DELETE FROM memories WHERE id = ?", (rid,))
        await self.conn.commit()
        return True

    async def list_memories(
        self, category: str | None = None, limit: int = 50
    ) -> list[dict]:
        """List stored memories, optionally filtered by category."""
        if category:
            async with self.conn.execute(
                """SELECT key, value, category, updated_at FROM memories
                   WHERE category = ? ORDER BY updated_at DESC LIMIT ?""",
                (category, limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self.conn.execute(
                """SELECT key, value, category, updated_at FROM memories
                   ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [
            {"key": r[0], "value": r[1], "category": r[2], "updated_at": r[3]}
            for r in rows
        ]


# ── Synchronous helpers (for CLI / non-async callers) ────────────────


def get_sync_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a synchronous SQLite connection for CLI use."""
    path = db_path or settings.db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    conn.execute(_FTS_SQL)
    # Ensure schema version
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
        )
    conn.commit()
    return conn


# ── Singleton accessor ───────────────────────────────────────────────

_db: Database | None = None


async def get_database() -> Database:
    """Get or create the singleton Database instance."""
    global _db
    if _db is None:
        _db = Database()
        await _db.initialize()
    return _db
