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
SCHEMA_VERSION = 1

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
