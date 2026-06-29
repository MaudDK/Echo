import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class Session:
    id: str
    user_id: int
    name: str
    agent_name: str
    messages: List[Dict[str, Any]]

class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id         TEXT PRIMARY KEY,
                    user_id    INTEGER NOT NULL,
                    name       TEXT NOT NULL DEFAULT 'New Session',
                    agent_name TEXT NOT NULL,
                    messages   TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")

    def create(self, user_id: int, agent_name: str, messages: Optional[List[Dict[str, Any]]] = None) -> str:
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, agent_name, messages) VALUES (?, ?, ?, ?)",
                (session_id, user_id, agent_name, json.dumps(messages or [])),
            )
        logger.info(f"Created session {session_id} for user {user_id} (agent '{agent_name}')")
        return session_id

    def load(self, session_id: str, user_id: int) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, name, agent_name, messages FROM sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            agent_name=row["agent_name"],
            messages=json.loads(row["messages"]),
        )

    def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET messages = ? WHERE id = ?",
                (json.dumps(messages), session_id),
            )

    def set_name(self, session_id: str, name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET name = ? WHERE id = ?",
                (name, session_id),
            )

    def delete(self, session_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0

    def list_for_user(self, user_id: int) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, agent_name FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return [{"id": r["id"], "name": r["name"], "agent_name": r["agent_name"]} for r in rows]
