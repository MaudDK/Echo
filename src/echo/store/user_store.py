import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bcrypt

logger = logging.getLogger(__name__)

class UserExistsError(Exception):
    """Raised when registering a username that is already taken."""

@dataclass
class User:
    id: int
    username: str

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

class UserStore:
    def __init__(self, db_path: str, token_ttl_seconds: int = 86400) -> None:
        self.db_path = db_path
        self.token_ttl_seconds = token_ttl_seconds
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
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token      TEXT PRIMARY KEY,
                    user_id    INTEGER NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )

    def register(self, username: str, password: str) -> User:
        if not username or not password:
            raise ValueError("username and password are required")
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, hash_password(password)),
                )
                user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            raise UserExistsError(f"username '{username}' is already taken")
        logger.info(f"Registered user '{username}' (id={user_id})")
        return User(id=user_id, username=username)

    def login(self, username: str, password: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row is None or not verify_password(password, row["password_hash"]):
                return None
            token = secrets.token_urlsafe(64)
            expires_at = time.time() + self.token_ttl_seconds
            conn.execute(
                "INSERT INTO tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, row["id"], expires_at),
            )
        return token

    def resolve_token(self, token: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT t.expires_at, u.id, u.username
                FROM tokens t JOIN users u ON u.id = t.user_id
                WHERE t.token = ?
                """,
                (token,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= time.time():
                conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
                return None
            return User(id=row["id"], username=row["username"])

    def logout(self, token: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
        return cur.rowcount > 0

    def logout_all(self, user_id: int) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM tokens WHERE user_id = ?", (user_id,))
        return cur.rowcount

    def purge_expired_tokens(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM tokens WHERE expires_at <= ?", (time.time(),))
        if cur.rowcount:
            logger.info(f"Purged {cur.rowcount} expired token(s)")
        return cur.rowcount
