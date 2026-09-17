"""Local SQLite-backed cache for UniProt / STRING-DB API responses.

Avoids redundant network calls by persisting JSON-serializable payloads
keyed by (namespace, key) under ./cache/phasenet_cache.db.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DB = CACHE_DIR / "phasenet_cache.db"

_lock = threading.Lock()


class ResponseCache:
    """Thread-safe SQLite cache with an optional TTL (seconds)."""

    def __init__(self, db_path: Path = CACHE_DB, ttl: Optional[float] = None):
        self.db_path = db_path
        self.ttl = ttl
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with _lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            conn.commit()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        with _lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value, created_at FROM cache_entries WHERE namespace=? AND key=?",
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        value, created_at = row
        if self.ttl is not None and (time.time() - created_at) > self.ttl:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with _lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (namespace, key, value, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value=excluded.value, created_at=excluded.created_at
                """,
                (namespace, key, payload, time.time()),
            )
            conn.commit()

    def clear(self, namespace: Optional[str] = None) -> None:
        with _lock, self._connect() as conn:
            if namespace is None:
                conn.execute("DELETE FROM cache_entries")
            else:
                conn.execute("DELETE FROM cache_entries WHERE namespace=?", (namespace,))
            conn.commit()


# Module-level singleton used across the app.
cache = ResponseCache()
