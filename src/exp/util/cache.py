import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional, Tuple

CACHE_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "cache" / "api_cache.sqlite"


class DiskCache:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or CACHE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    request_hash TEXT PRIMARY KEY,
                    response_text TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @staticmethod
    def compute_hash(messages: Any, schema: Any, model: str) -> str:
        req_repr = json.dumps({"messages": messages, "schema": schema, "model": model}, sort_keys=True)
        return hashlib.sha256(req_repr.encode("utf-8")).hexdigest()

    def get(self, request_hash: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT response_text, usage_json FROM api_cache WHERE request_hash = ?", (request_hash,))
            row = cur.fetchone()
            if row:
                return row[0], json.loads(row[1])
        return None

    def set(self, request_hash: str, response_text: str, usage: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (request_hash, response_text, usage_json) VALUES (?, ?, ?)",
                (request_hash, response_text, json.dumps(usage))
            )
