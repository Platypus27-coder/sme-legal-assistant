"""
VietPhapLy RAG — SQLite retrieval cache.

Thay JSONL + byte-offset index bằng SQLite:
  - O(1) lookup by question_id
  - Dễ debug: query trực tiếp bằng SQL
  - Crash-safe: mỗi INSERT là 1 transaction

Thiết kế crash-safe, O(1) lookup, không lo OOM và đặc biệt dễ debug.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from vpl.settings import CACHE_DIR, RETRIEVAL_DB_FILE


class RetrievalCache:
    """SQLite-backed cache cho retrieval results."""

    def __init__(self, path: Path = RETRIEVAL_DB_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._setup()

    def _setup(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                question_id  INTEGER PRIMARY KEY,
                question     TEXT NOT NULL,
                chunks_json  TEXT NOT NULL,
                cached_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    @property
    def completed_ids(self) -> set[int]:
        rows = self._conn.execute("SELECT question_id FROM retrieval_cache").fetchall()
        return {row[0] for row in rows}

    def append(
        self,
        question_id: int,
        question: str,
        chunks: Iterable[Any],
    ) -> bool:
        """
        Lưu retrieval result. Trả về False nếu đã tồn tại.

        chunks: iterable of ScoredChunk hoặc dict
        """
        question_id = int(question_id)
        existing = self._conn.execute(
            "SELECT 1 FROM retrieval_cache WHERE question_id = ?", (question_id,)
        ).fetchone()
        if existing:
            return False

        serialized = []
        for c in chunks:
            if isinstance(c, dict):
                serialized.append(c)
            else:
                # ScoredChunk
                serialized.append({
                    "chunk_id": c.chunk_id,
                    "text": c.chunk.get("text", ""),
                    "metadata": c.metadata,
                    "score": c.score,
                })

        self._conn.execute(
            "INSERT INTO retrieval_cache (question_id, question, chunks_json) VALUES (?, ?, ?)",
            (question_id, question, json.dumps(serialized, ensure_ascii=False)),
        )
        self._conn.commit()
        return True

    def get(self, question_id: int, question: str | None = None) -> list[dict[str, Any]]:
        """Lấy cached chunks cho một question_id."""
        row = self._conn.execute(
            "SELECT question, chunks_json FROM retrieval_cache WHERE question_id = ?",
            (int(question_id),),
        ).fetchone()
        if not row:
            raise KeyError(f"question_id {question_id} not in cache")
        if question is not None and row[0] != question:
            raise ValueError(f"Question mismatch for id {question_id}")
        return json.loads(row[1])

    def retrieve_by_id(self, question_id: int, question: str | None = None) -> list[dict[str, Any]]:
        """Alias for get(), compatible với pipeline interface."""
        return self.get(question_id, question)

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
