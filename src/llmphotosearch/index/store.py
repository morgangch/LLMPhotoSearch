"""Persistent embedding cache."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from platformdirs import user_cache_path


def default_db_path() -> Path:
    return user_cache_path("llmphotosearch", appauthor=False) / "index.sqlite"


class EmbeddingStore:
    """SQLite table of embeddings keyed by (model, path), invalidated on mtime/size change."""

    def __init__(self, db_path: Path | None = None):
        db_path = db_path or default_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                model TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model, path)
            )"""
        )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_fresh(self, model: str, paths: Iterable[Path]) -> dict[Path, np.ndarray]:
        """Cached embeddings for the given files that are still up to date."""
        fresh = {}
        for path in paths:
            row = self._db.execute(
                "SELECT mtime_ns, size, vector FROM embeddings WHERE model = ? AND path = ?",
                (model, str(path)),
            ).fetchone()
            if row is None:
                continue
            stat = path.stat()
            if (row[0], row[1]) == (stat.st_mtime_ns, stat.st_size):
                fresh[path] = np.frombuffer(row[2], dtype=np.float32)
        return fresh

    def put(self, model: str, items: Iterable[tuple[Path, np.ndarray]]) -> None:
        rows = []
        for path, vector in items:
            stat = path.stat()
            rows.append(
                (
                    model,
                    str(path),
                    stat.st_mtime_ns,
                    stat.st_size,
                    np.asarray(vector, dtype=np.float32).tobytes(),
                )
            )
        with self._db:
            self._db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)", rows)
