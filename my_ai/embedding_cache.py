"""Persistent embedding cache using SQLite.

This module provides a two-tier caching strategy:
1. In-memory "warm" cache for the current process lifetime
2. Persistent SQLite cache across process restarts

The cache stores embeddings keyed by model name and text content,
reducing redundant API calls to embedding services.
"""

import hashlib
import os
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np


class EmbeddingCache:
    """Persistent embedding cache with SQLite backend."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the embedding cache.

        Args:
            db_path: Path to SQLite database file. If None, uses
                    EMBEDDING_CACHE_DB env var or defaults to
                    'data/embeddings_cache.db'.
        """
        if db_path is None:
            db_path = os.getenv("EMBEDDING_CACHE_DB", "data/embeddings_cache.db")

        # Ensure directory exists
        if not os.path.isabs(db_path):
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self._warm_cache: dict[str, np.ndarray] = {}
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema for persistent cache."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _make_key(self, model: str, text: str) -> str:
        """Generate cache key from model and text.

        Args:
            model: Model name
            text: Input text (will be normalized)

        Returns:
            SHA256 hash of normalized "model::text"
        """
        normalized = f"{model}::{text.strip().lower()}"
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, model: str, text: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache.

        Checks warm cache first, then persistent cache.

        Args:
            model: Model name
            text: Input text

        Returns:
            Cached embedding as numpy array, or None if not found
        """
        key = self._make_key(model, text)

        # Check warm cache first
        if key in self._warm_cache:
            return self._warm_cache[key]

        # Check persistent cache
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute(
            "SELECT embedding, dim, dtype FROM embeddings WHERE key = ?",
            (key,)
        )
        row = c.fetchone()
        conn.close()

        if row is None:
            return None

        # Reconstruct numpy array from stored bytes
        embedding_bytes, dim, dtype = row
        embedding = np.frombuffer(embedding_bytes, dtype=dtype).reshape(dim)

        # Return a copy to avoid buffer issues
        embedding = embedding.copy()

        # Populate warm cache
        self._warm_cache[key] = embedding

        return embedding

    def set(self, model: str, text: str, embedding: np.ndarray):
        """Store embedding in both warm and persistent cache.

        Args:
            model: Model name
            text: Input text
            embedding: Numpy array to cache
        """
        key = self._make_key(model, text)

        # Store in warm cache
        self._warm_cache[key] = embedding

        # Store in persistent cache
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        text_hash = hashlib.sha256(text.strip().lower().encode()).hexdigest()

        # Replace if exists
        c.execute("""
            INSERT OR REPLACE INTO embeddings 
            (key, model, text_hash, embedding, dim, dtype, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            model,
            text_hash,
            embedding.tobytes(),
            embedding.shape[0],
            str(embedding.dtype),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def clear_warm_cache(self):
        """Clear in-memory warm cache only."""
        self._warm_cache.clear()

    def clear(self):
        """Clear both warm cache and persistent cache."""
        self._warm_cache.clear()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()

    def size(self) -> tuple[int, int]:
        """Get cache sizes.

        Returns:
            Tuple of (warm_cache_size, persistent_cache_size)
        """
        warm_size = len(self._warm_cache)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM embeddings")
        persistent_size = c.fetchone()[0]
        conn.close()

        return (warm_size, persistent_size)


# Global singleton pattern
_global_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache singleton.

    Returns:
        Global EmbeddingCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache
