"""Persistent embedding cache using SQLite to avoid redundant API calls.

This module provides a disk-backed cache for text embeddings, keyed by
model name and normalized text hash. Embeddings are stored as raw bytes
with metadata for reconstruction.
"""
import sqlite3
import hashlib
import os
import numpy as np
from typing import Optional
from datetime import datetime


class EmbeddingCache:
    """SQLite-backed persistent cache for embeddings."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the embedding cache.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/embeddings_cache.db
                    or EMBEDDING_CACHE_DB environment variable.
        """
        if db_path is None:
            db_path = os.getenv("EMBEDDING_CACHE_DB", 
                               os.path.join("data", "embeddings_cache.db"))
        
        # Ensure directory exists
        if not os.path.isabs(db_path):
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the cache database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Performance tuning
        c.execute('PRAGMA journal_mode = WAL')
        c.execute('PRAGMA synchronous = NORMAL')
        
        # Cache table
        c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Index for cleanup/maintenance queries
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_embeddings_created 
            ON embeddings(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def _make_key(self, model: str, text: str) -> str:
        """Generate a stable cache key from model name and normalized text.
        
        Args:
            model: Model name/identifier
            text: Raw text to embed
            
        Returns:
            SHA256 hash as hex string
        """
        # Normalize text: lowercase and strip whitespace
        text_norm = text.lower().strip()
        # Combine model and text for key
        combined = f"{model}::{text_norm}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, model: str, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available.
        
        Args:
            model: Model name used for embedding
            text: Text that was embedded
            
        Returns:
            Cached numpy array or None if not found
        """
        key = self._make_key(model, text)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute(
            'SELECT embedding, dim, dtype FROM embeddings WHERE key = ?',
            (key,)
        )
        row = c.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        emb_bytes, dim, dtype = row
        # Reconstruct numpy array from bytes
        arr = np.frombuffer(emb_bytes, dtype=dtype)
        return arr.copy()  # Return a copy to avoid buffer issues
    
    def set(self, model: str, text: str, embedding: np.ndarray):
        """Store an embedding in the cache.
        
        Args:
            model: Model name used for embedding
            text: Text that was embedded
            embedding: Numpy array to cache
        """
        key = self._make_key(model, text)
        
        # Convert embedding to bytes
        emb_bytes = embedding.tobytes()
        dim = embedding.shape[0]
        dtype = str(embedding.dtype)
        created_at = datetime.now().isoformat()
        
        # Hash of normalized text for potential debugging/cleanup
        text_norm = text.lower().strip()
        text_hash = hashlib.md5(text_norm.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Use INSERT OR REPLACE to handle duplicates
        c.execute('''
            INSERT OR REPLACE INTO embeddings 
            (key, model, text_hash, embedding, dim, dtype, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (key, model, text_hash, emb_bytes, dim, dtype, created_at))
        
        conn.commit()
        conn.close()
    
    def clear(self):
        """Clear all cached embeddings."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM embeddings')
        conn.commit()
        conn.close()
    
    def size(self) -> int:
        """Return the number of cached embeddings."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM embeddings')
        count = c.fetchone()[0]
        conn.close()
        return count
