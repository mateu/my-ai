"""Persistent embedding cache with SQLite backend and in-memory LRU warm cache."""

import os
import sqlite3
import hashlib
import numpy as np
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple


class EmbeddingCache:
    """Persistent embedding cache using SQLite with in-memory LRU cache."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.getenv("EMBEDDING_CACHE_DB", "data/embeddings_cache.db")
        
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.isabs(db_path):
            os.makedirs(db_dir, exist_ok=True)
        elif db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
        
        # In-memory cache for fast lookups during process lifetime
        self._warm_cache = {}
        
    def _init_db(self):
        """Initialize the embedding cache database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                     (key TEXT PRIMARY KEY,
                      model TEXT NOT NULL,
                      text_hash TEXT NOT NULL,
                      embedding BLOB NOT NULL,
                      dim INTEGER NOT NULL,
                      dtype TEXT NOT NULL,
                      created_at TEXT NOT NULL)''')
        
        # Index for faster lookups by model and text_hash
        c.execute('''CREATE INDEX IF NOT EXISTS idx_model_hash 
                     ON embeddings(model, text_hash)''')
        
        conn.commit()
        conn.close()
    
    def _generate_key(self, model: str, text: str) -> str:
        """Generate a stable cache key from model and normalized text.
        
        Args:
            model: The embedding model name
            text: The text to embed
            
        Returns:
            SHA256 hash as hex string
        """
        normalized_text = text.strip().lower()
        combined = f"{model}::{normalized_text}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def get(self, model: str, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for model and text.
        
        Args:
            model: The embedding model name
            text: The text to embed
            
        Returns:
            Cached embedding as numpy array, or None if not found
        """
        key = self._generate_key(model, text)
        
        # Check in-memory warm cache first
        if key in self._warm_cache:
            return self._warm_cache[key]
        
        # Check persistent cache
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT embedding, dim, dtype FROM embeddings WHERE key = ?''', (key,))
        row = c.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        embedding_bytes, dim, dtype = row
        
        # Reconstruct numpy array from bytes
        embedding = np.frombuffer(embedding_bytes, dtype=dtype).reshape(dim)
        
        # Store in warm cache for future lookups
        self._warm_cache[key] = embedding
        
        return embedding
    
    def set(self, model: str, text: str, embedding: np.ndarray):
        """Store embedding in cache.
        
        Args:
            model: The embedding model name
            text: The text that was embedded
            embedding: The embedding vector as numpy array
        """
        key = self._generate_key(model, text)
        normalized_text = text.strip().lower()
        text_hash = hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
        
        # Store in warm cache
        self._warm_cache[key] = embedding
        
        # Store in persistent cache
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        dim = embedding.shape[0]
        dtype = str(embedding.dtype)
        created_at = datetime.now().isoformat()
        
        # Use INSERT OR REPLACE to handle updates
        c.execute('''INSERT OR REPLACE INTO embeddings 
                     (key, model, text_hash, embedding, dim, dtype, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (key, model, text_hash, embedding_bytes, dim, dtype, created_at))
        
        conn.commit()
        conn.close()
    
    def clear_warm_cache(self):
        """Clear the in-memory warm cache."""
        self._warm_cache.clear()


# Global instance for use across the application
_global_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache
