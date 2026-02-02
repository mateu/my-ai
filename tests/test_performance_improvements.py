"""Tests for performance improvements.

This module tests:
- Numpy-backed VectorStore with vectorized search
- Persistent embedding cache
- SQLite performance tuning (WAL mode, indexes)
- Backwards compatibility of public APIs
"""

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_ai.memory import VectorStore, HybridMemorySystem
from my_ai.embedding_cache import EmbeddingCache, get_embedding_cache


class TestNumpyVectorStore:
    """Test numpy-matrix-backed VectorStore implementation."""

    def test_empty_store_returns_empty_results(self):
        """Empty store should return empty list."""
        store = VectorStore(dim=384)
        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, user_id="user1", top_k=3)
        assert results == []

    def test_add_and_search_single_vector(self):
        """Adding and searching a single vector should work."""
        store = VectorStore(dim=128)

        # Add a vector
        vec = np.random.randn(128).astype(np.float32)
        store.add(
            id="vec1",
            text="test content",
            metadata={"user_id": "user1", "type": "test"},
            embedding=vec
        )

        # Search should return it
        query = vec  # Same vector should have high similarity
        results = store.search(query, user_id="user1", top_k=1)

        assert len(results) == 1
        assert results[0][0] == "vec1"
        assert results[0][1] > 0.99  # Should be very similar (normalized)

    def test_user_id_filtering(self):
        """Search should filter by user_id."""
        store = VectorStore(dim=64)

        # Add vectors for different users
        vec1 = np.random.randn(64).astype(np.float32)
        vec2 = np.random.randn(64).astype(np.float32)

        store.add("vec1", "text1", {"user_id": "user1"}, vec1)
        store.add("vec2", "text2", {"user_id": "user2"}, vec2)

        # Search as user1
        results = store.search(vec1, user_id="user1", top_k=5)
        assert len(results) == 1
        assert results[0][0] == "vec1"

        # Search as user2
        results = store.search(vec2, user_id="user2", top_k=5)
        assert len(results) == 1
        assert results[0][0] == "vec2"

    def test_top_k_limiting(self):
        """Search should respect top_k parameter."""
        store = VectorStore(dim=64)

        # Add 5 vectors for same user
        for i in range(5):
            vec = np.random.randn(64).astype(np.float32)
            store.add(f"vec{i}", f"text{i}", {"user_id": "user1"}, vec)

        query = np.random.randn(64).astype(np.float32)
        
        # Request top 3
        results = store.search(query, user_id="user1", top_k=3)
        assert len(results) == 3

        # Request top 2
        results = store.search(query, user_id="user1", top_k=2)
        assert len(results) == 2

    def test_results_sorted_by_score(self):
        """Search results should be sorted by score (descending)."""
        store = VectorStore(dim=64)

        # Add vectors
        for i in range(5):
            vec = np.random.randn(64).astype(np.float32)
            store.add(f"vec{i}", f"text{i}", {"user_id": "user1"}, vec)

        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, user_id="user1", top_k=5)

        # Verify descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_update_existing_id(self):
        """Adding with existing ID should update the vector."""
        store = VectorStore(dim=64)

        vec1 = np.random.randn(64).astype(np.float32)
        vec2 = np.random.randn(64).astype(np.float32)

        # Add initial vector
        store.add("vec1", "text1", {"user_id": "user1"}, vec1)
        
        # Update with new vector
        store.add("vec1", "text2", {"user_id": "user1"}, vec2)

        # Should only have one entry
        results = store.search(vec2, user_id="user1", top_k=10)
        assert len(results) == 1

        # Should match the updated vector
        assert results[0][0] == "vec1"
        assert results[0][1] > 0.99

    def test_float32_dtype(self):
        """Vectors should be stored as float32."""
        store = VectorStore(dim=64)

        # Add with float64
        vec = np.random.randn(64).astype(np.float64)
        store.add("vec1", "text1", {"user_id": "user1"}, vec)

        # Internal storage should be float32
        assert store.vectors.dtype == np.float32

    def test_normalization(self):
        """Vectors should be normalized to unit length."""
        store = VectorStore(dim=64)

        vec = np.random.randn(64).astype(np.float32) * 10  # Large magnitude
        store.add("vec1", "text1", {"user_id": "user1"}, vec)

        # Check normalization
        idx = store.id_to_index["vec1"]
        stored_vec = store.vectors[idx]
        norm = np.linalg.norm(stored_vec)
        assert abs(norm - 1.0) < 1e-5  # Should be unit length


class TestEmbeddingCache:
    """Test persistent embedding cache."""

    def test_cache_miss_returns_none(self, tmp_path):
        """Cache miss should return None."""
        cache_db = tmp_path / "test_cache.db"
        cache = EmbeddingCache(str(cache_db))

        result = cache.get("test-model", "some text")
        assert result is None

    def test_cache_set_and_get(self, tmp_path):
        """Setting and getting from cache should work."""
        cache_db = tmp_path / "test_cache.db"
        cache = EmbeddingCache(str(cache_db))

        # Set an embedding
        embedding = np.random.randn(128).astype(np.float32)
        cache.set("test-model", "hello world", embedding)

        # Get it back
        result = cache.get("test-model", "hello world")
        assert result is not None
        assert np.allclose(result, embedding)

    def test_cache_key_normalization(self, tmp_path):
        """Cache keys should be normalized (case-insensitive, trimmed)."""
        cache_db = tmp_path / "test_cache.db"
        cache = EmbeddingCache(str(cache_db))

        embedding = np.random.randn(64).astype(np.float32)

        # Set with one format
        cache.set("model", "  Hello World  ", embedding)

        # Get with different format (should match)
        result = cache.get("model", "hello world")
        assert result is not None
        assert np.allclose(result, embedding)

    def test_cache_different_models(self, tmp_path):
        """Different models should have separate cache entries."""
        cache_db = tmp_path / "test_cache.db"
        cache = EmbeddingCache(str(cache_db))

        emb1 = np.ones(64, dtype=np.float32)
        emb2 = np.zeros(64, dtype=np.float32)

        cache.set("model1", "text", emb1)
        cache.set("model2", "text", emb2)

        result1 = cache.get("model1", "text")
        result2 = cache.get("model2", "text")

        assert np.allclose(result1, emb1)
        assert np.allclose(result2, emb2)

    def test_cache_clear(self, tmp_path):
        """Clearing cache should remove all entries."""
        cache_db = tmp_path / "test_cache.db"
        cache = EmbeddingCache(str(cache_db))

        # Add some entries
        cache.set("model", "text1", np.ones(64, dtype=np.float32))
        cache.set("model", "text2", np.zeros(64, dtype=np.float32))

        # Clear
        cache.clear()

        # Should be empty
        assert cache.get("model", "text1") is None
        assert cache.get("model", "text2") is None
        assert cache.size() == (0, 0)

    def test_cache_replace_existing(self, tmp_path):
        """Setting same key should replace existing entry."""
        cache_db = tmp_path / "test_cache.db"
        cache = EmbeddingCache(str(cache_db))

        emb1 = np.ones(64, dtype=np.float32)
        emb2 = np.zeros(64, dtype=np.float32)

        # Set initial
        cache.set("model", "text", emb1)

        # Replace
        cache.set("model", "text", emb2)

        # Should have the new value
        result = cache.get("model", "text")
        assert np.allclose(result, emb2)

        # Should only have one entry in persistent cache
        _, persistent_size = cache.size()
        assert persistent_size == 1


class TestSQLiteOptimizations:
    """Test SQLite performance tuning."""

    def test_wal_mode_enabled(self, tmp_path):
        """WAL mode should be enabled."""
        db_path = tmp_path / "test_memory.db"
        memory = HybridMemorySystem(str(db_path))

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("PRAGMA journal_mode")
        result = c.fetchone()[0]
        conn.close()

        assert result.lower() == "wal"

    def test_indexes_created(self, tmp_path):
        """Required indexes should be created."""
        db_path = tmp_path / "test_memory.db"
        memory = HybridMemorySystem(str(db_path))

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        # Get all indexes
        c.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name LIKE 'idx_%'
        """)
        indexes = {row[0] for row in c.fetchall()}
        conn.close()

        # Check required indexes exist
        required_indexes = {
            "idx_conversation_user_ts",
            "idx_implicit_user",
            "idx_explicit_user",
        }

        assert required_indexes.issubset(indexes)


class TestBackwardsCompatibility:
    """Test backwards compatibility of public APIs."""

    def test_vector_store_api_compatibility(self):
        """VectorStore public API should remain unchanged."""
        # Test constructor signature
        store = VectorStore(dim=384)
        assert hasattr(store, "add")
        assert hasattr(store, "search")

        # Test add signature
        vec = np.random.randn(384).astype(np.float32)
        store.add(
            id="test",
            text="content",
            metadata={"user_id": "user1"},
            embedding=vec
        )

        # Test search signature
        results = store.search(vec, user_id="user1", top_k=3)
        assert isinstance(results, list)
        
        # Results should be List[Tuple[str, float]]
        if results:
            assert isinstance(results[0], tuple)
            assert isinstance(results[0][0], str)
            assert isinstance(results[0][1], (float, np.floating))

    def test_hybrid_memory_system_api_compatibility(self, tmp_path):
        """HybridMemorySystem public API should remain unchanged."""
        db_path = tmp_path / "test_memory.db"
        
        # Test constructor
        memory = HybridMemorySystem(str(db_path))
        
        # Test public methods exist
        assert hasattr(memory, "store_interaction")
        assert hasattr(memory, "get_context")
        assert hasattr(memory, "format_for_prompt")

        # Test basic workflow
        memory.store_interaction("user1", "user", "my name is Alice")
        context = memory.get_context("user1", "what is my name?")
        
        assert "explicit_facts" in context
        assert "behavioral_patterns" in context
        assert "user_id" in context

        # Test format_for_prompt
        prompt_text = memory.format_for_prompt(context)
        assert isinstance(prompt_text, str)
