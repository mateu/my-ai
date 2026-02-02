"""Tests for performance improvements: VectorStore, embedding cache, etc."""
import pytest
import numpy as np
import os
import tempfile
import sqlite3
from my_ai.memory import VectorStore, HybridMemorySystem
from my_ai.embedding_cache import EmbeddingCache


class TestNumpyVectorStore:
    """Test the numpy-matrix-backed VectorStore implementation."""
    
    def test_empty_store_returns_empty_results(self):
        """Empty store should return empty list."""
        store = VectorStore(dim=384)
        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, user_id="user1", top_k=3)
        assert results == []
    
    def test_add_and_search_single_vector(self):
        """Adding and searching a single vector should work."""
        store = VectorStore(dim=384)
        
        # Add a vector
        emb = np.random.randn(384).astype(np.float32)
        store.add("id1", "test text", {"user_id": "user1"}, emb)
        
        # Search should return it
        results = store.search(emb, user_id="user1", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "id1"
        assert results[0][1] > 0.99  # Should be close to 1.0 (same vector)
    
    def test_user_id_filtering(self):
        """Search should filter by user_id."""
        store = VectorStore(dim=384)
        
        # Add vectors for different users
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        
        store.add("id1", "text1", {"user_id": "user1"}, emb1)
        store.add("id2", "text2", {"user_id": "user2"}, emb2)
        
        # Search for user1 should only return user1's vector
        results = store.search(emb1, user_id="user1", top_k=10)
        assert len(results) == 1
        assert results[0][0] == "id1"
        
        # Search for user2 should only return user2's vector
        results = store.search(emb2, user_id="user2", top_k=10)
        assert len(results) == 1
        assert results[0][0] == "id2"
    
    def test_top_k_limiting(self):
        """Search should respect top_k parameter."""
        store = VectorStore(dim=384)
        
        # Add multiple vectors for same user
        for i in range(10):
            emb = np.random.randn(384).astype(np.float32)
            store.add(f"id{i}", f"text{i}", {"user_id": "user1"}, emb)
        
        # Request only top 3
        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, user_id="user1", top_k=3)
        assert len(results) == 3
    
    def test_results_sorted_by_score(self):
        """Results should be sorted by similarity score descending."""
        store = VectorStore(dim=384)
        
        # Create a query vector
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Add vectors with varying similarity to query
        for i in range(5):
            # Mix query with random noise
            emb = query * (1 - i * 0.15) + np.random.randn(384).astype(np.float32) * (i * 0.15)
            store.add(f"id{i}", f"text{i}", {"user_id": "user1"}, emb)
        
        results = store.search(query, user_id="user1", top_k=5)
        
        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_update_existing_id(self):
        """Adding with same id should update the vector."""
        store = VectorStore(dim=384)
        
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        
        # Add initial vector
        store.add("id1", "text1", {"user_id": "user1"}, emb1)
        
        # Update with different vector
        store.add("id1", "text2", {"user_id": "user1"}, emb2)
        
        # Should still have only one vector
        assert len(store.ids) == 1
        
        # Search with new embedding should return high score
        results = store.search(emb2, user_id="user1", top_k=1)
        assert results[0][1] > 0.99
    
    def test_float32_dtype(self):
        """Vectors should be stored as float32."""
        store = VectorStore(dim=384)
        
        # Add with float64
        emb = np.random.randn(384).astype(np.float64)
        store.add("id1", "text", {"user_id": "user1"}, emb)
        
        # Should be converted to float32
        assert store.matrix.dtype == np.float32
    
    def test_normalization(self):
        """Vectors should be normalized to unit length."""
        store = VectorStore(dim=384)
        
        # Add unnormalized vector
        emb = np.random.randn(384).astype(np.float32) * 10
        store.add("id1", "text", {"user_id": "user1"}, emb)
        
        # Stored vector should have unit length
        stored = store.matrix[0]
        assert abs(np.linalg.norm(stored) - 1.0) < 1e-5


class TestEmbeddingCache:
    """Test the persistent embedding cache."""
    
    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        
        try:
            cache = EmbeddingCache(cache_path)
            result = cache.get("model1", "some text")
            assert result is None
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    
    def test_cache_set_and_get(self):
        """Should be able to store and retrieve embeddings."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        
        try:
            cache = EmbeddingCache(cache_path)
            
            # Store an embedding
            emb = np.random.randn(384).astype(np.float32)
            cache.set("model1", "hello world", emb)
            
            # Retrieve it
            retrieved = cache.get("model1", "hello world")
            assert retrieved is not None
            assert np.allclose(retrieved, emb)
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    
    def test_cache_key_normalization(self):
        """Text should be normalized (lowercase, trimmed) for key generation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        
        try:
            cache = EmbeddingCache(cache_path)
            
            emb = np.random.randn(384).astype(np.float32)
            
            # Store with one format
            cache.set("model1", "Hello World", emb)
            
            # Should retrieve with different formatting
            retrieved = cache.get("model1", "  hello world  ")
            assert retrieved is not None
            assert np.allclose(retrieved, emb)
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    
    def test_cache_different_models(self):
        """Same text with different models should be cached separately."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        
        try:
            cache = EmbeddingCache(cache_path)
            
            emb1 = np.random.randn(384).astype(np.float32)
            emb2 = np.random.randn(384).astype(np.float32)
            
            cache.set("model1", "text", emb1)
            cache.set("model2", "text", emb2)
            
            retrieved1 = cache.get("model1", "text")
            retrieved2 = cache.get("model2", "text")
            
            assert np.allclose(retrieved1, emb1)
            assert np.allclose(retrieved2, emb2)
            assert not np.allclose(retrieved1, retrieved2)
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    
    def test_cache_clear(self):
        """Clear should remove all cached embeddings."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        
        try:
            cache = EmbeddingCache(cache_path)
            
            # Add some embeddings
            for i in range(5):
                emb = np.random.randn(384).astype(np.float32)
                cache.set("model1", f"text{i}", emb)
            
            assert cache.size() == 5
            
            cache.clear()
            assert cache.size() == 0
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
    
    def test_cache_replace_existing(self):
        """Setting same key should replace existing value."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        
        try:
            cache = EmbeddingCache(cache_path)
            
            emb1 = np.random.randn(384).astype(np.float32)
            emb2 = np.random.randn(384).astype(np.float32)
            
            cache.set("model1", "text", emb1)
            cache.set("model1", "text", emb2)
            
            # Should only have one entry
            assert cache.size() == 1
            
            # Should retrieve the second embedding
            retrieved = cache.get("model1", "text")
            assert np.allclose(retrieved, emb2)
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)


class TestSQLiteOptimizations:
    """Test that SQLite optimizations are applied."""
    
    def test_wal_mode_enabled(self):
        """WAL mode should be enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            system = HybridMemorySystem(db_path=db_path)
            
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("PRAGMA journal_mode")
            mode = c.fetchone()[0]
            conn.close()
            
            assert mode.lower() == "wal"
    
    def test_indexes_created(self):
        """Performance indexes should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            system = HybridMemorySystem(db_path=db_path)
            
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Check for indexes
            c.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in c.fetchall()]
            conn.close()
            
            assert "idx_conversation_user_ts" in indexes
            assert "idx_implicit_user" in indexes
            assert "idx_explicit_user" in indexes


class TestBackwardsCompatibility:
    """Test that changes are backwards compatible."""
    
    def test_vector_store_api_compatibility(self):
        """VectorStore should maintain the same public API."""
        store = VectorStore(dim=384)
        
        # These methods should exist and work
        emb = np.random.randn(384).astype(np.float32)
        store.add("id1", "text", {"user_id": "user1"}, emb)
        results = store.search(emb, "user1", top_k=3)
        
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_hybrid_memory_system_api_compatibility(self):
        """HybridMemorySystem should maintain existing API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            system = HybridMemorySystem(db_path=db_path)
            
            # Existing methods should work
            system.store_interaction("user1", "user", "Hello")
            context = system.get_context("user1", "What's my name?")
            formatted = system.format_for_prompt(context)
            
            assert isinstance(context, dict)
            assert isinstance(formatted, str)
