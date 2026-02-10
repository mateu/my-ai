import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

# Try to import chromadb; skip tests if not available
try:
    import chromadb
    from my_ai.vector_backends.chroma_adapter import ChromaAdapter
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for Chroma persistence."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
def test_chroma_adapter_init(temp_chroma_dir):
    """Test ChromaAdapter initialization."""
    adapter = ChromaAdapter(persist_directory=temp_chroma_dir, embedding_dim=384)
    assert adapter.embedding_dim == 384
    assert adapter.collection is not None
    assert isinstance(adapter.metadata, dict)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
def test_chroma_adapter_add_and_search(temp_chroma_dir):
    """Test adding memories and searching."""
    adapter = ChromaAdapter(persist_directory=temp_chroma_dir, embedding_dim=384)
    
    # Add a memory
    user_id = "test_user"
    text = "My name is Hunter"
    metadata = {"user_id": user_id, "type": "explicit"}
    embedding = np.random.randn(384).astype(np.float32)
    
    adapter.add("mem_1", text, metadata, embedding)
    
    # Verify metadata cache
    assert "mem_1" in adapter.metadata
    assert adapter.metadata["mem_1"]["user_id"] == user_id
    assert adapter.metadata["mem_1"]["text"] == text
    
    # Search with the same embedding (should return high similarity)
    results = adapter.search(embedding, user_id, top_k=1)
    assert len(results) == 1
    assert results[0][0] == "mem_1"
    # Score should be high (close to 1.0 for same embedding)
    assert results[0][1] > 0.8


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
def test_chroma_adapter_user_filtering(temp_chroma_dir):
    """Test that search filters by user_id."""
    adapter = ChromaAdapter(persist_directory=temp_chroma_dir, embedding_dim=384)
    
    # Add memories for different users
    emb1 = np.random.randn(384).astype(np.float32)
    emb2 = np.random.randn(384).astype(np.float32)
    
    adapter.add("mem_user1", "User 1 memory", {"user_id": "user1", "type": "explicit"}, emb1)
    adapter.add("mem_user2", "User 2 memory", {"user_id": "user2", "type": "explicit"}, emb2)
    
    # Search for user1's memories
    results = adapter.search(emb1, "user1", top_k=5)
    
    # Should only return user1's memory
    assert len(results) >= 1
    # The first result should be user1's memory
    assert results[0][0] == "mem_user1"


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
def test_chroma_adapter_dimension_mismatch(temp_chroma_dir):
    """Test that the adapter handles dimension mismatches."""
    adapter = ChromaAdapter(persist_directory=temp_chroma_dir, embedding_dim=384)
    
    # Add with different dimension (should be padded/trimmed)
    user_id = "test_user"
    text = "Test memory"
    metadata = {"user_id": user_id, "type": "explicit"}
    
    # Too small embedding
    small_emb = np.random.randn(100).astype(np.float32)
    adapter.add("mem_small", text, metadata, small_emb)
    
    # Too large embedding
    large_emb = np.random.randn(500).astype(np.float32)
    adapter.add("mem_large", text, metadata, large_emb)
    
    # Search should work
    query_emb = np.random.randn(384).astype(np.float32)
    results = adapter.search(query_emb, user_id, top_k=5)
    assert len(results) >= 2


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
def test_chroma_adapter_persist(temp_chroma_dir):
    """Test persistence functionality."""
    adapter = ChromaAdapter(persist_directory=temp_chroma_dir, embedding_dim=384)
    
    # Add a memory
    user_id = "test_user"
    text = "Persistent memory"
    metadata = {"user_id": user_id, "type": "explicit"}
    embedding = np.random.randn(384).astype(np.float32)
    
    adapter.add("mem_persist", text, metadata, embedding)
    
    # Call persist (should not raise error)
    adapter.persist()
    
    # Create a new adapter with the same directory
    adapter2 = ChromaAdapter(persist_directory=temp_chroma_dir, embedding_dim=384)
    
    # The memory should still be there (verify via search)
    results = adapter2.search(embedding, user_id, top_k=1)
    assert len(results) >= 1
