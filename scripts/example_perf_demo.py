#!/usr/bin/env python3
"""Performance demonstration script for HybridMemorySystem.

This script demonstrates the performance improvements including:
- Numpy-backed VectorStore with vectorized search
- Persistent embedding cache
- SQLite performance tuning (WAL mode, indexes)
- Transaction batching
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_ai.memory import HybridMemorySystem
from my_ai.embedding_cache import get_embedding_cache


def demo_vector_store_performance():
    """Demonstrate vectorized search performance."""
    print("\n" + "=" * 60)
    print("1. NUMPY-BACKED VECTORSTORE DEMO")
    print("=" * 60)
    
    # Create a temporary memory system
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        memory = HybridMemorySystem(db_path)
        user_id = "demo_user"
        
        print("\nAdding explicit memories...")
        facts = [
            "my name is Alice",
            "my favorite color is blue",
            "I work at TechCorp",
            "I have two dogs named Max and Bella",
            "I live in San Francisco",
            "my wife is Emma",
            "I prefer Python programming",
            "remember that I love hiking",
        ]
        
        start = time.time()
        for fact in facts:
            memory.store_interaction(user_id, "user", fact)
        elapsed = time.time() - start
        
        print(f"✓ Stored {len(facts)} facts in {elapsed:.3f}s")
        print(f"  ({elapsed/len(facts)*1000:.2f}ms per fact)")
        
        print("\nPerforming similarity search...")
        queries = [
            "What's my name?",
            "What do I do for work?",
            "Tell me about my pets",
        ]
        
        for query in queries:
            start = time.time()
            context = memory.get_context(user_id, query)
            elapsed = time.time() - start
            
            print(f"\nQuery: '{query}'")
            print(f"  Search time: {elapsed*1000:.2f}ms")
            print(f"  Found {len(context['explicit_facts'])} relevant facts:")
            for fact in context['explicit_facts'][:3]:
                print(f"    - {fact['content']} (relevance: {fact['relevance']:.2f})")
    finally:
        # Clean up temporary file
        if os.path.exists(db_path):
            os.remove(db_path)


def demo_embedding_cache():
    """Demonstrate persistent embedding cache."""
    print("\n" + "=" * 60)
    print("2. PERSISTENT EMBEDDING CACHE DEMO")
    print("=" * 60)
    
    cache = get_embedding_cache()
    
    # Clear cache for clean demo
    cache.clear()
    print("\n✓ Cache cleared")
    
    # Simulate embedding lookups
    import numpy as np
    
    texts = [
        "my name is Alice",
        "I love programming",
        "Python is great",
    ]
    model = "mock-model"
    
    print("\nFirst lookup (cache miss)...")
    start = time.time()
    for text in texts:
        # Simulate getting embedding
        emb = np.random.randn(384).astype(np.float32)
        cache.set(model, text, emb)
    elapsed = time.time() - start
    print(f"✓ Stored {len(texts)} embeddings in {elapsed:.3f}s")
    
    warm, persistent = cache.size()
    print(f"  Cache size: {warm} warm, {persistent} persistent")
    
    print("\nSecond lookup (cache hit)...")
    cache.clear_warm_cache()  # Clear warm cache to test persistent cache
    
    start = time.time()
    for text in texts:
        result = cache.get(model, text)
        assert result is not None, f"Cache miss for: {text}"
    elapsed = time.time() - start
    print(f"✓ Retrieved {len(texts)} embeddings in {elapsed:.3f}s")
    print(f"  ({elapsed/len(texts)*1000:.2f}ms per lookup)")
    
    # Demonstrate key normalization
    print("\nTesting key normalization...")
    emb1 = np.ones(64, dtype=np.float32)
    cache.set(model, "  Hello World  ", emb1)
    
    result = cache.get(model, "hello world")
    assert result is not None, "Normalization failed"
    print("✓ Cache correctly normalizes keys (case-insensitive, trimmed)")


def demo_sqlite_optimizations():
    """Demonstrate SQLite performance tuning."""
    print("\n" + "=" * 60)
    print("3. SQLITE PERFORMANCE TUNING DEMO")
    print("=" * 60)
    
    import sqlite3
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        memory = HybridMemorySystem(db_path)
        
        # Check WAL mode
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("PRAGMA journal_mode")
        journal_mode = c.fetchone()[0]
        print(f"\n✓ Journal mode: {journal_mode}")
        
        c.execute("PRAGMA synchronous")
        sync_mode = c.fetchone()[0]
        print(f"✓ Synchronous mode: {sync_mode}")
        
        # Check indexes
        c.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name LIKE 'idx_%'
        """)
        indexes = [row[0] for row in c.fetchall()]
        print(f"\n✓ Performance indexes created:")
        for idx in indexes:
            print(f"  - {idx}")
        
        conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def demo_transaction_batching():
    """Demonstrate transaction batching performance."""
    print("\n" + "=" * 60)
    print("4. TRANSACTION BATCHING DEMO")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        memory = HybridMemorySystem(db_path)
        user_id = "batch_user"
        
        print("\nStoring interaction with multiple memories...")
        message = "my name is Bob and I work at DataCorp and I love Python"
        
        start = time.time()
        memory.store_interaction(user_id, "user", message)
        elapsed = time.time() - start
        
        print(f"✓ Stored interaction with 3 explicit memories in {elapsed:.3f}s")
        print("  (All operations in single transaction)")
        
        # Verify memories were stored
        import sqlite3
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT content FROM explicit_memories WHERE user_id = ?", (user_id,))
        memories = [row[0] for row in c.fetchall()]
        conn.close()
        
        print(f"\n✓ Verified {len(memories)} memories in database:")
        for mem in memories:
            print(f"  - {mem}")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def main():
    """Run all performance demos."""
    print("\n" + "=" * 60)
    print("HYBRIDMEMORYSYSTEM PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo showcases the performance improvements:")
    print("• Numpy-matrix VectorStore with BLAS-backed search")
    print("• Persistent embedding cache with 2-tier strategy")
    print("• SQLite WAL mode and indexes")
    print("• Transaction batching for bulk operations")
    
    try:
        demo_vector_store_performance()
        demo_embedding_cache()
        demo_sqlite_optimizations()
        demo_transaction_batching()
        
        print("\n" + "=" * 60)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
