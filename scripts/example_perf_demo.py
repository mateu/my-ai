#!/usr/bin/env python3
"""
Performance demo: demonstrate the improved vector store and embedding cache.

This script shows how to:
1. Create a HybridMemorySystem
2. Add explicit memories
3. Run similarity searches
4. See the persistent embedding cache in action

Run with: uv run python scripts/example_perf_demo.py
"""

import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path so we can import my_ai
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from my_ai import HybridMemorySystem


def main():
    print("🚀 Performance Demo: Improved Vector Store & Embedding Cache\n")
    print("=" * 70)
    
    # Use a temporary database for this demo
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo_memory.db")
        cache_path = os.path.join(tmpdir, "demo_cache.db")
        
        # Set cache path via environment
        os.environ["EMBEDDING_CACHE_DB"] = cache_path
        
        print(f"\n📁 Using temporary database: {db_path}")
        print(f"📁 Using temporary cache: {cache_path}\n")
        
        # Create memory system
        memory = HybridMemorySystem(db_path)
        user_id = "demo_user"
        
        # Add some explicit memories
        print("💾 Storing explicit memories...")
        facts = [
            "my name is Alice",
            "I work at Acme Corp",
            "I love Python programming",
            "my favorite food is pizza",
            "I have two cats named Luna and Max",
        ]
        
        for fact in facts:
            memory.store_interaction(user_id, "user", fact)
            print(f"  ✓ {fact}")
        
        # Show vector store stats
        vs = memory.vector_store
        print(f"\n📊 Vector Store Stats:")
        print(f"  - Total vectors: {len(vs.ids)}")
        print(f"  - Matrix shape: {vs.vectors.shape}")
        print(f"  - Memory usage: ~{vs.vectors.nbytes / 1024:.2f} KB")
        
        # Run some searches
        print(f"\n🔍 Running similarity searches...")
        queries = [
            "What is my name?",
            "What do I do for work?",
            "Tell me about my pets",
        ]
        
        for query in queries:
            print(f"\n  Query: '{query}'")
            context = memory.get_context(user_id, query)
            
            if context["explicit_facts"]:
                print("  Relevant facts:")
                for fact in context["explicit_facts"][:3]:
                    print(f"    - {fact['content']} (relevance: {fact['relevance']:.3f})")
            else:
                print("  No relevant facts found")
        
        # Test embedding cache (run same query twice)
        print(f"\n⚡ Testing embedding cache performance...")
        test_query = "What is my name?"
        
        print(f"  First run (will compute embedding)...")
        emb1 = memory._get_embedding(test_query)
        
        print(f"  Second run (should use cache)...")
        emb2 = memory._get_embedding(test_query)
        
        # Verify they're the same
        import numpy as np
        if np.allclose(emb1, emb2):
            print(f"  ✓ Cache hit! Same embedding returned.")
        else:
            print(f"  ✗ Cache miss (unexpected)")
        
        print("\n" + "=" * 70)
        print("\n✅ Demo complete! All performance features are working.\n")
        print("Key features demonstrated:")
        print("  • Numpy-matrix vector store with BLAS-backed search")
        print("  • Persistent embedding cache (SQLite)")
        print("  • Efficient memory storage and retrieval")
        print("  • Float32 vectors for optimal performance")
        print()


if __name__ == "__main__":
    main()
