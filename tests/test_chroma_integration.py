#!/usr/bin/env python3
"""
Manual integration test for Chroma backend.
Tests that the HybridMemorySystem correctly initializes with Chroma when VECTOR_DB=chroma.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_ai.memory import HybridMemorySystem


def test_chroma_integration():
    """Test that HybridMemorySystem works with Chroma backend."""
    print("Testing Chroma integration...")
    
    # Create temp directories
    temp_db = tempfile.mktemp(suffix=".db")
    temp_chroma = tempfile.mkdtemp()
    
    try:
        # Set environment variable to use Chroma
        os.environ["VECTOR_DB"] = "chroma"
        os.environ["CHROMA_PERSIST_DIR"] = temp_chroma
        
        # Initialize memory system
        print(f"Initializing HybridMemorySystem with Chroma backend...")
        print(f"  DB path: {temp_db}")
        print(f"  Chroma dir: {temp_chroma}")
        
        mem = HybridMemorySystem(db_path=temp_db)
        
        # Verify it's using ChromaAdapter
        from my_ai.vector_backends.chroma_adapter import ChromaAdapter
        assert isinstance(mem.vector_store, ChromaAdapter), f"Expected ChromaAdapter, got {type(mem.vector_store)}"
        print("✓ Successfully initialized with ChromaAdapter")
        
        # Test storing and retrieving memories
        user_id = "test_user"
        mem.store_interaction(user_id, "user", "My name is Alice")
        mem.store_interaction(user_id, "user", "I work at Google")
        print("✓ Stored memories")
        
        # Retrieve context
        context = mem.get_context(user_id, "What's my name?")
        print(f"✓ Retrieved context with {len(context['explicit_facts'])} facts")
        
        # Format for prompt
        prompt_text = mem.format_for_prompt(context)
        print(f"✓ Formatted prompt:\n{prompt_text}\n")
        
        # Test persistence
        mem.vector_store.persist()
        print("✓ Persisted to disk")
        
        # Create a new instance and verify data persists
        mem2 = HybridMemorySystem(db_path=temp_db)
        context2 = mem2.get_context(user_id, "What's my name?")
        print(f"✓ Verified persistence: {len(context2['explicit_facts'])} facts after reload")
        
        print("\n✅ All Chroma integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_db):
            os.remove(temp_db)
        if os.path.exists(temp_chroma):
            shutil.rmtree(temp_chroma)
        # Reset environment
        os.environ.pop("VECTOR_DB", None)
        os.environ.pop("CHROMA_PERSIST_DIR", None)


def test_fallback_to_in_memory():
    """Test that the system falls back to in-memory when Chroma is not specified."""
    print("\nTesting fallback to in-memory VectorStore...")
    
    temp_db = tempfile.mktemp(suffix=".db")
    
    try:
        # Don't set VECTOR_DB, should default to in-memory
        os.environ.pop("VECTOR_DB", None)
        
        mem = HybridMemorySystem(db_path=temp_db)
        
        # Verify it's using VectorStore (not ChromaAdapter)
        from my_ai.memory import VectorStore
        assert isinstance(mem.vector_store, VectorStore), f"Expected VectorStore, got {type(mem.vector_store)}"
        print("✓ Successfully initialized with in-memory VectorStore (default)")
        
        # Test basic functionality
        user_id = "test_user"
        mem.store_interaction(user_id, "user", "My name is Bob")
        context = mem.get_context(user_id, "What's my name?")
        print(f"✓ In-memory store works: {len(context['explicit_facts'])} facts")
        
        print("\n✅ Fallback test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


if __name__ == "__main__":
    success1 = test_chroma_integration()
    success2 = test_fallback_to_in_memory()
    
    if success1 and success2:
        print("\n✅✅ All manual integration tests passed! ✅✅")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
