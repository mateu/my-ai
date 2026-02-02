#!/usr/bin/env python3
"""Migration script to transfer explicit memories from SQLite to Chroma.

This script:
1. Loads explicit memories from the SQLite database
2. Uses HybridMemorySystem to get embeddings and insert into Chroma
3. Persists the Chroma collection

Usage:
    VECTOR_DB=chroma python scripts/migrate_to_chroma.py [--db-path path/to/memory.db]

Environment variables:
    VECTOR_DB: Must be set to "chroma" to use ChromaDB
    CHROMA_PERSIST_DIR: Optional, defaults to ./chroma_db
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from my_ai.memory import HybridMemorySystem


def migrate_to_chroma(db_path: str = None):
    """Migrate explicit memories from SQLite to Chroma."""
    
    # Verify VECTOR_DB is set to chroma
    if os.getenv("VECTOR_DB", "").lower() != "chroma":
        print("ERROR: VECTOR_DB environment variable must be set to 'chroma'")
        print("Usage: VECTOR_DB=chroma python scripts/migrate_to_chroma.py")
        sys.exit(1)
    
    # Default db_path
    if db_path is None:
        db_path = os.path.join("data", "chat_memory.db")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found: {db_path}")
        print("Please specify the correct path with --db-path")
        sys.exit(1)
    
    print(f"Migrating memories from: {db_path}")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    print(f"Target Chroma directory: {chroma_dir}")
    
    # Initialize HybridMemorySystem with VECTOR_DB=chroma
    # This will automatically use ChromaAdapter
    memory_system = HybridMemorySystem(db_path=db_path)
    
    # Verify we're using ChromaAdapter
    from my_ai.vector_backends.chroma_adapter import ChromaAdapter
    if not isinstance(memory_system.vector_store, ChromaAdapter):
        print("ERROR: ChromaAdapter not loaded. Is chromadb installed?")
        print("Install with: uv sync --extras chroma")
        sys.exit(1)
    
    print("Successfully initialized ChromaAdapter")
    
    # Load explicit memories from SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_id, memory_type, content, created_at 
        FROM explicit_memories 
        ORDER BY created_at
    """)
    
    memories = cursor.fetchall()
    conn.close()
    
    if not memories:
        print("No explicit memories found in database.")
        return
    
    print(f"Found {len(memories)} explicit memories to migrate")
    
    # Insert each memory into Chroma
    migrated_count = 0
    for user_id, mem_type, content, created_at in memories:
        # Generate embedding using the memory system's embedding function
        embedding = memory_system._get_embedding(content)
        
        # Create memory ID (same format as used in _store_explicit_memory)
        # Use deterministic hash for consistent IDs across runs
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
        mem_id = f"{user_id}_explicit_{content_hash}"
        
        # Add to vector store (ChromaAdapter)
        memory_system.vector_store.add(
            id=mem_id,
            text=content,
            metadata={
                "user_id": user_id,
                "type": "explicit",
                "memory_type": mem_type,
                "created_at": created_at
            },
            embedding=embedding
        )
        
        migrated_count += 1
        if migrated_count % 10 == 0:
            print(f"Migrated {migrated_count}/{len(memories)} memories...")
    
    # Persist the collection
    memory_system.vector_store.persist()
    
    print(f"\n✓ Successfully migrated {migrated_count} memories to Chroma")
    print(f"  Location: {chroma_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate explicit memories from SQLite to ChromaDB"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (default: data/chat_memory.db)"
    )
    
    args = parser.parse_args()
    migrate_to_chroma(db_path=args.db_path)
