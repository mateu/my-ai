#!/usr/bin/env python3
"""
Simple migration: read explicit_memories from SQLite and insert into Chroma.
Run with: VECTOR_DB=chroma python scripts/migrate_to_chroma.py
"""
import sqlite3
from my_ai.memory import HybridMemorySystem
import os
import hashlib

def main(db_path="data/chat_memory.db", user_filter: str | None = None):
    # Initialize the memory system so we can use its _get_embedding helper
    mem = HybridMemorySystem(db_path=db_path)
    # Expect mem.vector_store to be ChromaAdapter when VECTOR_DB=chroma
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    q = "SELECT user_id, memory_type, content FROM explicit_memories"
    if user_filter:
        q += " WHERE user_id = ?"
        rows = c.execute(q, (user_filter,)).fetchall()
    else:
        rows = c.execute(q).fetchall()

    for user_id, mem_type, content in rows:
        emb = mem._get_embedding(content)
        mem_id = f"{user_id}_explicit_{hashlib.md5(content.encode()).hexdigest()}"
        metadata = {"user_id": user_id, "type": "explicit", "memory_type": mem_type}
        # adapter add signature matches existing VectorStore.add
        mem.vector_store.add(mem_id, content, metadata, emb)
        print("Inserted", mem_id)

    # persist to disk (Chroma)
    try:
        mem.vector_store.persist()
    except Exception:
        pass
    conn.close()

if __name__ == "__main__":
    main()
