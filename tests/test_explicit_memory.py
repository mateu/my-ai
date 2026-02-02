import os
import sqlite3

import sys
from pathlib import Path

# Ensure the project root (which contains ai_memory_system.py) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_memory_system import HybridMemorySystem


def test_explicit_memories_are_canonical_key_value(tmp_path):
    """Storing structured facts should yield clean key:value explicit memories.

    This guards against duplicate / messy rows like:
      - "my name is Hunter"
      - "name: Hunter"
      - "my wife is Kirstenremember that my wife is Kirsten"
    for a single underlying fact.
    """

    db_path = tmp_path / "chat_memory_test.db"
    user_id = "test_user"

    memory = HybridMemorySystem(str(db_path))

    # Case 1: direct "my X is Y"
    memory.store_interaction(user_id, "user", "my name is Hunter")

    # Case 2: wrapped inside "remember that ..."
    memory.store_interaction(user_id, "user", "remember that my wife is Kirsten")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT memory_type, content FROM explicit_memories WHERE user_id = ? ORDER BY id",
            (user_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    # We expect two clean, canonical facts.
    assert rows == [
        ("fact", "name: Hunter"),
        ("fact", "wife: Kirsten"),
    ]
