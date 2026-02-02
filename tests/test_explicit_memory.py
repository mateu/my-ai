import os
import sqlite3

import sys
from pathlib import Path

# Ensure the project root (which contains the my_ai package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_ai import HybridMemorySystem


def test_explicit_memories_are_canonical_key_value(tmp_path):
    """Storing structured facts should yield clean key:value explicit memories.

    This guards against duplicate / messy rows like:
      - "my name is Hunter"
      - "name: Hunter"
      - "my wife is Deviremember that my wife is Devi"
    for a single underlying fact.
    """

    db_path = tmp_path / "chat_memory_test.db"
    user_id = "test_user"

    memory = HybridMemorySystem(str(db_path))

    # Case 1: direct "my X is Y"
    memory.store_interaction(user_id, "user", "my name is Hunter")

    # Case 2: wrapped inside "remember that ..."
    memory.store_interaction(user_id, "user", "remember that my wife is Devi")

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
        ("fact", "wife: Devi"),
    ]


def test_multi_entity_pets_are_canonical(tmp_path):
    """Multi-entity pet facts should normalize to a single canonical field.

    Variants like:
      - "I have two cats named Barcelona and Jesus"
      - "remember I have two cats named Barcelona and Jesus"
      - "remember that I have two cats named Barcelona and Jesus"
    should converge into one fact: "cats: Barcelona, Jesus".
    """

    db_path = tmp_path / "chat_memory_pets.db"
    user_id = "test_user"

    memory = HybridMemorySystem(str(db_path))

    messages = [
        "I have two cats named Barcelona and Jesus",
        "remember I have two cats named Barcelona and Jesus",
        "remember that I have two cats named Barcelona and Jesus",
    ]

    for msg in messages:
        memory.store_interaction(user_id, "user", msg)

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

    assert rows == [
        ("fact", "cats: Barcelona, Jesus"),
    ]


def test_i_have_sentences_store_only_canonical_fact(tmp_path):
    """Sentences like "I have two dogs ..." should store only the canonical fact.

    We don't want raw clauses like "I have two dogs ..." or "two dogs named ..."
    hanging around in explicit_memories when we can derive a clean field: value.
    """

    db_path = tmp_path / "chat_memory_have.db"
    user_id = "test_user"

    memory = HybridMemorySystem(str(db_path))

    memory.store_interaction(user_id, "user", "I have two dogs named Enola and Glacier")

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

    assert rows == [
        ("fact", "dogs: Enola, Glacier"),
    ]
