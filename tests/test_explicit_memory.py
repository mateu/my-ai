import os
import sqlite3
import json
import pytest
from unittest.mock import patch, Mock
import sys
from pathlib import Path

# Ensure the project root (which contains the my_ai package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_ai import HybridMemorySystem

# Helper to mock LLM response
def mock_llm_response(facts_list):
    """Create a mock response object for requests.post"""
    mock_resp = Mock()
    # The extraction method expects response.json()['message']['content'] -> JSON string
    content_json = json.dumps({"facts": facts_list})
    mock_resp.json.return_value = {
        "message": {
            "content": content_json
        }
    }
    mock_resp.raise_for_status = Mock()
    return mock_resp

@pytest.fixture
def mock_ollama():
    with patch('requests.post') as mock_post:
        yield mock_post

def test_explicit_memories_are_canonical_key_value(tmp_path, mock_ollama):
    """Storing structured facts should yield clean key:value explicit memories."""
    db_path = tmp_path / "chat_memory_test.db"
    user_id = "test_user"
    memory = HybridMemorySystem(str(db_path))

    # Enable mock extraction (bypass the check for "mock" backend in code)
    # We need to trick _extract_facts_llm to proceed.
    # It returns [] if AI_EMBEDDING_BACKEND == "mock".
    # So we need to ensure that check passes or use a real backend env var.
    # But wait, the makefile sets AI_EMBEDDING_BACKEND=mock.
    # We need to override it inside the test function or patch os.getenv.

    with patch.dict(os.environ, {"AI_EMBEDDING_BACKEND": "ollama", "AI_MEMORY_OFFLINE": ""}):
        # Case 1: "my name is Hunter"
        mock_ollama.return_value = mock_llm_response(["name: Hunter"])
        memory.store_interaction(user_id, "user", "my name is Hunter")

        # Case 2: "remember that my wife is Devi"
        mock_ollama.return_value = mock_llm_response(["wife: Devi"])
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

    assert rows == [
        ("fact", "name: Hunter"),
        ("fact", "wife: Devi"),
    ]

def test_multi_entity_pets_are_canonical(tmp_path, mock_ollama):
    """Multi-entity pet facts should normalize to a single canonical field."""
    db_path = tmp_path / "chat_memory_pets.db"
    user_id = "test_user"
    memory = HybridMemorySystem(str(db_path))

    with patch.dict(os.environ, {"AI_EMBEDDING_BACKEND": "ollama", "AI_MEMORY_OFFLINE": ""}):
        # We mock the LLM to return the normalized form we want
        mock_ollama.return_value = mock_llm_response(["cats: Barcelona, Jesus"])

        messages = [
            "I have two cats named Barcelona and Jesus",
            "remember I have two cats named Barcelona and Jesus",
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

    # Deduplication should keep only one
    assert rows == [
        ("fact", "cats: Barcelona, Jesus"),
    ]

def test_mule_normalization_single(tmp_path, mock_ollama):
    """Test that a single mule is normalized to 'mules: name' format."""
    db_path = tmp_path / "chat_memory_mule.db"
    user_id = "test_user"
    memory = HybridMemorySystem(str(db_path))

    with patch.dict(os.environ, {"AI_EMBEDDING_BACKEND": "ollama", "AI_MEMORY_OFFLINE": ""}):
        mock_ollama.return_value = mock_llm_response(["mules: Festus"])
        memory.store_interaction(user_id, "user", "I have a mule named Festus")

    import sqlite3
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

    assert rows == [("fact", "mules: Festus")]

def test_is_remember_command(tmp_path, mock_ollama):
    """Test that is_remember_command correctly identifies remember commands using LLM."""
    db_path = tmp_path / "chat_memory_cmd.db"
    memory = HybridMemorySystem(str(db_path))

    with patch.dict(os.environ, {"AI_EMBEDDING_BACKEND": "ollama", "AI_MEMORY_OFFLINE": ""}):
        # Mock LLM finding facts
        mock_ollama.return_value = mock_llm_response(["name: Hunter"])
        assert memory.is_remember_command("my name is Hunter")

        mock_ollama.return_value = mock_llm_response(["dog: Enola"])
        assert memory.is_remember_command("remember that my dog is Enola")

        # Mock LLM finding NOTHING (question)
        mock_ollama.return_value = mock_llm_response([])
        assert not memory.is_remember_command("what is my name?")

def test_process_remember_command(tmp_path, mock_ollama):
    """Test that process_remember_command stores facts and returns acknowledgment."""
    db_path = tmp_path / "chat_memory_process.db"
    user_id = "test_user"
    memory = HybridMemorySystem(str(db_path))

    with patch.dict(os.environ, {"AI_EMBEDDING_BACKEND": "ollama", "AI_MEMORY_OFFLINE": ""}):
        mock_ollama.return_value = mock_llm_response(["name: Hunter"])
        response = memory.process_remember_command(user_id, "my name is Hunter")

    assert "remember" in response.lower()
    assert "name: Hunter" in response

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT memory_type, content FROM explicit_memories WHERE user_id = ?",
            (user_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    assert rows == [("fact", "name: Hunter")]

def test_field_deduplication_overwrites_old_value(tmp_path, mock_ollama):
    """Test that storing a new value for the same field overwrites the old one."""
    db_path = tmp_path / "chat_memory_dedup.db"
    user_id = "test_user"
    memory = HybridMemorySystem(str(db_path))

    with patch.dict(os.environ, {"AI_EMBEDDING_BACKEND": "ollama", "AI_MEMORY_OFFLINE": ""}):
        # Initial value
        mock_ollama.return_value = mock_llm_response(["name: Hunter"])
        memory.store_interaction(user_id, "user", "my name is Hunter")

        # New value
        mock_ollama.return_value = mock_llm_response(["name: John"])
        memory.store_interaction(user_id, "user", "my name is John")

    import sqlite3
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

    assert rows == [("fact", "name: John")]
