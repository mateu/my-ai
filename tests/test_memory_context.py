import sqlite3

from my_ai import HybridMemorySystem


def test_get_context_and_format_for_prompt_smoke(tmp_path):
    """Smoke test for end-to-end explicit memory → context → prompt formatting.

    This is a lean version of the old demo() flow from the original core module.
    It ensures that:
    - explicit memories are extracted and stored
    - get_context returns them
    - format_for_prompt surfaces them under [Known Facts]
    """

    db_path = tmp_path / "chat_memory_demo.db"
    memory = HybridMemorySystem(str(db_path))
    user_id = "user_demo"

    # Turn 1: explicit memory captured via the "remember that" pattern.
    msg1 = "Remember that I work at OpenAI and I prefer concise technical answers."
    memory.store_interaction(user_id, "user", msg1)
    memory.store_interaction(user_id, "assistant", "Noted. I'll keep my responses brief and technical.")

    # A couple of additional turns to exercise the pipeline a bit.
    memory.store_interaction(user_id, "user", "How do I optimize a Python function?")
    memory.store_interaction(user_id, "assistant", "You can start by profiling your code.")

    # Use a query that closely matches the stored fact so the mock embedding
    # similarity clears the threshold used in get_context.
    query = "I work at OpenAI and I prefer concise technical answers."
    context = memory.get_context(user_id, query)
    formatted = memory.format_for_prompt(context)

    # We should at least see the Known Facts section and the OpenAI fact.
    assert "[Known Facts]" in formatted
    assert "OpenAI" in formatted
