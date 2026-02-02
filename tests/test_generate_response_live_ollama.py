import os

import pytest

import cli


pytestmark = pytest.mark.skipif(
    os.getenv("AI_BACKEND", "openai").lower() != "ollama",
    reason="Requires AI_BACKEND=ollama and a running Ollama server",
)


def test_generate_response_live_with_ollama(monkeypatch):
    """Exercise the real generate_response path against a local Ollama backend.

    This is an optional integration test. It will be skipped unless AI_BACKEND=ollama
    is set in the environment and an Ollama server is available.
    """

    # Ensure we go through the real LLM branch (not offline mode).
    monkeypatch.delenv("AI_MEMORY_OFFLINE", raising=False)

    # Provide a default model name for convenience if the caller hasn't set one.
    os.environ.setdefault("OLLAMA_MODEL", "memory-bot:latest")

    query = "What do you remember about my mules?"
    context = "[Known Facts]\n- I have two mules, one named Granite and the other is Pickles"

    result = cli.generate_response(query, context)

    # We mainly care that the call succeeds and returns non-empty text.
    assert isinstance(result, str)
    assert result.strip() != ""

    # Ideally the model mentions something about the mules; keep this loose to
    # avoid flakiness across different local models.
    lowered = result.lower()
    assert ("mule" in lowered) or ("granite" in lowered) or ("pickles" in lowered)
