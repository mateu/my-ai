import os

import cli


class _FakeMessage:
    def __init__(self, content: str = "fake response about mules"):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str = "fake response about mules"):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str = "fake response about mules"):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, calls: dict):
        self._calls = calls

    def create(self, **kwargs):  # matches OpenAI chat.completions.create signature style
        # Record the call so the test can make assertions about the prompt and parameters.
        self._calls["kwargs"] = kwargs

        # Synthesize a deterministic response that mirrors the mule facts present
        # in the system prompt/context so the test can assert on the content.
        messages = kwargs["messages"]
        system_content = messages[0]["content"]
        if "Granite" in system_content and "Pickles" in system_content:
            content = "I remember your two mules, Granite and Pickles."
        else:
            content = "test fallback response"

        return _FakeResponse(content)


class _FakeChat:
    def __init__(self, calls: dict):
        self.completions = _FakeChatCompletions(calls)


class _FakeClient:
    def __init__(self, calls: dict):
        self.chat = _FakeChat(calls)


def test_generate_response_builds_safe_prompt_and_uses_low_temperature(monkeypatch):
    """generate_response should construct a guarded system prompt and use temperature=0.2.

    We don't hit the real OpenAI API here; instead, we patch cli.get_client() to return
    a fake client whose chat.completions.create method just records the call.
    """

    calls: dict = {}

    def fake_get_client():
        return _FakeClient(calls)

    # Ensure we exercise the real LLM branch, not the offline shortcut.
    monkeypatch.delenv("AI_MEMORY_OFFLINE", raising=False)
    monkeypatch.setattr(cli, "get_client", fake_get_client)

    query = "What do you remember about my mules?"
    context = "[Known Facts]\n- I have two mules, one named Granite and the other is Pickles"

    result = cli.generate_response(query, context)

    # Our fake client returns a deterministic string that should mention both mules.
    assert "Granite" in result and "Pickles" in result

    # We should have exactly one recorded call into chat.completions.create.
    kwargs = calls["kwargs"]

    # Model and temperature should match our prompt-hardening configuration.
    assert kwargs["model"] == "gpt-3.5-turbo"
    assert kwargs["temperature"] == 0.2

    # Messages should contain a system prompt that embeds the guardrails plus the user context.
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    system_content = messages[0]["content"]

    # Guardrail text about not inventing user-specific details should be present.
    assert "ONLY source of truth" in system_content
    assert "Do NOT invent or speculate" in system_content

    # The user-specific context we pass in should also be appended to the system prompt.
    assert "Relevant context about the user" in system_content
    assert "I have two mules, one named Granite and the other is Pickles" in system_content

    # The user message should be passed through unchanged in the user role.
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == query
