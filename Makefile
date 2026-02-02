.PHONY: run run-openai run-ollama sync shell test tests tests-live tests-all

# Default Ollama URL and model for local development.
# Override as needed, e.g.:
#   OLLAMA_URL=http://localhost:11434 make run-ollama
#   OLLAMA_MODEL=deepseek-r1:1.5b make run-ollama
OLLAMA_URL ?= http://192.168.1.52:11434
OLLAMA_MODEL ?= memory-bot:latest

# Start the interactive chat CLI using the Ollama backend by default.
# `make run` is an alias for `make run-ollama`.
run: run-ollama

# Explicit Ollama chat target.
run-ollama:
	AI_BACKEND=ollama OLLAMA_URL=$(OLLAMA_URL) OLLAMA_MODEL=$(OLLAMA_MODEL) uv run cli.py

# Start the interactive chat CLI using the OpenAI backend.
# Requires OPENAI_API_KEY to be set in your environment.
# Optionally override AI_MODEL, e.g.:
#   AI_MODEL=gpt-4.1-mini make run-openai
run-openai:
	AI_BACKEND=openai uv run cli.py

sync:
	uv sync

shell:
	uv run python

# Run the fast test suite (verbose)
# Tests live under the standard tests/ directory.
test tests:
	uv run pytest -vv tests

# Run only the live Ollama integration test.
# Uses OLLAMA_URL (default above) but you can override when invoking make.
# Example:
#   OLLAMA_URL=http://localhost:11434 make tests-live
tests-live:
	AI_BACKEND=ollama OLLAMA_URL=$(OLLAMA_URL) OLLAMA_MODEL=$(OLLAMA_MODEL) uv run pytest -q tests/test_generate_response_live_ollama.py

# Run both the regular test suite and the live Ollama test.
# This is equivalent to running `make tests` followed by `make tests-live`.
tests-all: tests tests-live
