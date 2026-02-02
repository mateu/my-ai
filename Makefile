.PHONY: run sync shell test tests tests-live tests-all

# Default Ollama URL for live tests; can be overridden, e.g.:
#   OLLAMA_URL=http://localhost:11434 make tests-live
OLLAMA_URL ?= http://192.168.1.52:11434

run:
	uv run cli.py

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
	AI_BACKEND=ollama OLLAMA_URL=$(OLLAMA_URL) uv run pytest -q tests/test_generate_response_live_ollama.py

# Run both the regular test suite and the live Ollama test.
# This is equivalent to running `make tests` followed by `make tests-live`.
tests-all: tests tests-live
