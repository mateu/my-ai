.PHONY: run sync shell test tests

run:
	uv run cli.py

sync:
	uv sync

shell:
	uv run python

# Run the test suite (verbose)
# Tests live under the standard tests/ directory.
test tests:
	uv run pytest -vv tests
