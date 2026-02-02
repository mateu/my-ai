# Copilot Instructions for AI Memory Project

## Project Overview

AI Memory is a hybrid memory system for LLM-powered assistants that combines explicit memory (user-stated facts), implicit memory (inferred patterns), vector-style retrieval, and persistent storage using SQLite databases.

## Technology Stack

- **Language**: Python 3.10+
- **Package Manager**: `uv` (not pip or poetry)
- **Dependencies**: numpy, openai, python-dotenv, requests, pytest
- **Database**: SQLite (stored in `data/` directory)
- **AI Backends**: Supports both OpenAI and Ollama

## Project Structure

- `my_ai/memory.py` - Core `HybridMemorySystem` implementation (explicit/implicit memories, vector store, SQLite schema)
- `cli.py` - Interactive chat loop that uses `HybridMemorySystem`
- `data/` - SQLite database files created at runtime (e.g. `data/chat_memory.db`)
- `pyproject.toml` - Dependency and environment management (via `uv`)
- `Makefile` - Shortcuts for common commands
- `tests/` - Test suite

## Development Commands

### Package Management
- **Install dependencies**: `uv sync` or `make sync`
- **NEVER use pip or poetry** - always use `uv` for dependency management

### Running the Application
- **Run with Ollama** (default): `make run` or `make run-ollama`
- **Run with OpenAI**: `make run-openai`
- **Python REPL**: `make shell`

### Testing
- **Run tests**: `make test` or `uv run pytest -vv tests`
- **Live Ollama tests**: `make tests-live`
- **All tests**: `make tests-all`

## Code Style and Conventions

- Follow existing code patterns in the repository
- Use type hints where appropriate
- Keep functions focused and well-documented
- Database operations go through the `HybridMemorySystem` class

## Environment Variables

The project uses `.env` file for configuration:
- `OPENAI_API_KEY` - For OpenAI backend
- `AI_BACKEND` - Set to "openai" or "ollama"
- `OLLAMA_URL` - URL for Ollama server (default: http://192.168.1.52:11434)
- `OLLAMA_MODEL` - Ollama model name (default: memory-bot:latest)

## Important Notes

- All persistent memory is stored in `data/chat_memory.db`
- To reset state, delete files in `data/` directory
- The system stores conversation turns, explicit memories, and implicit memories in SQLite
- Vector embeddings are kept in-memory (not persisted)
- Always test changes with `make test` before submitting
