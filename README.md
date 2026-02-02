# AI Memory

AI Memory is a small experimental project that explores a **hybrid memory system** for LLM-powered assistants.

It combines:

- **Explicit memory** – facts and preferences the user states directly (e.g. "remember that …", "my X is Y").
- **Implicit memory** – traits and patterns inferred from conversation history.
- **Vector-style retrieval** – a simple in-memory vector store to surface relevant memories for a new query.
- **Persistent storage** – local SQLite databases stored under the `data/` directory.

The project currently ships a simple CLI that lets you chat while the system learns and recalls information about you.

## Project layout

- `my_ai/memory.py` – core `HybridMemorySystem` implementation (explicit/implicit memories, vector store, SQLite schema).
- `cli.py` – interactive chat loop that uses `HybridMemorySystem`.
- `data/` – SQLite database files created at runtime (e.g. `data/chat_memory.db`, `data/chat_memory_*.db`).
- `pyproject.toml` – dependency and environment management (via `uv`).
- `Makefile` – shortcuts for common commands.

## Requirements

- Python **3.10+**
- [`uv`](https://github.com/astral-sh/uv) installed on your PATH

All Python dependencies are managed through `uv` and declared in `pyproject.toml`:

- `numpy` – used for simple vector operations in the in-memory vector store
- `openai` – used for real LLM integration
- `python-dotenv` – for loading environment variables (e.g. API keys) from a `.env` file
- `pytest` – for running the small test suite

## Setup

From the project root:

```bash
# Install / update dependencies in the uv-managed environment
uv sync
```

If you prefer `make` shortcuts, you can also run:

```bash
make sync
```

## Running the CLI

Start an interactive chat session:

```bash
# Using uv directly
uv run cli.py

# Or via Makefile shortcut
make run
```

You should see output like:

```text
🧠 Hybrid Memory System - Interactive Mode
Commands: /quit, /memories, /forget <text>, /save, /load <file>
--------------------------------------------------
```

### CLI commands

Inside the chat, you can use:

- `/quit` – save the session and exit.
- `/memories` – print stored explicit memories and inferred implicit patterns for the current user.
- `/forget <text>` – delete explicit memories whose content contains `<text>`.
- `/save` – write a timestamped snapshot of the current database into `data/`.
- `/load <file>` – load a snapshot back into the active database. If you pass just a filename like `chat_memory_20260201_164942.db`, it is resolved relative to `data/`.

### How memory works (high level)

At a high level, for each turn of the conversation:

1. The user/assistant message is stored in `conversation_turns` (SQLite).
2. User messages are scanned for explicit memory patterns, normalized into key:value style facts (e.g. `name: Hunter`, `wife: Kirsten`).
3. Every few user turns, the system infers implicit traits (e.g. communication style, technical profile) and stores them with a confidence score.
4. On each new query, the system retrieves context from both explicit and implicit memories to help the assistant respond.

All of this is handled by `HybridMemorySystem` in `ai_memory_system.py`.

## Using OpenAI and dotenv

The project already declares `openai` and `python-dotenv` as dependencies.

To wire in a real LLM:

1. Create a `.env` file in the project root with your API key, for example:

   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

2. `cli.py` already imports and uses:

   ```python
   from dotenv import load_dotenv
   from openai import OpenAI
   ```

   and calls `load_dotenv()` at startup to pick up `OPENAI_API_KEY`.

## Resetting state

All persistent memory for the CLI is stored in the local SQLite file:

- `data/chat_memory.db`

To reset the system back to a clean slate, you can simply delete this file (and any snapshots if you like):

```bash
rm -f data/chat_memory.db data/chat_memory_*.db
```

The next run will recreate `data/chat_memory.db` with a fresh schema.

## Tests

There is a small pytest that asserts explicit memories are stored as clean `key: value` facts, without duplicate messy rows.

Run the tests with:

```bash
make test
```

or directly:

```bash
uv run pytest -vv
```

## Development

Useful commands:

```bash
# Install dependencies
uv sync

# Run the CLI
uv run cli.py

# Open a Python REPL inside the project environment
make shell
```

You can explore or extend the memory behavior by editing `ai_memory_system.py` and adjusting the patterns or scoring logic for explicit and implicit memories.

## Performance Improvements

This repository includes several performance optimizations for production use:

### Numpy-Matrix Vector Store

The in-memory vector store uses a numpy matrix-backed implementation with vectorized operations:

- Contiguous float32 matrix storage for fast BLAS-backed dot products
- Vectorized similarity search using numpy operations
- Efficient top-k selection via `argpartition` (partial sorting)
- Pre-normalized embeddings stored at insertion time

This provides significant speedup over Python-loop-based similarity search, especially for larger memory stores.

### SQLite Tuning

The SQLite database includes performance optimizations:

- **WAL (Write-Ahead Logging)** mode for better concurrency
- **NORMAL synchronous** setting for improved throughput
- Indexes on common query patterns:
  - `idx_conversation_user_ts` on `conversation_turns(user_id, timestamp DESC)`
  - `idx_implicit_user` on `implicit_memories(user_id)`
  - `idx_explicit_user` on `explicit_memories(user_id)`

### Persistent Embedding Cache

A SQLite-backed persistent cache stores embeddings to avoid redundant API calls:

- Two-tier caching: in-memory dict + persistent disk cache
- Default cache location: `data/embeddings_cache.db`
- Override with `EMBEDDING_CACHE_DB` environment variable
- Keyed by model name and normalized text hash (SHA256)
- Embeddings stored as raw bytes with dtype/dimension metadata

**Example usage:**
```bash
# Use default cache location
python cli.py

# Or specify custom cache path
EMBEDDING_CACHE_DB=/path/to/cache.db python cli.py
```

The embedding cache is most beneficial when using real embedding models (e.g., with `AI_EMBEDDING_BACKEND=ollama`), but is transparent when using the default mock backend.

### Other Optimizations

- Precompiled regex patterns for explicit memory extraction
- Vectorized user_id filtering in search using boolean masks
- Reduced memory overhead with float32 embeddings

## Testing

Tests live under the standard `tests/` directory. Run them with:

- `make test` (preferred) — runs `uv run pytest -vv tests`
- Or `uv run pytest -vv tests` directly
