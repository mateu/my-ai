# Chroma Vector Backend Integration

This project now supports using Chroma as a persistent vector database backend instead of the default in-memory vector store.

## Features

- **Optional Chroma Integration**: Switch to Chroma via environment variable
- **Backward Compatible**: Falls back to in-memory store if Chroma is unavailable
- **Persistent Storage**: Store embeddings across sessions
- **Migration Script**: Migrate existing SQLite memories to Chroma

## Installation

To use the Chroma backend, install chromadb:

```bash
pip install chromadb
# or if using uv:
uv pip install chromadb
```

The dependency is already included in `pyproject.toml`, so installing the project with dependencies will include it.

## Usage

### Using Chroma Backend

Set the `VECTOR_DB` environment variable to `chroma`:

```bash
export VECTOR_DB=chroma
export CHROMA_PERSIST_DIR=./chroma_db  # optional, defaults to ./chroma_db

# Run your application
python cli.py
```

Or set it inline:

```bash
VECTOR_DB=chroma python cli.py
```

### Using Default In-Memory Store

Simply run without setting `VECTOR_DB`:

```bash
python cli.py
```

### Migrating Existing Data

If you have existing explicit memories in SQLite and want to migrate them to Chroma:

```bash
VECTOR_DB=chroma python scripts/migrate_to_chroma.py
```

This will:
1. Read all explicit memories from the SQLite database
2. Generate embeddings for each memory
3. Store them in Chroma with proper metadata
4. Persist to disk

## Configuration

### Environment Variables

- `VECTOR_DB`: Vector backend to use (`memory` or `chroma`). Default: `memory`
- `CHROMA_PERSIST_DIR`: Directory for Chroma persistence. Default: `./chroma_db`

### Programmatic Usage

```python
import os
from my_ai.memory import HybridMemorySystem

# Use Chroma
os.environ["VECTOR_DB"] = "chroma"
os.environ["CHROMA_PERSIST_DIR"] = "./my_chroma_db"

mem = HybridMemorySystem(db_path="data/memory.db")
# mem.vector_store is now a ChromaAdapter instance

# Store and retrieve memories as usual
mem.store_interaction("user123", "user", "My name is Alice")
context = mem.get_context("user123", "What's my name?")
```

## Implementation Details

### ChromaAdapter

The `ChromaAdapter` class in `my_ai/vector_backends/chroma_adapter.py` implements the same interface as the in-memory `VectorStore`:

- `add(id, text, metadata, embedding)`: Add a memory with embedding
- `search(query_emb, user_id, top_k=3)`: Search for similar memories
- `persist()`: Persist to disk (automatic with PersistentClient)
- `metadata`: Dictionary of memory metadata

### Distance to Similarity Conversion

Chroma returns distances (lower is better). The adapter converts these to similarity scores using `similarity = 1.0 - distance` for consistency with the in-memory store's cosine similarity.

### Fallback Behavior

If Chroma initialization fails (e.g., chromadb not installed, permission issues), the system automatically falls back to the in-memory `VectorStore` and logs a warning.

## Testing

Run tests including Chroma integration:

```bash
pytest tests/
```

Specific Chroma tests:

```bash
pytest tests/test_chroma_adapter.py tests/test_chroma_integration.py
```

## Notes

- Chroma data is stored in the directory specified by `CHROMA_PERSIST_DIR`
- The directory is automatically created if it doesn't exist
- Add `chroma_db/` to `.gitignore` to avoid committing database files
- The ChromaAdapter handles dimension mismatches by padding/trimming embeddings
