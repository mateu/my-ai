import sqlite3
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
import hashlib
from .embedding_cache import get_embedding_cache

# Precompile regexes for performance (avoid recompilation in _extract_explicit_commands)
_EXPLICIT_PATTERNS = [
    (re.compile(r"my (\w+) is (.+)", re.IGNORECASE), None),  # dynamic key:value (name: Hunter)
    (re.compile(r"remember that (.+)", re.IGNORECASE), "fact"),
    (re.compile(r"remember (?!that\b)(.+)", re.IGNORECASE), "fact"),
    (re.compile(r"(?:don't forget|note that) (.+)", re.IGNORECASE), "fact"),
    (re.compile(r"i have (.+)", re.IGNORECASE), "fact"),
    (re.compile(r"i (?:work at|live in|prefer|hate|love|need) (.+)", re.IGNORECASE), "preference"),
    (re.compile(r"i'm (?:a|an) (.+)", re.IGNORECASE), "identity"),
]

# Precompile normalization patterns
_DOGS_PATTERN_1 = re.compile(
    r"i have two dogs: one named ([^,]+),?\s*and the other(?: is)? named ([^,]+)",
    re.IGNORECASE,
)
_DOGS_PATTERN_2 = re.compile(
    r"two dogs: one named ([^,]+),?\s*and the other(?: is)? named ([^,]+)",
    re.IGNORECASE,
)
_MULTI_PATTERN_1 = re.compile(
    r"i have\s+(\w+)\s+(\w+)s\s+named\s+([^,]+),?\s*(?:and|, and)\s*([^,]+)",
    re.IGNORECASE,
)
_MULTI_PATTERN_2 = re.compile(
    r"(\w+)\s+(\w+)s\s+named\s+([^,]+),?\s*(?:and|, and)\s*([^,]+)",
    re.IGNORECASE,
)
_SUB_MY_PATTERN = re.compile(r"my (\w+) is (.+)", re.IGNORECASE)

class VectorStore:
    """Numpy-matrix-backed vector store with vectorized operations for performance.
    
    Uses a contiguous numpy float32 matrix (N x D) for normalized vectors with
    parallel metadata/id/index mappings. Search uses single BLAS-backed dot product
    for efficient similarity computation.
    """
    def __init__(self, dim: int = 384):  # all-MiniLM-L6-v2 dimension
        self.embedding_dim = dim
        # Contiguous numpy matrix for all vectors (N x D)
        self.vectors = np.empty((0, dim), dtype=np.float32)
        # Parallel arrays for metadata tracking
        self.ids: List[str] = []  # id at each index
        self.metadata: List[dict] = []  # metadata at each index
        self.id_to_index: Dict[str, int] = {}  # fast id -> index lookup
        
    def add(self, id: str, text: str, metadata: dict, embedding: np.ndarray):
        """Add vector to store with automatic padding/trimming and normalization.
        
        Args:
            id: Unique identifier for this vector
            text: The text associated with this vector
            metadata: Metadata dict (must include user_id)
            embedding: Embedding vector (will be padded/trimmed to embedding_dim)
        """
        # Ensure embedding is float32
        vec = np.array(embedding, dtype=np.float32)
        
        # Pad or trim to embedding_dim
        if vec.shape[0] != self.embedding_dim:
            if vec.shape[0] > self.embedding_dim:
                vec = vec[:self.embedding_dim]
            else:
                vec = np.pad(vec, (0, self.embedding_dim - vec.shape[0]))
        
        # Normalize the vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        # Check if updating existing entry
        if id in self.id_to_index:
            idx = self.id_to_index[id]
            self.vectors[idx] = vec
            self.metadata[idx] = {**metadata, "text": text}
        else:
            # Add new entry
            idx = len(self.ids)
            self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])
            self.ids.append(id)
            self.metadata.append({**metadata, "text": text})
            self.id_to_index[id] = idx
    
    def search(self, query_emb: np.ndarray, user_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top_k matches filtered by user_id using vectorized operations.
        
        Args:
            query_emb: Query embedding vector
            user_id: Filter results to this user
            top_k: Number of results to return
            
        Returns:
            List of (id, score) tuples sorted by score (higher is better)
        """
        if len(self.ids) == 0:
            return []
        
        # Ensure query is float32 and normalized
        query = np.array(query_emb, dtype=np.float32)
        
        # Pad or trim to embedding_dim
        if query.shape[0] != self.embedding_dim:
            if query.shape[0] > self.embedding_dim:
                query = query[:self.embedding_dim]
            else:
                query = np.pad(query, (0, self.embedding_dim - query.shape[0]))
        
        # Normalize
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Single BLAS-backed matrix-vector product for all similarities
        similarities = self.vectors.dot(query)  # (N,) array
        
        # Create user_id mask efficiently
        user_mask = np.array([meta.get("user_id") == user_id for meta in self.metadata], dtype=bool)
        
        # Apply mask by setting non-matching entries to very low value
        masked_similarities = np.where(user_mask, similarities, -np.inf)
        
        # Find indices of valid (not -inf) entries
        valid_indices = np.where(np.isfinite(masked_similarities))[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Get top_k using argpartition for O(n) performance
        k = min(top_k, len(valid_indices))
        
        if k == len(valid_indices):
            # All entries requested, just sort
            top_indices = valid_indices[np.argsort(masked_similarities[valid_indices])[-k:][::-1]]
        else:
            # Use argpartition for better performance with large N
            partition_indices = np.argpartition(masked_similarities[valid_indices], -k)[-k:]
            top_indices = valid_indices[partition_indices[np.argsort(masked_similarities[valid_indices[partition_indices]])[::-1]]]
        
        # Build results
        results = [(self.ids[idx], float(similarities[idx])) for idx in top_indices]
        
        return results

class HybridMemorySystem:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.path.join("data", "memory.db")

        # Ensure directory for the DB exists if it is relative and has a parent
        if not os.path.isabs(db_path):
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self.vector_store = VectorStore()
        self.embeddings_cache = {}  # Simple embedding cache
        
        # Mock embedding function (replace with real model)
        self.mock_vocabulary = {}
        
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite tables with performance optimizations"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Performance tuning PRAGMAs (safe and idempotent)
        c.execute('PRAGMA journal_mode = WAL')
        c.execute('PRAGMA synchronous = NORMAL')
        
        # Explicit memories
        c.execute('''CREATE TABLE IF NOT EXISTS explicit_memories
                     (id INTEGER PRIMARY KEY, user_id TEXT, memory_type TEXT, 
                      content TEXT, created_at TEXT, source TEXT)''')
        
        # Implicit memories
        c.execute('''CREATE TABLE IF NOT EXISTS implicit_memories
                     (id INTEGER PRIMARY KEY, user_id TEXT, category TEXT,
                      pattern TEXT, confidence REAL, last_observed TEXT, 
                      decay_factor REAL)''')
        
        # Conversation history for batch processing
        c.execute('''CREATE TABLE IF NOT EXISTS conversation_turns
                     (id INTEGER PRIMARY KEY, user_id TEXT, role TEXT, 
                      content TEXT, timestamp TEXT)''')
        
        # Performance indexes (idempotent with IF NOT EXISTS)
        c.execute('''CREATE INDEX IF NOT EXISTS idx_conversation_user_ts 
                     ON conversation_turns(user_id, timestamp DESC)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_implicit_user 
                     ON implicit_memories(user_id)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_explicit_user 
                     ON explicit_memories(user_id)''')
        
        conn.commit()
        conn.close()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get an embedding for `text` using a configurable backend.

        By default we use a mock, hash-based embedding so tests and offline runs
        do not depend on a real model. If AI_EMBEDDING_BACKEND="ollama" is set,
        we call an Ollama embedding model instead (default: qwen3-embedding:0.6b).
        
        Uses persistent embedding cache to avoid redundant computations.
        """
        backend = os.getenv("AI_EMBEDDING_BACKEND", "mock").lower()

        # Fast, deterministic mock for tests and simple local runs.
        if backend == "mock":
            text_norm = text.lower()
            np.random.seed(int(hashlib.md5(text_norm.encode()).hexdigest(), 16) % (2**32))
            return np.random.randn(self.vector_store.embedding_dim).astype(np.float32)

        if backend == "ollama":
            import requests

            # Allow overriding the embedding model via environment.
            # Default to qwen3-embedding:0.6b for better performance.
            model = os.getenv("OLLAMA_EMBED_MODEL", os.getenv("AI_EMBED_MODEL", "qwen3-embedding:0.6b"))
            base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
            url = f"{base_url}/api/embed"
            
            # Check persistent cache first
            cache = get_embedding_cache()
            cached_embedding = cache.get(model, text)
            if cached_embedding is not None:
                # Also store in in-process cache for fast access
                cache_key = (backend, model, text)
                self.embeddings_cache[cache_key] = cached_embedding
                return cached_embedding

            # Check in-process cache (for backwards compatibility)
            cache_key = (backend, model, text)
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]

            try:
                resp = requests.post(
                    url,
                    json={"model": model, "input": text},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                # Ollama's /api/embed returns { "embeddings": [[...]] } or {"embedding": [...]}
                emb = None
                if isinstance(data.get("embeddings"), list) and data["embeddings"]:
                    emb = data["embeddings"][0]
                elif isinstance(data.get("embedding"), list):
                    emb = data["embedding"]

                if not emb:
                    raise ValueError(f"Unexpected Ollama embed response: {data}")

                vec = np.array(emb, dtype=np.float32)
                # Optionally pad/trim to our VectorStore dim to keep interfaces simple.
                if vec.shape[0] != self.vector_store.embedding_dim:
                    if vec.shape[0] > self.vector_store.embedding_dim:
                        vec = vec[: self.vector_store.embedding_dim]
                    else:
                        pad = self.vector_store.embedding_dim - vec.shape[0]
                        vec = np.pad(vec, (0, pad))

                # Store in both caches
                self.embeddings_cache[cache_key] = vec
                cache.set(model, text, vec)
                return vec
            except Exception:
                # Fall back to mock embedding if Ollama is unavailable or misconfigured.
                text_norm = text.lower()
                np.random.seed(int(hashlib.md5(text_norm.encode()).hexdigest(), 16) % (2**32))
                return np.random.randn(self.vector_store.embedding_dim).astype(np.float32)

        # Unknown backend: fall back to mock.
        text_norm = text.lower()
        np.random.seed(int(hashlib.md5(text_norm.encode()).hexdigest(), 16) % (2**32))
        return np.random.randn(self.vector_store.embedding_dim).astype(np.float32)
    

    def _normalize_explicit_match(self, pattern: re.Pattern, mem_type: str | None, match: re.Match) -> list[str]:
        """Normalize a single regex match into one or more canonical fact strings.

        Today we always return a single fact string, but this helper gives us a
        dedicated place to evolve more complex parsing (multiple facts, richer
        structures) without bloating _extract_explicit_commands.
        """
        # Pattern-specific handling for "my X is Y" which we treat as key:value.
        if pattern.pattern.startswith("my "):
            return [f"{match.group(1)}: {match.group(2)}"]

        if mem_type:
            clause = match.group(1)
            # If a generic wrapper like "remember that ..." contains a more
            # structured "my X is Y" pattern, normalize that too.
            sub = _SUB_MY_PATTERN.match(clause)
            if sub:
                return [f"{sub.group(1)}: {sub.group(2)}"]

            # Special-case normalization for common multi-value patterns,
            # e.g. "I have two dogs: one named Enola, and the other named Glacier"
            # or "I have two cats named Barcelona and Jesus".
            dogs_match = _DOGS_PATTERN_1.match(clause) or _DOGS_PATTERN_2.match(clause)
            if dogs_match:
                first = dogs_match.group(1).strip()
                second = dogs_match.group(2).strip()
                return [f"dogs: {first}, {second}"]

            # Generic multi-entity pattern: "I have two cats named A and B" or
            # "two cats named A and B". We canonicalize as "cats: A, B" so that
            # downstream code can treat it like any other field:value fact.
            multi = _MULTI_PATTERN_1.match(clause) or _MULTI_PATTERN_2.match(clause)
            if multi:
                _, type_word, first, second = multi.groups()
                field = type_word.rstrip('s').lower() + 's'
                return [f"{field}: {first.strip()}, {second.strip()}"]

            # For generic wrappers like "remember that ...", avoid storing the
            # wrapper tokens themselves ("that", etc.) when the clause is a
            # simple prefix like "that X".
            lowered = clause.strip().lower()
            if lowered.startswith("that ") and len(clause.split()) > 1:
                return [clause.split(None, 1)[1]]

            # Default: return the clause as-is.
            return [clause]

        # Fallback, though in practice only the first pattern uses mem_type=None.
        return [match.group(1)]


    def _extract_explicit_commands(self, text: str) -> List[Dict]:
        """Parse explicit memory commands from user input"""
        # Order is important: handle more structured patterns first so we can
        # prefer key:value style facts and avoid redundant entries.
        # Uses precompiled patterns for performance.

        extracted: List[Dict] = []
        seen_contents = set()

        for pattern, mem_type in _EXPLICIT_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                contents = self._normalize_explicit_match(pattern, mem_type, match)
                for content in contents:
                    content = content.strip()
                    if not content or content in seen_contents:
                        continue
                    seen_contents.add(content)

                    extracted.append({
                        "type": mem_type or "fact",
                        "content": content,
                        "source": "user_command",
                    })
        return extracted

    def _extract_implicit_traits(self, user_id: str) -> List[Dict]:
        """Analyze recent conversation for implicit patterns (simulated LLM extraction)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get last 10 user messages
        c.execute('''SELECT content FROM conversation_turns 
                     WHERE user_id = ? AND role = 'user' 
                     ORDER BY timestamp DESC LIMIT 10''', (user_id,))
        
        recent_msgs = [row[0] for row in c.fetchall()]
        conn.close()
        
        if len(recent_msgs) < 3:
            return []
        
        # Simulated LLM extraction (replace with actual OpenAI/Anthropic call)
        # In production: send recent_msgs to LLM with extraction prompt
        text_combined = " ".join(recent_msgs).lower()
        
        traits = []
        
        # Simple rule-based extraction for MVP
        indicators = {
            "communication": {
                "prefers_concise": len(" ".join(recent_msgs)) / len(recent_msgs) < 50,
                "asks_detailed_questions": "?" in text_combined and ("how" in text_combined or "why" in text_combined),
                "technical_terminology": any(w in text_combined for w in ["python", "code", "database", "api", "function"])
            },
            "technical": {
                "python_user": "python" in text_combined,
                "web_dev": any(w in text_combined for w in ["react", "javascript", "frontend", "backend"]),
                "data_focus": any(w in text_combined for w in ["sql", "data", "analysis", "pandas"])
            }
        }
        
        for category, patterns in indicators.items():
            for pattern, detected in patterns.items():
                if detected:
                    traits.append({
                        "category": category,
                        "pattern": pattern,
                        "confidence": 0.7 + np.random.random() * 0.2  # Simulated confidence
                    })
        
        return traits
    
    def store_interaction(self, user_id: str, role: str, content: str):
        """Store conversation turn and trigger memory processing"""
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Use transaction for all writes
            c = conn.cursor()
            c.execute('''INSERT INTO conversation_turns 
                         (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)''',
                      (user_id, role, content, timestamp))
            
            # Extract explicit memories immediately
            if role == "user":
                explicit = self._extract_explicit_commands(content)
                for mem in explicit:
                    self._store_explicit_memory_transactional(c, user_id, mem)
                
                # Batch implicit extraction every 5 messages
                c.execute('''SELECT COUNT(*) FROM conversation_turns 
                            WHERE user_id = ? AND role = 'user' ''', (user_id,))
                count = c.fetchone()[0]
                
                if count % 5 == 0:
                    implicit = self._extract_implicit_traits(user_id)
                    for trait in implicit:
                        self._store_implicit_memory_transactional(c, user_id, trait)
            
            # Commit all changes at once
            conn.commit()
        finally:
            conn.close()
    
    def _store_explicit_memory_transactional(self, cursor: sqlite3.Cursor, user_id: str, memory: Dict):
        """Store factual memory with vector embedding (transactional version).

        We canonicalize simple "field: value" facts (e.g. "name: Hunter") and
        ensure there is at most one explicit memory per (user, type, field). This
        keeps the [Known Facts] block compact and reduces confusing duplicates.
        """
        content = memory["content"].strip()
        mem_type = memory["type"]

        # If this looks like a canonical "field: value" fact, deduplicate on the
        # field name so only the latest value is kept (e.g. only one "name: ...").
        field_match = re.match(r"^(\w+):\s*(.+)$", content)
        if field_match:
            field = field_match.group(1).strip().lower()
            # Remove any older facts for the same logical field for this user/type.
            cursor.execute(
                "DELETE FROM explicit_memories "
                "WHERE user_id = ? AND memory_type = ? "
                "AND lower(substr(content, 1, instr(content, ':') - 1)) = ?",
                (user_id, mem_type, field),
            )

        # Check for existing *identical* memory so we don't duplicate exact rows,
        # but allow multiple memories of the same type (e.g. multiple pets).
        cursor.execute(
            "SELECT id FROM explicit_memories "
            "WHERE user_id = ? AND memory_type = ? AND content = ?",
            (user_id, mem_type, content),
        )

        existing = cursor.fetchone()
        if existing:
            # If we already have the canonical fact for this field, skip storing
            # a redundant non-canonical variant of the same underlying fact.
            if field_match is not None:
                return

            # Otherwise, refresh timestamp / source on the existing row.
            cursor.execute(
                "UPDATE explicit_memories SET created_at = ?, source = ? "
                "WHERE id = ?",
                (datetime.now().isoformat(), memory["source"], existing[0]),
            )
        else:
            # Insert new.
            cursor.execute(
                "INSERT INTO explicit_memories "
                "(user_id, memory_type, content, created_at, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, mem_type, content, datetime.now().isoformat(), memory["source"]),
            )

        # Add to vector store for retrieval.
        emb = self._get_embedding(content)
        mem_id = f"{user_id}_explicit_{hash(content)}"
        self.vector_store.add(mem_id, content, {"user_id": user_id, "type": "explicit"}, emb)

    
    def _store_explicit_memory(self, user_id: str, memory: Dict):
            """Store factual memory with vector embedding.

            We canonicalize simple "field: value" facts (e.g. "name: Hunter") and
            ensure there is at most one explicit memory per (user, type, field). This
            keeps the [Known Facts] block compact and reduces confusing duplicates.
            """
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            content = memory["content"].strip()
            mem_type = memory["type"]

            # If this looks like a canonical "field: value" fact, deduplicate on the
            # field name so only the latest value is kept (e.g. only one "name: ...").
            field_match = re.match(r"^(\w+):\s*(.+)$", content)
            if field_match:
                field = field_match.group(1).strip().lower()
                # Remove any older facts for the same logical field for this user/type.
                c.execute(
                    "DELETE FROM explicit_memories "
                    "WHERE user_id = ? AND memory_type = ? "
                    "AND lower(substr(content, 1, instr(content, ':') - 1)) = ?",
                    (user_id, mem_type, field),
                )

            # Check for existing *identical* memory so we don't duplicate exact rows,
            # but allow multiple memories of the same type (e.g. multiple pets).
            c.execute(
                "SELECT id FROM explicit_memories "
                "WHERE user_id = ? AND memory_type = ? AND content = ?",
                (user_id, mem_type, content),
            )

            existing = c.fetchone()
            if existing:
                # If we already have the canonical fact for this field, skip storing
                # a redundant non-canonical variant of the same underlying fact.
                if field_match is not None:
                    conn.commit()
                    conn.close()
                    return

                # Otherwise, refresh timestamp / source on the existing row.
                c.execute(
                    "UPDATE explicit_memories SET created_at = ?, source = ? "
                    "WHERE id = ?",
                    (datetime.now().isoformat(), memory["source"], existing[0]),
                )
            else:
                # Insert new.
                c.execute(
                    "INSERT INTO explicit_memories "
                    "(user_id, memory_type, content, created_at, source) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, mem_type, content, datetime.now().isoformat(), memory["source"]),
                )

            conn.commit()
            conn.close()

            # Add to vector store for retrieval.
            emb = self._get_embedding(content)
            mem_id = f"{user_id}_explicit_{hash(content)}"
            self.vector_store.add(mem_id, content, {"user_id": user_id, "type": "explicit"}, emb)



    def _store_implicit_memory_transactional(self, cursor: sqlite3.Cursor, user_id: str, trait: Dict):
        """Store inferred pattern with confidence scoring (transactional version)"""
        # Check for existing similar pattern
        cursor.execute('''SELECT id, confidence FROM implicit_memories 
                     WHERE user_id = ? AND category = ? AND pattern = ?''',
                  (user_id, trait["category"], trait["pattern"]))
        
        existing = cursor.fetchone()
        if existing:
            # Bayesian update of confidence
            old_conf = existing[1]
            new_conf = old_conf + (1 - old_conf) * trait["confidence"] * 0.3
            cursor.execute('''UPDATE implicit_memories SET confidence = ?, last_observed = ?
                         WHERE id = ?''', (new_conf, datetime.now().isoformat(), existing[0]))
        else:
            cursor.execute('''INSERT INTO implicit_memories 
                         (user_id, category, pattern, confidence, last_observed, decay_factor)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (user_id, trait["category"], trait["pattern"], 
                       trait["confidence"], datetime.now().isoformat(), 0.95))

    def _store_implicit_memory(self, user_id: str, trait: Dict):
        """Store inferred pattern with confidence scoring"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Check for existing similar pattern
        c.execute('''SELECT id, confidence FROM implicit_memories 
                     WHERE user_id = ? AND category = ? AND pattern = ?''',
                  (user_id, trait["category"], trait["pattern"]))
        
        existing = c.fetchone()
        if existing:
            # Bayesian update of confidence
            old_conf = existing[1]
            new_conf = old_conf + (1 - old_conf) * trait["confidence"] * 0.3
            c.execute('''UPDATE implicit_memories SET confidence = ?, last_observed = ?
                         WHERE id = ?''', (new_conf, datetime.now().isoformat(), existing[0]))
        else:
            c.execute('''INSERT INTO implicit_memories 
                         (user_id, category, pattern, confidence, last_observed, decay_factor)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (user_id, trait["category"], trait["pattern"], 
                       trait["confidence"], datetime.now().isoformat(), 0.95))
        
        conn.commit()
        conn.close()
    
    def get_context(self, user_id: str, current_query: str) -> Dict:
        """Retrieve relevant memories for injection into prompt"""
        query_emb = self._get_embedding(current_query)
        
        # 1. Explicit memories (semantic search)
        explicit_results = self.vector_store.search(query_emb, user_id, top_k=3)
        explicit_memories = []
        for mem_id, score in explicit_results:
            if score > 0.0:  # Similarity threshold (lowered for mock embedding)
                # Use id_to_index to get the metadata
                idx = self.vector_store.id_to_index.get(mem_id)
                if idx is not None:
                    meta = self.vector_store.metadata[idx]
                    explicit_memories.append({
                        "content": meta["text"],
                        "relevance": score
                    })
        
        # Fallback: if nothing surfaced via the vector store (e.g., in tests or
        # with the mock embedding), include the most recent explicit memories
        # directly from SQLite so important facts like a user's name are always
        # available to the LLM. Even when we *do* have vector hits, we also
        # augment them with a few of the most recent facts so that very fresh
        # memories (like a newly mentioned pet) are unlikely to be dropped by a
        # noisy embedding similarity score.
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute(
            """SELECT content FROM explicit_memories
               WHERE user_id = ?
               ORDER BY datetime(created_at) DESC
               LIMIT 5""",
            (user_id,),
        )
        recent_rows = [content for (content,) in c.fetchall()]

        if not explicit_memories:
            # No vector hits at all: just use the most recent facts.
            for content in recent_rows:
                explicit_memories.append({
                    "content": content,
                    "relevance": 1.0,
                })
        else:
            # We already have some vector-surfaced facts; make sure we also
            # include a few of the latest explicit memories so that brand-new
            # facts are always candidates for the prompt.
            existing_texts = {m["content"] for m in explicit_memories}
            for content in recent_rows:
                if content not in existing_texts:
                    explicit_memories.append({
                        "content": content,
                        "relevance": 1.0,
                    })

        # Consolidate explicit facts by logical field to avoid redundant variants.
        consolidated: dict = {}
        for mem in explicit_memories:
            text = mem["content"]
            m = re.match(r"^(\w+):\s*(.+)$", text)
            if m:
                field = m.group(1).strip().lower()
                key = ("field", field)
            else:
                key = ("raw", text)

            existing = consolidated.get(key)
            if existing is None:
                consolidated[key] = mem
            else:
                # Prefer the shorter, cleaner representation for the same logical fact.
                if len(text) < len(existing["content"]):
                    consolidated[key] = mem

        explicit_memories = list(consolidated.values())



        # 2. Implicit memories (with decay and filtering)
        c.execute('''SELECT pattern, confidence, last_observed, decay_factor 
                     FROM implicit_memories WHERE user_id = ?''', (user_id,))
        
        implicit_memories = []
        now = datetime.now()
        
        for row in c.fetchall():
            pattern, conf, last_obs, decay = row
            last_dt = datetime.fromisoformat(last_obs)
            days_old = (now - last_dt).days
            
            # Apply temporal decay
            current_conf = conf * (decay ** days_old)
            
            if current_conf > 0.6:  # Threshold
                # Check relevance to current query (simplified)
                pattern_emb = self._get_embedding(pattern)
                relevance = float(np.dot(query_emb / np.linalg.norm(query_emb), 
                                       pattern_emb / np.linalg.norm(pattern_emb)))
                
                if relevance > 0.3 or current_conf > 0.8:  # High confidence bypasses relevance
                    implicit_memories.append({
                        "pattern": pattern,
                        "confidence": current_conf,
                        "category": row[0]  # You'd store category separately
                    })
        
        conn.close()
        
        return {
            "explicit_facts": explicit_memories,
            "behavioral_patterns": sorted(implicit_memories, key=lambda x: x["confidence"], reverse=True)[:2],
            "user_id": user_id
        }
    
    def format_for_prompt(self, context: Dict) -> str:
        """Format retrieved memories as text for LLM prompt"""
        sections = []
        
        if context["explicit_facts"]:
            facts_text = "\n".join([f"- {m['content']}" for m in context["explicit_facts"]])
            sections.append(f"[Known Facts]\n{facts_text}")
        
        if context["behavioral_patterns"]:
            patterns_text = "\n".join([f"- User tends to: {p['pattern']} (confidence: {p['confidence']:.2f})" 
                                     for p in context["behavioral_patterns"]])
            sections.append(f"[Observed Preferences]\n{patterns_text}")
        
        return "\n\n".join(sections) if sections else ""
