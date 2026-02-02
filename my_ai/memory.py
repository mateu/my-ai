import sqlite3
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
import hashlib

from .embedding_cache import EmbeddingCache

# Precompiled regex patterns for performance
_EXPLICIT_PATTERNS = [
    (re.compile(r"my (\w+) is (.+)", re.IGNORECASE), None),
    (re.compile(r"remember that (.+)", re.IGNORECASE), "fact"),
    (re.compile(r"remember (?!that\b)(.+)", re.IGNORECASE), "fact"),
    (re.compile(r"(?:don't forget|note that) (.+)", re.IGNORECASE), "fact"),
    (re.compile(r"i have (.+)", re.IGNORECASE), "fact"),
    (re.compile(r"i (?:work at|live in|prefer|hate|love|need) (.+)", re.IGNORECASE), "preference"),
    (re.compile(r"i'm (?:a|an) (.+)", re.IGNORECASE), "identity"),
]

# Precompiled patterns for normalization
_FIELD_VALUE_PATTERN = re.compile(r"^(\w+):\s*(.+)$")
_MY_X_IS_Y_PATTERN = re.compile(r"my (\w+) is (.+)", re.IGNORECASE)
_DOGS_PATTERN_1 = re.compile(
    r"i have two dogs: one named ([^,]+),?\s*and the other(?: is)? named ([^,]+)",
    re.IGNORECASE,
)
_DOGS_PATTERN_2 = re.compile(
    r"two dogs: one named ([^,]+),?\s*and the other(?: is)? named ([^,]+)",
    re.IGNORECASE,
)
_MULTI_ENTITY_1 = re.compile(
    r"i have\s+(\w+)\s+(\w+)s\s+named\s+([^,]+),?\s*(?:and|, and)\s*([^,]+)",
    re.IGNORECASE,
)
_MULTI_ENTITY_2 = re.compile(
    r"(\w+)\s+(\w+)s\s+named\s+([^,]+),?\s*(?:and|, and)\s*([^,]+)",
    re.IGNORECASE,
)


class VectorStore:
    """Numpy-matrix-backed vector store with vectorized search.
    
    This implementation uses a contiguous numpy matrix for fast BLAS-backed
    dot products and vectorized operations for efficient similarity search.
    All embeddings are normalized to unit length and stored as float32.
    """
    def __init__(self, dim: int = 384):  # all-MiniLM-L6-v2 dimension
        self.dim = dim
        # Contiguous numpy matrix (N x D) for fast dot products
        self.matrix = np.empty((0, dim), dtype=np.float32)
        # Parallel lists/dicts for metadata
        self.ids: List[str] = []  # Parallel to matrix rows
        self.metadata: List[dict] = []  # Parallel to matrix rows
        self.id_to_index: Dict[str, int] = {}  # Fast lookup
        
    def add(self, id: str, text: str, metadata: dict, embedding: np.ndarray):
        """Add a normalized vector to the store."""
        # Normalize embedding to unit length and ensure float32
        emb_norm = embedding.astype(np.float32) / np.linalg.norm(embedding)
        emb_norm = emb_norm.reshape(1, -1)  # Shape: (1, D)
        
        # Check if id already exists
        if id in self.id_to_index:
            # Update existing entry
            idx = self.id_to_index[id]
            self.matrix[idx] = emb_norm
            self.metadata[idx] = {**metadata, "text": text}
        else:
            # Append new entry - ensure dtype is preserved
            self.matrix = np.vstack([self.matrix, emb_norm]).astype(np.float32)
            self.ids.append(id)
            self.metadata.append({**metadata, "text": text})
            self.id_to_index[id] = len(self.ids) - 1
    
    def search(self, query_emb: np.ndarray, user_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top_k matches filtered by user_id using vectorized operations."""
        if self.matrix.shape[0] == 0:
            return []
        
        # Normalize query to unit length
        query_norm = query_emb.astype(np.float32) / np.linalg.norm(query_emb)
        
        # Vectorized dot product: (N x D) @ (D,) = (N,)
        scores = self.matrix.dot(query_norm)
        
        # Create boolean mask for user_id filtering
        user_mask = np.array([meta.get("user_id") == user_id 
                              for meta in self.metadata], dtype=bool)
        
        # Apply mask - set non-matching scores to -inf
        masked_scores = np.where(user_mask, scores, -np.inf)
        
        # Find top_k using argpartition for efficiency
        # (only partially sorts, which is faster than full sort)
        valid_count = np.sum(user_mask)
        if valid_count == 0:
            return []
        
        k = min(top_k, valid_count)
        
        # Get indices of top k scores
        if k < valid_count:
            # Use argpartition for partial sorting (faster)
            partition_idx = np.argpartition(masked_scores, -k)[-k:]
            # Final sort of the top k for correct ordering
            top_indices = partition_idx[np.argsort(masked_scores[partition_idx])][::-1]
        else:
            # If we need all valid items, just sort them
            top_indices = np.argsort(masked_scores)[::-1][:k]
        
        # Build result list
        results = [(self.ids[idx], float(scores[idx])) for idx in top_indices]
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
        
        # Two-tier caching: in-memory LRU + persistent SQLite
        self.embeddings_cache = {}  # In-memory cache for current session
        self.persistent_cache = EmbeddingCache()  # Persistent disk cache
        
        # Mock embedding function (replace with real model)
        self.mock_vocabulary = {}
        
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite tables with performance optimizations"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Performance tuning: WAL mode for better concurrency
        c.execute('PRAGMA journal_mode = WAL')
        # Performance tuning: NORMAL synchronous for better throughput
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
        
        # Performance indexes for common queries
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
        we call an Ollama embedding model instead (e.g. nomic-embed-text).
        
        Uses two-tier caching: in-memory dict + persistent SQLite cache.
        """
        backend = os.getenv("AI_EMBEDDING_BACKEND", "mock").lower()

        # Fast, deterministic mock for tests and simple local runs.
        # Mock embeddings are not cached persistently since they're deterministic
        if backend == "mock":
            text_norm = text.lower()
            np.random.seed(int(hashlib.md5(text_norm.encode()).hexdigest(), 16) % (2**32))
            return np.random.randn(self.vector_store.dim).astype(np.float32)

        if backend == "ollama":
            import requests

            # Allow overriding the embedding model and URL via environment.
            # Default to a general-purpose text embedding model.
            model = os.getenv("OLLAMA_EMBED_MODEL", os.getenv("AI_EMBED_MODEL", "nomic-embed-text:latest"))
            base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
            url = f"{base_url}/api/embed"

            # Check in-memory cache first (fastest)
            cache_key = (backend, model, text)
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]
            
            # Check persistent cache second
            cache_model_key = f"{backend}::{model}"
            cached = self.persistent_cache.get(cache_model_key, text)
            if cached is not None:
                # Store in memory cache for next time
                self.embeddings_cache[cache_key] = cached
                return cached

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
                if vec.shape[0] != self.vector_store.dim:
                    if vec.shape[0] > self.vector_store.dim:
                        vec = vec[: self.vector_store.dim]
                    else:
                        pad = self.vector_store.dim - vec.shape[0]
                        vec = np.pad(vec, (0, pad))

                # Store in both caches
                self.embeddings_cache[cache_key] = vec
                self.persistent_cache.set(cache_model_key, text, vec)
                return vec
            except Exception:
                # Fall back to mock embedding if Ollama is unavailable or misconfigured.
                text_norm = text.lower()
                np.random.seed(int(hashlib.md5(text_norm.encode()).hexdigest(), 16) % (2**32))
                return np.random.randn(self.vector_store.dim).astype(np.float32)

        # Unknown backend: fall back to mock.
        text_norm = text.lower()
        np.random.seed(int(hashlib.md5(text_norm.encode()).hexdigest(), 16) % (2**32))
        return np.random.randn(self.vector_store.dim).astype(np.float32)
    

    def _normalize_explicit_match(self, pattern: str, mem_type: str | None, match: re.Match) -> list[str]:
        """Normalize a single regex match into one or more canonical fact strings.

        Today we always return a single fact string, but this helper gives us a
        dedicated place to evolve more complex parsing (multiple facts, richer
        structures) without bloating _extract_explicit_commands.
        """
        # Pattern-specific handling for "my X is Y" which we treat as key:value.
        if pattern.startswith("my "):
            return [f"{match.group(1)}: {match.group(2)}"]

        if mem_type:
            clause = match.group(1)
            # If a generic wrapper like "remember that ..." contains a more
            # structured "my X is Y" pattern, normalize that too.
            sub = _MY_X_IS_Y_PATTERN.match(clause)
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
            multi = _MULTI_ENTITY_1.match(clause) or _MULTI_ENTITY_2.match(clause)
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
        # Use precompiled patterns for performance
        extracted: List[Dict] = []
        seen_contents = set()

        for pattern, mem_type in _EXPLICIT_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                # Get pattern string for normalization logic
                pattern_str = pattern.pattern
                contents = self._normalize_explicit_match(pattern_str, mem_type, match)
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
        c = conn.cursor()
        c.execute('''INSERT INTO conversation_turns 
                     (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)''',
                  (user_id, role, content, timestamp))
        conn.commit()
        
        # Extract explicit memories immediately
        if role == "user":
            explicit = self._extract_explicit_commands(content)
            for mem in explicit:
                self._store_explicit_memory(user_id, mem)
            
            # Batch implicit extraction every 5 messages
            c.execute('''SELECT COUNT(*) FROM conversation_turns 
                        WHERE user_id = ? AND role = 'user' ''', (user_id,))
            count = c.fetchone()[0]
            
            if count % 5 == 0:
                implicit = self._extract_implicit_traits(user_id)
                for trait in implicit:
                    self._store_implicit_memory(user_id, trait)
        
        conn.close()
    
    
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
            field_match = _FIELD_VALUE_PATTERN.match(content)
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
                # Find metadata by id
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
            m = _FIELD_VALUE_PATTERN.match(text)
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
