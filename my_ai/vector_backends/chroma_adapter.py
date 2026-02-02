from typing import List, Tuple, Dict, Optional
import os
import numpy as np
import chromadb

class ChromaAdapter:
    """Adapter that exposes add/search/persist and a metadata dict to mimic VectorStore."""

    def __init__(self, collection_name: str = "memories", persist_directory: Optional[str] = None, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.dim = embedding_dim  # Alias for compatibility with VectorStore
        persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        # Use the modern PersistentClient instead of deprecated Settings
        self.client = chromadb.PersistentClient(path=persist_directory)
        # get_or_create_collection ensures idempotence
        self.collection = self.client.get_or_create_collection(name=collection_name)
        # keep a lightweight in-memory metadata cache for compatibility
        self.metadata: Dict[str, Dict] = {}

    def add(self, id: str, text: str, metadata: dict, embedding: np.ndarray):
        """Add a single memory. embedding -> list[float]"""
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.size != self.embedding_dim:
            # pad/trim to embedding_dim
            if emb.size > self.embedding_dim:
                emb = emb[: self.embedding_dim]
            else:
                pad = self.embedding_dim - emb.size
                emb = np.pad(emb, (0, pad))
        # store with metadata; ensure user_id present for filtering
        meta = dict(metadata)
        meta.setdefault("text", text)
        self.collection.add(ids=[id], embeddings=[emb.tolist()], metadatas=[meta], documents=[text])
        # update local metadata cache
        self.metadata[id] = meta

    def _fetch_metadata_if_missing(self, id: str):
        if id in self.metadata:
            return self.metadata[id]
        try:
            res = self.collection.get(ids=[id], include=["metadatas", "documents"])
            metadatas = res.get("metadatas", []) or []
            docs = res.get("documents", []) or []
            if metadatas:
                meta = metadatas[0]
                if "text" not in meta and docs:
                    meta["text"] = docs[0]
                self.metadata[id] = meta
                return meta
        except Exception:
            pass
        return {}

    def search(self, query_emb: np.ndarray, user_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Query Chroma and return list of (id, similarity_score). We convert distances -> similarity (1 - distance)."""
        q = np.asarray(query_emb, dtype=np.float32)
        # ensure dimensional shape compatibility
        if q.size != self.embedding_dim:
            q = q[: self.embedding_dim] if q.size > self.embedding_dim else np.pad(q, (0, self.embedding_dim - q.size))
        # Use 'where' metadata filter to restrict to user_id (Chroma supports simple metadata queries)
        where = {"user_id": user_id} if user_id is not None else None
        try:
            results = self.collection.query(query_embeddings=[q.tolist()], n_results=top_k, where=where)
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]  # lower = closer
            # Convert to similarity-like score where higher is better.
            # For cosine-like distances, similarity ~= 1 - distance.
            # This is a pragmatic conversion — verify on your data.
            out = []
            for _id, dist in zip(ids, distances):
                try:
                    score = 1.0 - float(dist)
                except Exception:
                    score = float(dist)
                out.append((_id, score))
            return out
        except Exception as e:
            import warnings
            warnings.warn(f"Chroma search failed, returning empty results: {e}")
            return []

    def persist(self):
        # PersistentClient automatically persists, so this is a no-op for compatibility
        pass
