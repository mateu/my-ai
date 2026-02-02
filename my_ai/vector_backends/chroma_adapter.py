"""ChromaDB adapter implementing VectorStore API."""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional


class ChromaAdapter:
    """Chroma-backed vector store compatible with the VectorStore API.
    
    Implements the minimal API used by HybridMemorySystem:
    - add(id, text, metadata, embedding)
    - search(query_emb, user_id, top_k=3) -> List[(id, score)]
    - persist()
    - metadata dict for compatibility
    
    Uses chromadb.Client with duckdb+parquet persist_directory.
    Converts Chroma distances into similarity-like scores (1 - distance).
    Handles dimension padding/trimming to match configured embedding dimension.
    """
    
    def __init__(self, dim: int = 384, persist_dir: Optional[str] = None):
        """Initialize ChromaDB adapter.
        
        Args:
            dim: Embedding dimension (default 384 for all-MiniLM-L6-v2)
            persist_dir: Directory for persistent storage (default: ./chroma_db)
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Install it with: "
                "uv sync --extras chroma or pip install chromadb"
            )
        
        self.dim = dim
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create collection for memories
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Maintain metadata dict for API compatibility
        self.metadata: Dict[str, dict] = {}
        
        # Load existing metadata from Chroma
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing metadata from Chroma collection into memory."""
        try:
            # Get all documents from collection
            results = self.collection.get(include=["metadatas"])
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    if results["metadatas"] and i < len(results["metadatas"]):
                        self.metadata[doc_id] = results["metadatas"][i] or {}
        except Exception:
            # If collection is empty or error occurs, start fresh
            pass
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize and resize embedding to match configured dimension.
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Normalized and resized embedding as float32 array
        """
        # Convert to float32
        vec = np.array(embedding, dtype=np.float32)
        
        # Pad or trim to match dimension
        if vec.shape[0] != self.dim:
            if vec.shape[0] > self.dim:
                vec = vec[:self.dim]
            else:
                pad = self.dim - vec.shape[0]
                vec = np.pad(vec, (0, pad), mode='constant')
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def add(self, id: str, text: str, metadata: dict, embedding: np.ndarray):
        """Add a document with embedding to the collection.
        
        Args:
            id: Unique identifier for the document
            text: Text content
            metadata: Metadata dictionary
            embedding: Embedding vector
        """
        # Normalize embedding
        vec = self._normalize_embedding(embedding)
        
        # Store metadata locally
        self.metadata[id] = {**metadata, "text": text}
        
        # Add to Chroma
        self.collection.upsert(
            ids=[id],
            embeddings=[vec.tolist()],
            metadatas=[{**metadata, "text": text}],
            documents=[text]
        )
    
    def search(self, query_emb: np.ndarray, user_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar documents filtered by user_id.
        
        Args:
            query_emb: Query embedding vector
            user_id: User ID to filter results
            top_k: Number of results to return
            
        Returns:
            List of (id, similarity_score) tuples, sorted by score descending
        """
        # Normalize query embedding
        query_vec = self._normalize_embedding(query_emb)
        
        # Query Chroma with filtering
        # Note: We query for more results initially to account for filtering
        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k * 10, 100),  # Query more to account for filtering
            where={"user_id": user_id} if user_id else None,
            include=["distances", "metadatas"]
        )
        
        if not results or not results["ids"] or not results["ids"][0]:
            return []
        
        # Convert Chroma distances to similarity scores
        # Chroma returns distances for cosine space, where lower is more similar
        # Convert to similarity: score = 1 - distance
        matches = []
        for i, doc_id in enumerate(results["ids"][0]):
            if i < len(results["distances"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1.0 - distance
                matches.append((doc_id, float(similarity)))
        
        # Sort by similarity (descending) and return top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
    
    def persist(self):
        """Persist the collection to disk.
        
        ChromaDB with PersistentClient auto-persists, but this method
        is provided for API compatibility and to ensure data is flushed.
        """
        # PersistentClient auto-persists, but we can be explicit
        # No explicit persist method needed with PersistentClient
        pass
