"""
vector_store.py
Embeds document chunks with sentence-transformers (CPU) and stores in FAISS.
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class VectorStore:
    def __init__(self):
        print("Loading embedding model on CPU...")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.index = None
        self.chunks: List[Dict] = []
        self.dimension = 384   # all-MiniLM-L6-v2 output size

    def add_chunks(self, chunks: List[Dict]):
        """Embed chunks and add to FAISS index."""
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True,
                                       batch_size=16, device="cpu")
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find top-k most relevant chunks for a query."""
        if self.index is None or not self.chunks:
            return []
        q_emb = self.model.encode([query], device="cpu")
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0]) if idx >= 0
        ]

    def clear(self):
        self.index = None
        self.chunks = []

    @property
    def total_chunks(self):
        return len(self.chunks)