import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class VectorMemory:
    """
    Session-scoped vector memory using FAISS IndexFlatIP (cosine similarity).

    Intentionally NOT persistent — each research query gets a clean memory
    slate. Persisting across queries caused cross-contamination where chunks
    from a previous topic would surface in unrelated searches.

    Stores: {id, url, chunk} in-memory only.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384          # all-MiniLM-L6-v2 output dim
        self.index = faiss.IndexFlatIP(self.dimension)
        self.memory: list[dict] = []  # [{id, url, chunk}]
        self.next_id = 0

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single string, return normalised (unit) vector."""
        emb = self.model.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)   # normalise ONCE, here, for everything
        return emb

    def _is_duplicate(self, emb: np.ndarray, threshold: float = 0.92) -> bool:
        """
        Check cosine similarity against stored vectors.
        Uses the same normalised embedding so scores are correct.
        Threshold 0.92 = very similar but not identical.
        """
        if self.index.ntotal == 0:
            return False
        scores, _ = self.index.search(emb, 1)
        return float(scores[0][0]) > threshold

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def add_chunks(self, url: str, chunks: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """
        Add pre-chunked text to the index.

        Args:
            url:    Source URL for provenance
            chunks: List of (chunk_id, chunk_text) from chunker.py

        Returns:
            List of (assigned_id, chunk_text) for chunks that were stored
            (duplicates are silently skipped).
        """
        stored = []

        for _chunk_id, chunk_text in chunks:
            emb = self._embed(chunk_text)

            if self._is_duplicate(emb):
                continue

            self.index.add(emb)
            self.memory.append({
                "id":    self.next_id,
                "url":   url,
                "chunk": chunk_text,
            })
            stored.append((self.next_id, chunk_text))
            self.next_id += 1

        return stored

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Semantic search over stored chunks.

        Returns:
            List of {score, url, chunk} sorted by descending similarity.
        """
        if self.index.ntotal == 0:
            return []

        emb = self._embed(query)
        k_actual = min(k, self.index.ntotal)
        scores, ids = self.index.search(emb, k_actual)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.memory):
                continue
            m = self.memory[idx]
            results.append({
                "score": float(score),
                "url":   m["url"],
                "chunk": m["chunk"],
            })

        return results

    def size(self) -> int:
        return self.index.ntotal