import numpy as np
import faiss
from fastembed import TextEmbedding


class VectorMemory:
    """
    Session-scoped vector memory using FAISS IndexFlatIP (cosine similarity).

    Uses fastembed instead of sentence-transformers + torch.
    fastembed runs on ONNX Runtime — no torch dependency, ~50MB RAM vs ~500MB+.
    Same embedding quality, free-tier deployment compatible.

    Intentionally NOT persistent — each research query gets a clean memory
    slate. Persisting across queries caused cross-contamination where chunks
    from a previous topic would surface in unrelated searches.
    """

    # BAAI/bge-small-en-v1.5 — 384 dims, fast, accurate, ~25MB
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        self.model     = TextEmbedding(model_name=self.MODEL_NAME)
        self.dimension = 384
        self.index     = faiss.IndexFlatIP(self.dimension)
        self.memory:   list[dict] = []
        self.next_id   = 0

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single string, return normalised (unit) vector."""
        # fastembed returns a generator — consume into array
        emb = np.array(list(self.model.embed([text])), dtype="float32")
        faiss.normalize_L2(emb)
        return emb

    def _is_duplicate(self, emb: np.ndarray, threshold: float = 0.92) -> bool:
        """Cosine similarity duplicate check against stored vectors."""
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
        Duplicates (cosine similarity > 0.92) are silently skipped.
        """
        stored = []
        for _chunk_id, chunk_text in chunks:
            emb = self._embed(chunk_text)
            if self._is_duplicate(emb):
                continue
            self.index.add(emb)
            self.memory.append({"id": self.next_id, "url": url, "chunk": chunk_text})
            stored.append((self.next_id, chunk_text))
            self.next_id += 1
        return stored

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search — returns [{score, url, chunk}] sorted by similarity."""
        if self.index.ntotal == 0:
            return []
        emb      = self._embed(query)
        k_actual = min(k, self.index.ntotal)
        scores, ids = self.index.search(emb, k_actual)
        results  = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.memory):
                continue
            m = self.memory[idx]
            results.append({"score": float(score), "url": m["url"], "chunk": m["chunk"]})
        return results

    def size(self) -> int:
        return self.index.ntotal
