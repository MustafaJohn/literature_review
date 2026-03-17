import logging
from memory.chunker import chunk_text
from memory.vector_memory import VectorMemory
from orchestration.state import LitReviewState

logger = logging.getLogger(__name__)


def memory_agent(state: LitReviewState, vector_mem: VectorMemory) -> LitReviewState:
    logger.info("[memory] Chunking and storing %d papers…", len(state["fetched_docs"]))

    for doc in state["fetched_docs"]:
        text = doc.get("abstract") or doc.get("text") or ""
        url  = doc.get("url", "unknown")
        if not text.strip():
            continue
        chunks = chunk_text(text)
        vector_mem.add_chunks(url, chunks)

    logger.info("[memory] Vector store size: %d", vector_mem.size())
    return state