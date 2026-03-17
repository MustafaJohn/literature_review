"""
agents/analyst.py

Uses vector retrieval to surface the most relevant paper chunks,
then asks the LLM to group them into thematic clusters.

Each cluster contains:
  - theme name
  - 2-sentence description of the theme
  - which paper indices belong to it
  - any contradictions or debates within the theme
"""

import json
import logging
from orchestration.state import LitReviewState
from memory.vector_memory import VectorMemory
from tools.call_llm import call_llm
from tools.fetch_web import papers_to_llm_context

logger = logging.getLogger(__name__)

MIN_HITS      = 3
MIN_AVG_SCORE = 0.25


def analyst_agent(state: LitReviewState, vector_mem: VectorMemory) -> LitReviewState:
    query       = state["query"]
    papers      = state.get("fetched_docs", [])

    # ── 1. Vector retrieval ────────────────────────────────────
    vector_hits = vector_mem.search(query, k=12)
    logger.info("[analyst] %d vector hits", len(vector_hits))
    state["vector_results"] = vector_hits

    if not vector_hits or len(vector_hits) < MIN_HITS:
        state["analysis_decision"] = "need_more_info"
        return state

    avg_score = sum(v["score"] for v in vector_hits) / len(vector_hits)
    if avg_score < MIN_AVG_SCORE:
        state["analysis_decision"] = "need_more_info"
        return state

    # ── 2. Build paper list for clustering prompt ──────────────
    paper_context = papers_to_llm_context(papers, max_abstract_chars=300)
    n_papers      = len(papers)

    prompt = f"""You are a research librarian helping cluster academic papers for a literature review.

Research topic/query: "{query}"

Papers fetched ({n_papers} total):
{paper_context}

Group these papers into 3-5 thematic clusters. Each cluster should represent a distinct
sub-theme, methodological approach, or line of argument within the literature.

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "clusters": [
    {{
      "theme": "Short theme title (4-6 words)",
      "description": "2 sentences describing what unifies these papers and why this theme matters",
      "paper_indices": [1, 3, 5],
      "contradictions": "One sentence on any disagreements or debates among papers in this cluster, or null if none"
    }}
  ]
}}

Rules:
- paper_indices are 1-based (paper 1 = first paper listed above)
- Every paper must appear in at least one cluster
- A paper can appear in multiple clusters if genuinely cross-cutting
- Contradictions should only be noted if real tension exists between papers
- Cluster themes should be meaningfully distinct from each other"""

    try:
        raw = call_llm(prompt)
        clean = raw.replace("```json", "").replace("```", "").strip()
        match = __import__("re").search(r"\{[\s\S]*\}", clean)
        parsed = json.loads(match.group(0) if match else clean)
        state["clusters"] = parsed.get("clusters", [])
        logger.info("[analyst] %d clusters identified", len(state["clusters"]))
    except Exception as e:
        logger.error("[analyst] Clustering failed: %s", e)
        state["clusters"] = []

    # Assemble retrieval context for summarizer
    context_blocks = [f"[SOURCE: {v['url']}]\n{v['chunk']}" for v in vector_hits]
    state["final_context"]     = "\n\n".join(context_blocks)
    state["analysis_decision"] = "ready"
    return state