"""
agents/analyst.py

Groups fetched papers into thematic clusters using the LLM directly.
Vector retrieval removed — at 10-50 papers the full abstract set fits
comfortably in Gemini's context window, so chunking + FAISS adds latency
and RAM cost for no benefit.
"""

import json
import logging
from orchestration.state import LitReviewState
from tools.call_llm import call_llm
from tools.fetch_web import papers_to_llm_context

logger = logging.getLogger(__name__)

# Clustering is a mechanical structured task — flash is fast and accurate enough.
# Pro is reserved for the summarizer which writes the narrative the user reads.
_ANALYST_MODEL = "gemini-2.5-flash"

# Minimum papers needed to attempt clustering
_MIN_PAPERS = 3


def analyst_agent(state: LitReviewState) -> LitReviewState:
    query  = state["query"]
    papers = state.get("fetched_docs", [])

    if len(papers) < _MIN_PAPERS:
        logger.warning(
            "[analyst] Only %d papers — below minimum (%d), flagging need_more_info",
            len(papers), _MIN_PAPERS,
        )
        state["analysis_decision"] = "need_more_info"
        return state

    paper_context = papers_to_llm_context(papers, max_abstract_chars=300)
    n_papers      = len(papers)

    prompt = f"""You are a research librarian helping cluster papers for a literature review.

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
        raw   = call_llm(prompt, model=_ANALYST_MODEL)
        logger.info("[analyst] Using model: %s", _ANALYST_MODEL)
        clean = raw.replace("```json", "").replace("```", "").strip()
        match = __import__("re").search(r"\{[\s\S]*\}", clean)
        parsed = json.loads(match.group(0) if match else clean)
        state["clusters"] = parsed.get("clusters", [])
        logger.info("[analyst] %d clusters identified", len(state["clusters"]))
    except Exception as e:
        logger.error("[analyst] Clustering failed: %s", e)
        state["clusters"] = []

    state["analysis_decision"] = "ready"
    return state
