"""
agents/summarizer.py

Writes a thematic literature review narrative — one paragraph per cluster —
and builds a formatted citation list.

The output is structured so the API can return both separately to the frontend.
"""

import logging
from time import sleep

from tools.call_llm import call_llm
from tools.fetch_web import papers_to_llm_context, build_citation_list
from orchestration.state import LitReviewState

logger    = logging.getLogger(__name__)
MAX_RETRY = 3


def summarizer_agent(state: LitReviewState) -> LitReviewState:
    query         = state["query"]
    papers        = state.get("fetched_docs", [])
    clusters      = state.get("clusters", [])
    citation_style = state.get("citation_style", "APA")

    paper_context = papers_to_llm_context(papers, max_abstract_chars=300)

    # Build a readable cluster summary for the prompt
    cluster_text = ""
    for i, c in enumerate(clusters, 1):
        indices       = c.get("paper_indices", [])
        cluster_papers = [papers[j-1] for j in indices if 0 < j <= len(papers)]
        paper_titles  = "; ".join(f'"{p["title"]}"' for p in cluster_papers[:4])
        contradiction = c.get("contradictions") or "None noted."
        cluster_text += (
            f"\nCluster {i}: {c['theme']}\n"
            f"Description: {c['description']}\n"
            f"Papers: {paper_titles}\n"
            f"Contradictions/debates: {contradiction}\n"
        )

    prompt = f"""You are an academic writing assistant drafting a literature review.

Research topic: "{query}"

Thematic clusters identified:
{cluster_text}

All papers with abstracts:
{paper_context}

Write a structured literature review. For each cluster write one focused paragraph that:
1. Opens by naming the theme and its significance
2. Synthesises what the key papers argue or demonstrate (cite by author, year)
3. Notes any contradictions or unresolved debates within the cluster
4. Closes with a transition sentence linking to the next theme (except the last cluster)

Important:
- Only cite papers from the list above — do not invent citations
- Use ({citation_style}-style) in-text citations: (Author, Year) for APA or [N] for IEEE
- Write in formal academic prose
- Do not use bullet points or headers — continuous paragraphs only
- Total length: 400-600 words"""

    for attempt in range(1, MAX_RETRY + 1):
        try:
            logger.info("[summarizer] LLM call attempt %d/%d", attempt, MAX_RETRY)
            narrative = call_llm(prompt)
            state["final_context"] = narrative

            # Build citation list from real paper metadata
            state["citation_list"] = build_citation_list(papers, style=citation_style)
            logger.info("[summarizer] Done.")
            return state
        except RuntimeError as e:
            logger.warning("[summarizer] Attempt %d failed: %s", attempt, e)
            if attempt < MAX_RETRY:
                sleep(2 ** attempt)
            else:
                state["final_context"] = "Error: Could not generate literature review. Check GEMINI_API_KEY."
                state["citation_list"] = build_citation_list(papers, style=citation_style)

    return state