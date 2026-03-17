"""
agents/researcher.py

Detects whether the query is a DOI/paper title or a free-text topic,
then calls the appropriate fetch function.

DOI detection: starts with "10." followed by digits and slash.
Title detection: quoted string or clearly a paper title heuristic.
Everything else: treated as a topic.
"""

import re
import logging
from orchestration.state import LitReviewState
from tools.fetch_web import fetch_papers

logger = logging.getLogger(__name__)

_DOI_RE    = re.compile(r"^10\.\d{4,}/")
_QUOTE_RE  = re.compile(r'^["\'](.+)["\']$')


def _detect_input_type(query: str) -> str:
    """
    Returns "paper" if the query looks like a DOI or specific paper title,
    "topic" otherwise.
    """
    q = query.strip()
    if _DOI_RE.match(q):
        return "paper"
    # Quoted string → explicit paper title
    if _QUOTE_RE.match(q):
        return "paper"
    # Heuristic: short query with title-case and no common topic words → paper
    words = q.split()
    if len(words) <= 12 and q[0].isupper() and ":" in q:
        return "paper"
    return "topic"


def research_agent(state: LitReviewState) -> LitReviewState:
    query      = state["query"]
    input_type = state.get("input_type") or _detect_input_type(query)

    logger.info("[researcher] Query: %s | Detected type: %s", query, input_type)

    result     = fetch_papers(query, input_type=input_type, max_results=14)
    all_papers = result["papers"]

    valid_docs = [p for p in all_papers if _is_valid(p)]
    logger.info("[researcher] %d valid papers from %s", len(valid_docs), result["sources_used"])

    state["fetched_docs"] = valid_docs
    state["input_type"]   = input_type
    state["sources"]      = [
        {
            "title":          p["title"],
            "authors":        p["authors"],
            "year":           p["year"],
            "url":            p["url"],
            "citations":      p["citations"],
            "is_open_access": p["is_open_access"],
            "source":         p["source"],
            "doi":            p.get("doi"),
        }
        for p in valid_docs
    ]

    # If seeded by a specific paper, note it in logs
    if result.get("seed_paper"):
        seed = result["seed_paper"]
        logger.info("[researcher] Seed paper: %s", seed["title"])
        state["logs"] = state.get("logs", []) + [
            f"Seeded from: \"{seed['title']}\" ({seed.get('authors','')}, {seed.get('year','')})"
        ]

    return state


def _is_valid(paper: dict) -> bool:
    text = paper.get("abstract") or paper.get("text") or ""
    return len(text.strip()) >= 80 and "\x00" not in text