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
from urllib.parse import unquote, urlparse
from tools.fetch_web import fetch_papers

logger = logging.getLogger(__name__)

_DOI_RE    = re.compile(r"^10\.\d{4,}/")
_QUOTE_RE  = re.compile(r'^["\'](.+)["\']$')
_DOI_ANY_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)

DEFAULT_MAX_RESULTS = 14

def _resolve_reference_link(link: str) -> str:
    """
    Convert a pasted reference link into a cleaner seed query.
    Priority:
      1) DOI if present in URL/text
      2) arXiv ID if present
      3) fallback to the original string
    """
    raw = (link or "").strip()
    if not raw:
        return ""

    doi_match = _DOI_ANY_RE.search(raw)
    if doi_match:
        return doi_match.group(1).rstrip(").,;")

    arxiv_match = _ARXIV_URL_RE.search(raw)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1).replace(".pdf", "").strip("/")
        return f"arXiv:{arxiv_id}"

    # Non-DOI publisher links: keep URL form to maximize retrievability
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return unquote(raw)

    return raw

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
    input_mode = state.get("input_mode") or "topic"
    reference_link = state.get("reference_link")
    query = state["query"]

    if input_mode == "link":
        seed_value = reference_link or query
        query = _resolve_reference_link(seed_value)
        state["query"] = query
        state["logs"] = state.get("logs", []) + [f"Seed link provided: {seed_value}"]
        input_type = "paper"
    else:
        input_type = state.get("input_type") or _detect_input_type(query)

    max_results = state.get("max_results") or DEFAULT_MAX_RESULTS
    sort_by     = state.get("sort_by") or "relevance"

    logger.info(
        "[researcher] Query: %s | Mode: %s | Type: %s | Max results: %d | Sort: %s",
        query, input_mode, input_type, max_results, sort_by,
    )

    result     = fetch_papers(query, input_type=input_type, max_results=max_results, sort_by=sort_by)
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
