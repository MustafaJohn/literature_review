"""
tools/fetch_web.py

Two fetch modes:
  1. topic_fetch   — query Semantic Scholar + arXiv by keyword
  2. paper_fetch   — seed from a specific paper (DOI or title), expand via
                     its references and citing papers

Both return the same paper dict shape so the rest of the pipeline is identical.
"""

import re
import time
import logging
from typing import Optional

import requests
import feedparser

logger = logging.getLogger(__name__)

SS_SEARCH  = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_PAPER   = "https://api.semanticscholar.org/graph/v1/paper/{id}"
SS_REFS    = "https://api.semanticscholar.org/graph/v1/paper/{id}/references"
SS_CITES   = "https://api.semanticscholar.org/graph/v1/paper/{id}/citations"
ARXIV_URL  = "https://export.arxiv.org/api/query"

_HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "lit-review-agent/1.0",
}
_PAPER_FIELDS = "title,abstract,authors,year,citationCount,openAccessPdf,externalIds,url"

# 429 retry config
_SS_MAX_RETRIES  = 3
_SS_BACKOFF_BASE = 2.0   # seconds; doubles each retry (2 → 4 → 8)


# ─────────────────────────────────────────────────────────────
# Internal: normalise a paper dict from Semantic Scholar
# ─────────────────────────────────────────────────────────────

def _normalise_ss(p: dict) -> Optional[dict]:
    if not p.get("title") or not p.get("abstract"):
        return None
    open_pdf = p.get("openAccessPdf") or {}
    paper_id = p.get("paperId", "")
    return {
        "source":         "semantic_scholar",
        "paper_id":       paper_id,
        "title":          p["title"],
        "authors":        ", ".join(a["name"] for a in (p.get("authors") or [])[:4]),
        "year":           p.get("year"),
        "abstract":       p["abstract"],
        "citations":      p.get("citationCount", 0),
        "url":            open_pdf.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}",
        "is_open_access": bool(open_pdf.get("url")),
        "doi":            (p.get("externalIds") or {}).get("DOI"),
        "arxiv_id":       (p.get("externalIds") or {}).get("ArXiv"),
        "text":           p["abstract"],
    }


# ─────────────────────────────────────────────────────────────
# Internal: Semantic Scholar keyword search — with 429 retry
# ─────────────────────────────────────────────────────────────

def _ss_search(query: str, limit: int = 8) -> list[dict]:
    """
    Search Semantic Scholar with exponential backoff on 429.
    Returns [] on persistent failure so callers can fall back to arXiv.
    """
    for attempt in range(_SS_MAX_RETRIES):
        try:
            r = requests.get(
                SS_SEARCH,
                params={"query": query, "fields": _PAPER_FIELDS, "limit": limit},
                headers=_HEADERS, timeout=12,
            )
            if r.status_code == 429:
                wait = _SS_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "SS 429 on query '%s' (attempt %d/%d) — waiting %.1fs",
                    query, attempt + 1, _SS_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue   # retry
            r.raise_for_status()
            return [p for p in (_normalise_ss(x) for x in r.json().get("data", [])) if p]
        except requests.exceptions.HTTPError:
            # Non-429 HTTP error — don't retry
            logger.warning("SS search HTTP error for '%s'", query)
            return []
        except Exception as e:
            logger.warning("SS search failed (%s): %s", query, e)
            return []

    logger.warning("SS search gave up after %d retries for '%s'", _SS_MAX_RETRIES, query)
    return []


# ─────────────────────────────────────────────────────────────
# Internal: arXiv keyword search
# ─────────────────────────────────────────────────────────────

def _arxiv_search(query: str, limit: int = 5) -> list[dict]:
    try:
        r = requests.get(
            ARXIV_URL,
            params={"search_query": f"all:{query}", "start": 0,
                    "max_results": limit, "sortBy": "relevance"},
            timeout=12,
        )
        r.raise_for_status()
        feed = feedparser.parse(r.text)
    except Exception as e:
        logger.warning("arXiv search failed: %s", e)
        return []

    results = []
    for entry in feed.entries:
        arxiv_id = getattr(entry, "id", "").split("/abs/")[-1].strip()
        title    = getattr(entry, "title",   "").replace("\n", " ").strip()
        abstract = getattr(entry, "summary", "").replace("\n", " ").strip()
        if not title or not abstract or not arxiv_id:
            continue
        authors   = ", ".join(getattr(a, "name", "") for a in getattr(entry, "authors", [])[:4])
        published = getattr(entry, "published", "")
        year      = int(published[:4]) if published else None
        results.append({
            "source": "arxiv", "paper_id": None,
            "title": title, "authors": authors, "year": year,
            "abstract": abstract, "citations": None,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "is_open_access": True, "doi": None, "arxiv_id": arxiv_id,
            "text": abstract,
        })
    return results


# ─────────────────────────────────────────────────────────────
# Internal: resolve a paper by DOI or title → get its SS paper_id
# ─────────────────────────────────────────────────────────────

def _resolve_paper(query: str) -> Optional[dict]:
    """
    Given a DOI (e.g. '10.1145/...') or a paper title, return the
    Semantic Scholar paper dict for that specific paper.
    """
    is_doi     = bool(re.match(r"10\.\d{4,}/", query.strip()))
    identifier = f"DOI:{query.strip()}" if is_doi else None

    if identifier:
        try:
            r = requests.get(
                SS_PAPER.format(id=identifier),
                params={"fields": _PAPER_FIELDS},
                headers=_HEADERS, timeout=12,
            )
            if r.status_code == 200:
                return _normalise_ss(r.json())
        except Exception as e:
            logger.warning("DOI lookup failed: %s", e)

    results = _ss_search(query, limit=1)
    return results[0] if results else None


# ─────────────────────────────────────────────────────────────
# Internal: fetch references + citing papers for a paper_id
# ─────────────────────────────────────────────────────────────

def _fetch_references(paper_id: str, limit: int = 10) -> list[dict]:
    results = []
    for url_tpl in [SS_REFS, SS_CITES]:
        try:
            r = requests.get(
                url_tpl.format(id=paper_id),
                params={"fields": _PAPER_FIELDS, "limit": limit},
                headers=_HEADERS, timeout=12,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            for item in data:
                paper = item.get("citedPaper") or item.get("citingPaper") or {}
                p = _normalise_ss(paper)
                if p:
                    results.append(p)
        except Exception as e:
            logger.warning("Reference fetch failed: %s", e)
        time.sleep(0.2)
    return results


# ─────────────────────────────────────────────────────────────
# Internal: merge + deduplicate
# ─────────────────────────────────────────────────────────────

def _dedup_and_rank(papers: list[dict], max_results: int) -> list[dict]:
    seen, merged = set(), []
    for p in papers:
        key = re.sub(r"[^a-z0-9 ]", "", p["title"].lower()).strip()[:80]
        if key in seen or len(p["title"]) < 10:
            continue
        seen.add(key)
        merged.append(p)
    merged.sort(key=lambda p: (not p["is_open_access"], -(p["citations"] or -1)))
    return merged[:max_results]


# ─────────────────────────────────────────────────────────────
# Internal: build sub-queries from a topic for broader coverage
# ─────────────────────────────────────────────────────────────

def _make_subqueries(topic: str) -> list[str]:
    """
    Produce up to 3 distinct queries from a topic.
    For short/precise queries just return the topic.
    For longer queries, also add a shortened core + a broader variant.
    This avoids repeating the same failing query after a 429.
    """
    topic = topic.strip()
    queries = [topic]

    words = topic.split()
    # Add a shortened version if the query is long
    if len(words) > 5:
        queries.append(" ".join(words[:4]))
    # Add a broader variant by dropping stop-words
    _stop = {"and", "or", "in", "the", "of", "a", "an", "for", "with", "on"}
    core = " ".join(w for w in words if w.lower() not in _stop)
    if core and core != topic and core not in queries:
        queries.append(core)

    return list(dict.fromkeys(queries))[:3]  # deduplicate, cap at 3


# ─────────────────────────────────────────────────────────────
# Public: topic-based fetch
# ─────────────────────────────────────────────────────────────

def fetch_by_topic(topic: str, max_results: int = 14) -> dict:
    queries      = _make_subqueries(topic)
    all_ss, all_ax = [], []
    sources_used   = []

    for i, q in enumerate(queries):
        if i > 0:
            time.sleep(0.4)   # brief courtesy pause between sub-queries
        ss = _ss_search(q, limit=8)
        if ss and "semantic_scholar" not in sources_used:
            sources_used.append("semantic_scholar")
        all_ss.extend(ss)

        ax = _arxiv_search(q, limit=5)
        if ax and "arxiv" not in sources_used:
            sources_used.append("arxiv")
        all_ax.extend(ax)

        # Stop early if we already have more than enough raw candidates
        if len(all_ss) + len(all_ax) >= max_results * 2:
            break

    papers = _dedup_and_rank(all_ss + all_ax, max_results)

    # Surface a warning if SS was entirely absent (helps frontend badge)
    ss_failed = "semantic_scholar" not in sources_used
    if ss_failed:
        logger.warning("fetch_by_topic: Semantic Scholar returned nothing — arXiv only")

    return {
        "papers":       papers,
        "api_worked":   len(papers) > 0,
        "sources_used": sources_used,
        "seed_paper":   None,
        "ss_failed":    ss_failed,       # new — surfaced to state for frontend
    }


# ─────────────────────────────────────────────────────────────
# Public: paper-seeded fetch (DOI or title)
# ─────────────────────────────────────────────────────────────

def fetch_by_paper(query: str, max_results: int = 14) -> dict:
    seed = _resolve_paper(query)
    if not seed:
        logger.warning("Could not resolve seed paper: %s — falling back to topic search", query)
        return fetch_by_topic(query, max_results)

    logger.info("Seed paper resolved: %s", seed["title"])

    related = [seed]
    if seed.get("paper_id"):
        related += _fetch_references(seed["paper_id"], limit=12)

    topic_results = _ss_search(seed["title"], limit=6)
    papers = _dedup_and_rank(related + topic_results, max_results)

    return {
        "papers":       papers,
        "api_worked":   len(papers) > 0,
        "sources_used": ["semantic_scholar"],
        "seed_paper":   seed,
        "ss_failed":    False,
    }


# ─────────────────────────────────────────────────────────────
# Public: unified entrypoint called by researcher.py
# ─────────────────────────────────────────────────────────────

def fetch_papers(query: str, input_type: str = "topic",
                 max_results: int = 14) -> dict:
    if input_type == "paper":
        return fetch_by_paper(query, max_results)
    return fetch_by_topic(query, max_results)


# ─────────────────────────────────────────────────────────────
# LLM context formatter
# ─────────────────────────────────────────────────────────────

def papers_to_llm_context(papers: list[dict], max_abstract_chars: int = 350) -> str:
    if not papers:
        return "No papers could be fetched."
    lines = []
    for i, p in enumerate(papers, 1):
        author_year   = f"{p['authors']}, {p['year']}" if p.get("year") else p.get("authors", "")
        snippet       = (p.get("abstract") or "")[:max_abstract_chars].rstrip()
        if len(p.get("abstract") or "") > max_abstract_chars:
            snippet += "…"
        citation_note = f" [{p['citations']:,} citations]" if p.get("citations") else ""
        access_note   = " [OPEN ACCESS]" if p["is_open_access"] else ""
        lines.append(
            f"{i}. \"{p['title']}\" ({author_year}){citation_note}{access_note}\n"
            f"   URL: {p['url']}\n"
            f"   Abstract: {snippet}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Citation formatters
# ─────────────────────────────────────────────────────────────

def format_citation_apa(p: dict) -> str:
    authors = p.get("authors", "Unknown")
    year    = p.get("year", "n.d.")
    title   = p.get("title", "Untitled")
    url     = p.get("url", "")
    doi     = p.get("doi")
    loc     = f"https://doi.org/{doi}" if doi else url
    return f"{authors} ({year}). {title}. {loc}"


def format_citation_ieee(p: dict, index: int) -> str:
    authors = p.get("authors", "Unknown")
    title   = p.get("title", "Untitled")
    year    = p.get("year", "n.d.")
    url     = p.get("url", "")
    doi     = p.get("doi")
    loc     = f"doi: {doi}" if doi else f"[Online]. Available: {url}"
    return f"[{index}] {authors}, \"{title},\" {year}. {loc}"


def build_citation_list(papers: list[dict], style: str = "APA") -> str:
    if style.upper() == "IEEE":
        lines = [format_citation_ieee(p, i+1) for i, p in enumerate(papers)]
    else:
        lines = [format_citation_apa(p) for p in papers]
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("\n=== Topic fetch ===")
    r = fetch_papers("transformer attention mechanisms NLP", input_type="topic")
    print(f"Papers: {len(r['papers'])}, API worked: {r['api_worked']}, SS failed: {r.get('ss_failed')}")
    print(papers_to_llm_context(r["papers"][:3]))

    print("\n=== Citation list (APA) ===")
    print(build_citation_list(r["papers"][:3], "APA"))