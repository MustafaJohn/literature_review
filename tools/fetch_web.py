"""
tools/fetch_web.py

Parallel fetch from three sources:
  - OpenAlex   (250M+ scholarly works, free, no auth, structured metadata)
  - Crossref   (150M+ DOI-registered works including book chapters — abstract scraped from DOI URL)
  - arXiv      (open-access preprints, strong for STEM/CS)

All three fire simultaneously via ThreadPoolExecutor.
Total fetch time = slowest of the three, not their sum.

No API keys required.
"""

import re
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import feedparser
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"
ARXIV_URL    = "https://export.arxiv.org/api/query"

_HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "lit-review-agent/1.0 (mailto:contact@litdraft.app)",
}

# Crossref: work types worth including (exclude datasets, components etc)
_CROSSREF_TYPES = {
    "journal-article", "book-chapter", "proceedings-article",
    "monograph", "book", "report", "posted-content",
}


# ─────────────────────────────────────────────────────────────
# Internal: reconstruct abstract from OpenAlex inverted index
# ─────────────────────────────────────────────────────────────

def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """
    OpenAlex stores abstracts as an inverted index for copyright reasons:
      {"word": [position, position, ...], ...}
    Reconstructs original text by sorting words by position.
    """
    if not inverted_index:
        return ""
    try:
        pairs = []
        for word, positions in inverted_index.items():
            for pos in positions:
                pairs.append((pos, word))
        pairs.sort(key=lambda x: x[0])
        return " ".join(word for _, word in pairs)
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# Internal: scrape abstract from a DOI landing page
# ─────────────────────────────────────────────────────────────

def _scrape_abstract_from_doi(doi: str) -> str:
    """
    Attempt to extract an abstract from a DOI landing page.

    Strategy (in order of reliability):
    1. <meta name="citation_abstract"> — Google Scholar standard, widely adopted
    2. <meta name="description"> — generic fallback, often contains abstract
    3. <section class*="abstract"> / <div class*="abstract"> — publisher HTML
    4. <p class*="abstract"> — paragraph-level fallback

    Returns empty string if blocked, timed out, or no abstract found.
    """
    url = f"https://doi.org/{doi}"
    try:
        head = requests.head(url, timeout=8, allow_redirects=True,
                             headers={"User-Agent": "lit-review-agent/1.0"})
        ctype = head.headers.get("Content-Type", "").lower()

        # Only attempt HTML pages — skip PDFs, datasets etc
        if "text/html" not in ctype:
            return ""

        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "lit-review-agent/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # 1. citation_abstract meta tag (most reliable across publishers)
        meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if meta and meta.get("content", "").strip():
            return meta["content"].strip()

        # 2. description meta tag
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and len(meta.get("content", "").strip()) > 100:
            return meta["content"].strip()

        # 3. abstract section/div
        for tag in soup.find_all(["section", "div", "p"]):
            cls = " ".join(tag.get("class", []))
            if "abstract" in cls.lower():
                text = tag.get_text(separator=" ", strip=True)
                # Strip leading "Abstract" label if present
                text = re.sub(r"^abstract\s*:?\s*", "", text, flags=re.IGNORECASE)
                if len(text) > 100:
                    return text

        return ""

    except Exception as e:
        logger.debug("DOI scrape failed for %s: %s", doi, e)
        return ""


# ─────────────────────────────────────────────────────────────
# Fetcher 1: OpenAlex
# ─────────────────────────────────────────────────────────────

def _openalex_search(query: str, limit: int = 10,
                     sort_by: str = "relevance") -> list[dict]:
    """
    Search OpenAlex works API.
    sort_by: "relevance" | "recent" | "cited"
    """
    sort_map = {
        "relevance": "relevance_score:desc",
        "recent":    "publication_year:desc",
        "cited":     "cited_by_count:desc",
    }
    oa_sort = sort_map.get(sort_by, "relevance_score:desc")

    try:
        r = requests.get(
            OPENALEX_URL,
            params={
                "search.title_and_abstract": query,
                "filter":                    "has_abstract:true",
                "sort":                      oa_sort,
                #"per_page":                  limit,
                "select":                    "id,title,abstract_inverted_index,authorships,"
                                             "publication_year,cited_by_count,doi,"
                                             "primary_location,open_access",
            },
            headers=_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("results", [])
    except Exception as e:
        logger.warning("OpenAlex search failed (%s): %s", query, e)
        return []

    results = []
    for work in data:
        title    = (work.get("title") or "").strip()
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

        if not title or not abstract or len(abstract) < 80:
            continue

        authorships = work.get("authorships") or []
        authors = ", ".join(
            a.get("author", {}).get("display_name", "")
            for a in authorships[:4]
            if a.get("author", {}).get("display_name")
        )

        year = work.get("publication_year")
        doi  = (work.get("doi") or "").replace("https://doi.org/", "") or None

        location = work.get("primary_location") or {}
        oa_info  = work.get("open_access") or {}
        is_open  = bool(oa_info.get("is_oa"))
        oa_url   = oa_info.get("oa_url")

        url = (
            oa_url
            or location.get("landing_page_url")
            or (f"https://doi.org/{doi}" if doi else None)
            or work.get("id", "")
        )

        results.append({
            "source":         "openalex",
            "paper_id":       work.get("id", ""),
            "title":          title,
            "authors":        authors or None,
            "year":           year,
            "abstract":       abstract,
            "citations":      work.get("cited_by_count"),
            "url":            url or "",
            "is_open_access": is_open,
            "doi":            doi,
            "arxiv_id":       None,
            "text":           abstract,
        })

    logger.info("[fetch] OpenAlex returned %d results for '%s'", len(results), query)
    return results


# ─────────────────────────────────────────────────────────────
# Fetcher 2: Crossref + DOI abstract scraping
# ─────────────────────────────────────────────────────────────

def _crossref_search(query: str, limit: int = 8,
                     sort_by: str = "relevance") -> list[dict]:
    """
    Search Crossref for scholarly works including book chapters and edited volumes.
    sort_by: "relevance" | "recent" | "cited"
    """
    sort_map = {
        "relevance": ("relevance",              "desc"),
        "recent":    ("published",              "desc"),
        "cited":     ("is-referenced-by-count", "desc"),
    }
    cr_sort, cr_order = sort_map.get(sort_by, ("relevance", "desc"))

    try:
        r = requests.get(
            CROSSREF_URL,
            params={
                "query.title": query,
                #"rows":        limit,
                "select":      "DOI,title,author,published,type,"
                               "container-title,is-referenced-by-count",
                "sort":        cr_sort,
                "order":       cr_order,
            },
            headers={**_HEADERS, "User-Agent": "lit-review-agent/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
    except Exception as e:
        logger.warning("Crossref search failed (%s): %s", query, e)
        return []

    # Filter to useful work types and those with a DOI
    candidates = []
    for item in items:
        work_type = item.get("type", "")
        doi       = item.get("DOI", "").strip()
        titles    = item.get("title", [])
        title     = titles[0].strip() if titles else ""

        if not doi or not title or len(title) < 10:
            continue
        if work_type not in _CROSSREF_TYPES:
            continue

        candidates.append(item)

    if not candidates:
        logger.info("[fetch] Crossref returned 0 usable candidates for '%s'", query)
        return []

    # Scrape abstracts in parallel
    def _process_item(item: dict) -> Optional[dict]:
        doi    = item.get("DOI", "").strip()
        titles = item.get("title", [])
        title  = titles[0].strip() if titles else ""

        abstract = _scrape_abstract_from_doi(doi)
        if not abstract or len(abstract) < 80:
            logger.debug("[Crossref] No abstract scraped for DOI: %s", doi)
            return None

        # Authors
        authors_raw = item.get("author", [])
        author_strs = []
        for a in authors_raw[:4]:
            name = " ".join(filter(None, [a.get("given", ""), a.get("family", "")]))
            if name:
                author_strs.append(name)
        authors = ", ".join(author_strs) or None

        # Year
        pub = item.get("published", {})
        date_parts = pub.get("date-parts", [[]])[0]
        year = date_parts[0] if date_parts else None

        citations = item.get("is-referenced-by-count")

        return {
            "source":         "crossref",
            "paper_id":       None,
            "title":          title,
            "authors":        authors,
            "year":           year,
            "abstract":       abstract,
            "citations":      citations,
            "url":            f"https://doi.org/{doi}",
            "is_open_access": False,   # conservative — we don't know
            "doi":            doi,
            "arxiv_id":       None,
            "text":           abstract,
        }

    results = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_process_item, item): item for item in candidates}
        for future in as_completed(futures):
            try:
                paper = future.result()
                if paper:
                    results.append(paper)
            except Exception as e:
                logger.debug("[Crossref] Item processing raised: %s", e)

    logger.info("[fetch] Crossref returned %d results (with abstracts) for '%s'",
                len(results), query)
    return results


# ─────────────────────────────────────────────────────────────
# Fetcher 3: arXiv
# ─────────────────────────────────────────────────────────────

def _arxiv_search(query: str, limit: int = 6,
                  sort_by: str = "relevance") -> list[dict]:
    """
    Search arXiv by keyword.
    sort_by: "relevance" | "recent" | "cited" (cited falls back to relevance — arXiv has no citation sort)
    """
    arxiv_sort_map = {
        "relevance": "relevance",
        "recent":    "submittedDate",
        "cited":     "relevance",   # arXiv has no citation count sort
    }
    arxiv_sort = arxiv_sort_map.get(sort_by, "relevance")
    try:
        r = requests.get(
            ARXIV_URL,
            params={
                "search_query": f"all:{query}",
                "start":        0,
                #"max_results":  limit,
                "sortBy":       arxiv_sort,
            },
            timeout=25,
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
        authors   = ", ".join(
            getattr(a, "name", "") for a in getattr(entry, "authors", [])[:4]
        )
        published = getattr(entry, "published", "")
        year      = int(published[:4]) if published else None
        results.append({
            "source":         "arxiv",
            "paper_id":       None,
            "title":          title,
            "authors":        authors or None,
            "year":           year,
            "abstract":       abstract,
            "citations":      None,
            "url":            f"https://arxiv.org/abs/{arxiv_id}",
            "is_open_access": True,
            "doi":            None,
            "arxiv_id":       arxiv_id,
            "text":           abstract,
        })

    logger.info("[fetch] arXiv returned %d results for '%s'", len(results), query)
    return results


# ─────────────────────────────────────────────────────────────
# Internal: deduplicate and rank
# ─────────────────────────────────────────────────────────────

def _dedup_and_rank(papers: list[dict], max_results: int) -> list[dict]:
    """
    Deduplicate by normalised title — first-seen wins.
    Merge order: OpenAlex → Crossref → arXiv.
    OpenAlex wins on overlap since it has structured metadata.
    Rank: open access first, then citation count descending.
    """
    seen, merged = set(), []
    for p in papers:
        key = re.sub(r"[^a-z0-9 ]", "", p["title"].lower()).strip()[:80]
        if key in seen or len(p["title"]) < 10:
            continue
        seen.add(key)
        merged.append(p)

    merged.sort(key=lambda p: (
        not p["is_open_access"],
        -(p["citations"] or -1),
    ))
    return merged[:max_results]


# ─────────────────────────────────────────────────────────────
# Public: unified fetch entry point
# ─────────────────────────────────────────────────────────────

def fetch_papers(query: str, input_type: str = "topic",
                 max_results: int = 14,
                 sort_by: str = "relevance") -> dict:
    """
    Fire OpenAlex, Crossref, and arXiv simultaneously.
    sort_by: "relevance" | "recent" | "cited"
    input_type kept for API compatibility.
    """
    openalex_limit  = min(max_results, 12)
    crossref_limit  = min(max_results, 8)
    arxiv_limit     = min(max_results, 6)

    openalex_results  = []
    crossref_results  = []
    arxiv_results     = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_oa       = executor.submit(_openalex_search,  query, openalex_limit, sort_by)
        future_crossref = executor.submit(_crossref_search,  query, crossref_limit, sort_by)
        future_arxiv    = executor.submit(_arxiv_search,     query, arxiv_limit,    sort_by)

        futures = {
            future_oa:       "OpenAlex",
            future_crossref: "Crossref",
            future_arxiv:    "arXiv",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "OpenAlex":
                    openalex_results = result
                elif name == "Crossref":
                    crossref_results = result
                else:
                    arxiv_results = result
                logger.info("[fetch] %s finished: %d results", name, len(result))
            except Exception as e:
                logger.warning("[fetch] %s raised an exception: %s", name, e)

    # OpenAlex first → wins on dedup
    all_papers = _dedup_and_rank(
        openalex_results + crossref_results + arxiv_results, max_results
    )

    sources_used = []
    if openalex_results:  sources_used.append("openalex")
    if crossref_results:  sources_used.append("crossref")
    if arxiv_results:     sources_used.append("arxiv")

    logger.info(
        "[fetch] Total after dedup: %d papers | sources: %s | sort: %s",
        len(all_papers), sources_used, sort_by,
    )

    return {
        "papers":       all_papers,
        "api_worked":   len(all_papers) > 0,
        "sources_used": sources_used,
        "seed_paper":   None,
        "ss_failed":    False,
        "sort_by":      sort_by,
    }


# ─────────────────────────────────────────────────────────────
# LLM context formatter
# ─────────────────────────────────────────────────────────────

def papers_to_llm_context(papers: list[dict], max_abstract_chars: int = 350) -> str:
    if not papers:
        return "No papers could be fetched."
    lines = []
    for i, p in enumerate(papers, 1):
        author_year   = f"{p['authors']}, {p['year']}" if p.get("year") else (p.get("authors") or "")
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
    authors = p.get("authors") or "Unknown"
    year    = p.get("year") or "n.d."
    title   = p.get("title") or "Untitled"
    url     = p.get("url", "")
    doi     = p.get("doi")
    loc     = f"https://doi.org/{doi}" if doi else url
    return f"{authors} ({year}). {title}. {loc}"


def format_citation_ieee(p: dict, index: int) -> str:
    authors = p.get("authors") or "Unknown"
    title   = p.get("title") or "Untitled"
    year    = p.get("year") or "n.d."
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
    import time as _time
    logging.basicConfig(level=logging.INFO)

    for query in [
        "Article 8 ECHR mass surveillance Europe",
        "transformer attention mechanisms NLP",
    ]:
        t0 = _time.time()
        r  = fetch_papers(query)
        print(f"\n[{query}]")
        print(f"Done in {_time.time()-t0:.1f}s — {len(r['papers'])} papers | sources: {r['sources_used']}")
        print(papers_to_llm_context(r["papers"][:2]))
        print("---")
