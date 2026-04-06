"""
tools/fetch_web.py (ENHANCED with Query Decomposition + Research Breakdown)

Academic paper fetcher using OpenAlex + Crossref + arXiv.
NEW: Gemini-powered query decomposition for complex research questions.
NEW: Automatic research question breakdown showing paper structure guidance.

Key Enhancements:
- Complex queries decomposed into 2-4 focused sub-topics
- Papers fetched for each sub-topic in parallel
- Results merged and ranked by relevance to original query
- Research question breakdown generated for structure guidance
- Significantly improves search quality for specific/complex questions
"""

import re
import logging
import os
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

# Crossref: work types worth including
_CROSSREF_TYPES = {
    "journal-article", "book-chapter", "proceedings-article",
    "monograph", "book", "report", "posted-content",
}

# Stop words for relevance scoring
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "as",
    "its", "it", "this", "that", "these", "those", "their", "which", "who",
    "how", "what", "when", "where", "via", "into", "within", "between",
    "about", "through", "during", "under", "over", "after", "before",
}


# ═══════════════════════════════════════════════════════════════
# NEW: RESEARCH QUESTION BREAKDOWN GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_research_breakdown(query: str, clusters: Optional[list] = None) -> dict:
    """
    Generate a structured breakdown of how to approach the research question.
    
    Provides guidance on:
    - Introduction: Background, problem statement, research objectives
    - Literature Review: Theoretical foundations, key themes, critical analysis
    - Methodology: Research design, data collection, analysis techniques
    - Expected Contributions: Theoretical/practical implications
    
    This helps users understand how to structure their actual research paper/proposal.
    
    Args:
        query: The research question/topic
        clusters: Optional list of thematic clusters from clustering step
        
    Returns:
        dict with structure: {
            "title": str,
            "sections": [{"heading": str, "content": str}, ...]
        }
    """
    # Extract key themes from clusters if available
    themes = []
    if clusters:
        themes = [c.get("theme", "") for c in clusters if c.get("theme")]
        themes = themes[:4]  # Top 4 themes
    
    # Parse main concepts from query
    query_terms = _parse_query_terms(query)
    main_concepts = sorted(query_terms, key=len, reverse=True)[:3]
    main_concepts_str = ", ".join(main_concepts) if main_concepts else query
    
    sections = [
        {
            "heading": "Introduction",
            "content": f"""• Background and context of {query}
• Problem statement: What gap exists in current understanding of {main_concepts_str}?
• Research objectives: What specifically will this research investigate?
• Research questions: Clear, answerable questions driving the investigation
• Significance: Why does this research matter to the field?"""
        },
        {
            "heading": "Literature Review",
            "content": f"""• Theoretical foundations relevant to {query}
{f"• Key themes identified: {', '.join(themes)}" if themes else f"• Current state of research in {main_concepts_str}"}
• Critical analysis: Strengths and limitations of existing approaches
• Synthesis: How different perspectives relate to each other
• Research gap identification: What remains unexplored?"""
        },
        {
            "heading": "Methodology",
            "content": f"""• Research design: Quantitative, qualitative, or mixed methods approach
• Data collection: Sources, sampling, instruments for studying {main_concepts_str}
• Analysis techniques: Methods appropriate for your research questions
• Validation: How will you ensure reliability and validity?
• Ethical considerations: Particularly relevant for {query}"""
        },
        {
            "heading": "Expected Contributions",
            "content": f"""• Theoretical contributions: New insights into {main_concepts_str}
• Practical implications: How findings can be applied in real-world contexts
• Methodological advances: Novel approaches or techniques
• Future research directions: Questions opened by this investigation
• Impact on the field: How this advances understanding of {query}"""
        },
    ]
    
    return {
        "title": query,
        "sections": sections,
    }


# ═══════════════════════════════════════════════════════════════
# QUERY DECOMPOSITION USING GEMINI
# ═══════════════════════════════════════════════════════════════

def _should_decompose_query(query: str) -> bool:
    """
    Determine if a query is complex enough to benefit from decomposition.
    
    Indicators:
    - More than 8 words
    - Contains domain-specific connectors (for, in, with, using, via)
    - Contains multiple concepts
    """
    words = query.split()
    if len(words) <= 6:
        return False
    
    # Check for complexity indicators
    complexity_markers = [
        ' for ', ' in ', ' with ', ' using ', ' via ', ' on ',
        ' and ', ' or ', ' across ', ' between '
    ]
    
    query_lower = query.lower()
    marker_count = sum(1 for marker in complexity_markers if marker in query_lower)
    
    return marker_count >= 2 or len(words) > 10


def decompose_query(query: str) -> list[str]:
    """
    Use Gemini Flash to decompose a complex research query into focused sub-topics.
    
    Returns:
        List of 2-4 focused sub-topics that together cover the original query.
        Always includes the original query as the first item.
    """
    if not _should_decompose_query(query):
        logger.info("[decompose] Query is simple, no decomposition needed")
        return [query]
    
    try:
        from tools.call_llm import call_llm
        
        prompt = f"""You are a research librarian helping to improve academic paper search quality.

Given this research query, break it down into 2-4 focused sub-topics that would help find relevant papers:

Query: {query}

Requirements:
- Each sub-topic should be a searchable phrase (3-8 words)
- Sub-topics should cover different aspects of the query
- Avoid redundancy between sub-topics
- Focus on core concepts, methods, domains, and applications
- Do NOT include the original query in your list

Output ONLY a numbered list, nothing else:
1. [sub-topic]
2. [sub-topic]
3. [sub-topic]
4. [sub-topic]

Example for "Privacy-preserving techniques in federated learning for healthcare":
1. federated learning privacy techniques
2. healthcare machine learning applications
3. differential privacy medical data
4. secure multi-party computation

Now decompose: {query}
"""
        
        # Use Flash model for speed (1-2s vs 3-4s for Pro)
        response = call_llm(prompt, model="gemini-2.5-flash")
        
        # Parse numbered list
        sub_topics = [query]  # Always include original query first
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Match patterns like "1. topic" or "1) topic" or "- topic"
            match = re.match(r'^[\d\-\*\)\.]+\s*(.+)$', line)
            if match:
                topic = match.group(1).strip()
                # Remove quotes if present
                topic = topic.strip('"\'')
                if len(topic) > 5 and topic.lower() != query.lower():
                    sub_topics.append(topic)
        
        # Cap at 5 total (original + 4 decomposed)
        sub_topics = sub_topics[:5]
        
        logger.info("[decompose] Decomposed '%s' into %d sub-topics: %s", 
                   query, len(sub_topics), sub_topics)
        
        return sub_topics
        
    except Exception as e:
        logger.warning("[decompose] Decomposition failed (%s), using original query only", e)
        return [query]


# ═══════════════════════════════════════════════════════════════
# EXISTING CODE (UNCHANGED FROM YOUR PROVIDED FILE)
# ═══════════════════════════════════════════════════════════════

def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """OpenAlex inverted index → text"""
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


def _scrape_abstract_from_doi(doi: str) -> str:
    """Scrape abstract from DOI landing page"""
    url = f"https://doi.org/{doi}"
    try:
        head = requests.head(url, timeout=8, allow_redirects=True,
                             headers={"User-Agent": "lit-review-agent/1.0"})
        ctype = head.headers.get("Content-Type", "").lower()
        if "text/html" not in ctype:
            return ""
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "lit-review-agent/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        # citation_abstract meta tag
        meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if meta and meta.get("content", "").strip():
            return meta["content"].strip()
        
        # description meta tag
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and len(meta.get("content", "").strip()) > 100:
            return meta["content"].strip()
        
        # abstract section/div/p
        for tag in soup.find_all(["section", "div", "p"]):
            cls = " ".join(tag.get("class", []))
            if "abstract" in cls.lower():
                text = tag.get_text(separator=" ", strip=True)
                text = re.sub(r"^abstract\s*:?\s*", "", text, flags=re.IGNORECASE)
                if len(text) > 100:
                    return text
        return ""
    except Exception as e:
        logger.debug("DOI scrape failed for %s: %s", doi, e)
        return ""


def _openalex_search(query: str, limit: int = 10,
                     sort_by: str = "relevance") -> list[dict]:
    """Search OpenAlex"""
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
                "per_page":                  limit,
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


def _crossref_search(query: str, limit: int = 8,
                     sort_by: str = "relevance") -> list[dict]:
    """Search Crossref + scrape abstracts"""
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
                "rows":        limit,
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
    
    def _process_item(item: dict) -> Optional[dict]:
        doi    = item.get("DOI", "").strip()
        titles = item.get("title", [])
        title  = titles[0].strip() if titles else ""
        abstract = _scrape_abstract_from_doi(doi)
        if not abstract or len(abstract) < 80:
            logger.debug("[Crossref] No abstract scraped for DOI: %s", doi)
            return None
        
        authors_raw = item.get("author", [])
        author_strs = []
        for a in authors_raw[:4]:
            name = " ".join(filter(None, [a.get("given", ""), a.get("family", "")]))
            if name:
                author_strs.append(name)
        authors = ", ".join(author_strs) or None
        
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
            "is_open_access": False,
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


def _arxiv_search(query: str, limit: int = 6,
                  sort_by: str = "relevance") -> list[dict]:
    """Search arXiv"""
    arxiv_sort_map = {
        "relevance": "relevance",
        "recent":    "submittedDate",
        "cited":     "relevance",
    }
    arxiv_sort = arxiv_sort_map.get(sort_by, "relevance")
    
    try:
        r = requests.get(
            ARXIV_URL,
            params={
                "search_query": f"all:{query}",
                "start":        0,
                "max_results":  limit,
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


def _relevance_score(paper: dict, query_terms: set[str]) -> float:
    """Compute relevance score based on query term overlap"""
    if not query_terms:
        return 0.0
    
    title    = re.sub(r"[^a-z0-9 ]", "", (paper.get("title")    or "").lower())
    abstract = re.sub(r"[^a-z0-9 ]", "", (paper.get("abstract") or "").lower())
    
    title_words    = set(title.split())
    abstract_words = set(abstract.split())
    
    title_hits    = len(query_terms & title_words)
    abstract_hits = len(query_terms & abstract_words)
    
    return (2 * title_hits + abstract_hits) / (2 * len(query_terms))


def _parse_query_terms(query: str) -> set[str]:
    """Extract meaningful terms from query"""
    tokens = re.sub(r"[^a-z0-9 ]", "", query.lower()).split()
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _dedup_and_rank(papers: list[dict], max_results: int,
                    query: str = "") -> list[dict]:
    """Deduplicate by title and rank by relevance"""
    seen, merged = set(), []
    for p in papers:
        key = re.sub(r"[^a-z0-9 ]", "", p["title"].lower()).strip()[:80]
        if key in seen or len(p["title"]) < 10:
            continue
        seen.add(key)
        merged.append(p)
    
    query_terms = _parse_query_terms(query)
    if query_terms:
        merged.sort(key=lambda p: (
            -_relevance_score(p, query_terms),
            -(p.get("citations") or -1),
        ))
    else:
        merged.sort(key=lambda p: -(p.get("citations") or -1))
    
    return merged[:max_results]


# ═══════════════════════════════════════════════════════════════
# ENHANCED: Main fetch function with query decomposition
# ═══════════════════════════════════════════════════════════════

def fetch_papers(query: str, input_type: str = "topic",
                 max_results: int = 14,
                 sort_by: str = "relevance",
                 use_decomposition: bool = True,
                 generate_breakdown: bool = False) -> dict:
    """
    Fire OpenAlex, Crossref, and arXiv simultaneously.
    
    NEW: If query is complex and use_decomposition=True:
    - Decomposes query into focused sub-topics using Gemini
    - Fetches papers for each sub-topic in parallel
    - Merges and ranks by relevance to original query
    
    NEW: If generate_breakdown=True:
    - Generates research question breakdown showing paper structure
    
    Args:
        query: Research question or topic
        input_type: "topic" (kept for API compatibility)
        max_results: Maximum papers to return
        sort_by: "relevance" | "recent" | "cited"
        use_decomposition: Enable Gemini-powered query decomposition (default: True)
        generate_breakdown: Generate research question breakdown (default: False)
    """
    
    # NEW: Query decomposition for complex queries
    if use_decomposition:
        sub_topics = decompose_query(query)
    else:
        sub_topics = [query]
    
    logger.info("[fetch] Fetching papers for %d sub-topics: %s", 
                len(sub_topics), sub_topics)
    
    # Calculate per-topic limits
    # Fetch more per topic, then deduplicate and rank globally
    openalex_limit  = min(max_results * 2, 20)
    crossref_limit  = min(max_results, 10)
    arxiv_limit     = min(max_results, 8)
    
    all_openalex_results = []
    all_crossref_results = []
    all_arxiv_results = []
    
    # Fetch for each sub-topic in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        
        for topic in sub_topics:
            futures.append(executor.submit(_openalex_search, topic, openalex_limit, sort_by))
            futures.append(executor.submit(_crossref_search, topic, crossref_limit, sort_by))
            futures.append(executor.submit(_arxiv_search, topic, arxiv_limit, sort_by))
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result:
                    continue
                
                # Identify source by first result
                if result[0]["source"] == "openalex":
                    all_openalex_results.extend(result)
                elif result[0]["source"] == "crossref":
                    all_crossref_results.extend(result)
                else:
                    all_arxiv_results.extend(result)
            except Exception as e:
                logger.warning("[fetch] Future raised exception: %s", e)
    
    # Deduplicate and rank by relevance to ORIGINAL query
    # This ensures papers relevant to the original complex question rank highest
    all_papers = _dedup_and_rank(
        all_openalex_results + all_crossref_results + all_arxiv_results,
        max_results,
        query  # Rank by relevance to original query, not sub-topics
    )
    
    sources_used = []
    if all_openalex_results:  sources_used.append("openalex")
    if all_crossref_results:  sources_used.append("crossref")
    if all_arxiv_results:     sources_used.append("arxiv")
    
    logger.info(
        "[fetch] Total after dedup: %d papers | sources: %s | sub-topics: %d | sort: %s",
        len(all_papers), sources_used, len(sub_topics), sort_by,
    )
    
    result = {
        "papers":       all_papers,
        "api_worked":   len(all_papers) > 0,
        "sources_used": sources_used,
        "sub_topics":   sub_topics if len(sub_topics) > 1 else None,  # Expose decomposition
        "seed_paper":   None,
        "ss_failed":    False,
        "sort_by":      sort_by,
    }
    
    # NEW: Generate research breakdown if requested
    if generate_breakdown:
        result["breakdown"] = generate_research_breakdown(query, clusters=None)
    
    return result


# ═══════════════════════════════════════════════════════════════
# PAPER-SEEDED FETCH (DOI/URL) - WITH AUTO-DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

def _extract_doi_from_input(raw: str) -> Optional[str]:
    """
    Extract a DOI from whatever the user pasted:
      - Bare DOI:          10.1093/ojls/gqac001
      - DOI URL:           https://doi.org/10.1093/ojls/gqac001
      - Any academic URL:  scrape the page and look for citation_doi meta tag
                           or a doi.org link in the HTML
    Returns the bare DOI string (without https://doi.org/) or None.
    """
    raw = raw.strip()
    # Bare DOI
    if re.match(r"^10\.\d{4,}/", raw):
        return raw
    # doi.org URL
    m = re.search(r"doi\.org/(10\.\d{4,}/\S+)", raw)
    if m:
        return m.group(1).rstrip(".,;)")
    # Any other URL — fetch the page and look for DOI signals
    if raw.startswith("http"):
        try:
            r = requests.get(raw, timeout=12,
                             headers={"User-Agent": "lit-review-agent/1.0"},
                             allow_redirects=True)
            r.raise_for_status()
            # Check if we ended up at a doi.org redirect
            final_url = r.url
            m = re.search(r"doi\.org/(10\.\d{4,}/\S+)", final_url)
            if m:
                return m.group(1).rstrip(".,;)")
            soup = BeautifulSoup(r.text, "html.parser")
            # citation_doi meta tag (widely adopted by publishers)
            meta = soup.find("meta", attrs={"name": "citation_doi"})
            if meta and meta.get("content", "").strip():
                doi = meta["content"].strip().replace("https://doi.org/", "")
                if re.match(r"^10\.\d{4,}/", doi):
                    return doi
            # DC.Identifier meta tag
            meta = soup.find("meta", attrs={"name": re.compile(r"DC\.Identifier", re.I)})
            if meta and meta.get("content", ""):
                m = re.search(r"10\.\d{4,}/\S+", meta["content"])
                if m:
                    return m.group(0).rstrip(".,;)")
            # Any doi.org link in the HTML
            for a in soup.find_all("a", href=True):
                m = re.search(r"doi\.org/(10\.\d{4,}/\S+)", a["href"])
                if m:
                    return m.group(1).rstrip(".,;)")
        except Exception as e:
            logger.warning("URL DOI extraction failed for %s: %s", raw, e)
    return None


def _openalex_resolve_doi(doi: str) -> Optional[dict]:
    """
    Look up a single paper in OpenAlex by DOI.
    Returns normalised paper dict or None.
    """
    try:
        r = requests.get(
            f"{OPENALEX_URL}/doi:{doi}",
            params={"select": "id,title,abstract_inverted_index,authorships,"
                              "publication_year,cited_by_count,doi,"
                              "primary_location,open_access,referenced_works,related_works"},
            headers=_HEADERS,
            timeout=15,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("OpenAlex DOI resolve failed (%s): %s", doi, e)
        return None


def _openalex_fetch_related(openalex_id: str, max_results: int = 20) -> list[dict]:
    """
    Fetch referenced works and related works for a given OpenAlex work ID.
    Both endpoints return lists of OpenAlex IDs — we then batch-fetch their metadata.
    """
    ids = set()
    for endpoint in ["references", "related_works"]:
        try:
            r = requests.get(
                f"{OPENALEX_URL}/{openalex_id.split('/')[-1]}/{endpoint}",
                params={"per_page": max_results,
                        "select":   "id,title,abstract_inverted_index,authorships,"
                                    "publication_year,cited_by_count,doi,"
                                    "primary_location,open_access"},
                headers=_HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            for work in r.json().get("results", []):
                ids.add(work.get("id", ""))
        except Exception as e:
            logger.warning("OpenAlex %s fetch failed for %s: %s", endpoint, openalex_id, e)
    
    if not ids:
        return []
    
    # Batch fetch metadata for all collected IDs
    id_filter = "|".join(list(ids)[:max_results])
    try:
        r = requests.get(
            OPENALEX_URL,
            params={"filter":   f"ids.openalex:{id_filter}",
                    "per_page": max_results,
                    "select":   "id,title,abstract_inverted_index,authorships,"
                                "publication_year,cited_by_count,doi,"
                                "primary_location,open_access"},
            headers=_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        works = r.json().get("results", [])
    except Exception as e:
        logger.warning("OpenAlex batch fetch failed: %s", e)
        return []
    
    results = []
    for work in works:
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
        
        year    = work.get("publication_year")
        doi     = (work.get("doi") or "").replace("https://doi.org/", "") or None
        oa_info = work.get("open_access") or {}
        is_open = bool(oa_info.get("is_oa"))
        oa_url  = oa_info.get("oa_url")
        location = work.get("primary_location") or {}
        
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
    
    return results


def _normalise_openalex_work(work: dict) -> Optional[dict]:
    """Normalise a raw OpenAlex work dict to the standard paper shape."""
    title    = (work.get("title") or "").strip()
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    if not title:
        return None
    
    authorships = work.get("authorships") or []
    authors = ", ".join(
        a.get("author", {}).get("display_name", "")
        for a in authorships[:4]
        if a.get("author", {}).get("display_name")
    )
    
    year    = work.get("publication_year")
    doi     = (work.get("doi") or "").replace("https://doi.org/", "") or None
    oa_info = work.get("open_access") or {}
    is_open = bool(oa_info.get("is_oa"))
    oa_url  = oa_info.get("oa_url")
    location = work.get("primary_location") or {}
    
    url = (
        oa_url
        or location.get("landing_page_url")
        or (f"https://doi.org/{doi}" if doi else None)
        or work.get("id", "")
    )
    
    return {
        "source":         "openalex",
        "paper_id":       work.get("id", ""),
        "title":          title,
        "authors":        authors or None,
        "year":           year,
        "abstract":       abstract or "",
        "citations":      work.get("cited_by_count"),
        "url":            url or "",
        "is_open_access": is_open,
        "doi":            doi,
        "arxiv_id":       None,
        "text":           abstract or "",
    }


def fetch_from_paper(url_or_doi: str, max_results: int = 14,
                     use_decomposition: bool = True,
                     generate_breakdown: bool = False) -> dict:
    """
    Seed a search from a single URL, DOI URL, or bare DOI.
    
    NEW: Paper titles are inherently complex, so decomposition is auto-enabled
    to find better related papers.
    
    NEW: Can optionally generate research breakdown.
    
    Steps:
    1. Extract DOI from whatever the user pasted
    2. Resolve the seed paper via OpenAlex DOI lookup
    3. Fetch references + related works from OpenAlex
    4. Also run keyword search on seed paper title for broader coverage
       - WITH DECOMPOSITION: Title is decomposed into sub-topics for better coverage
    5. Merge, dedup, return
    
    Falls back to keyword search on the raw input if DOI extraction fails.
    """
    # Step 1: extract DOI
    doi = _extract_doi_from_input(url_or_doi)
    if not doi:
        logger.warning("Could not extract DOI from '%s' — falling back to keyword search", url_or_doi)
        return fetch_papers(url_or_doi, max_results=max_results, 
                          use_decomposition=use_decomposition,
                          generate_breakdown=generate_breakdown)
    
    logger.info("[paper-seed] Resolved DOI: %s", doi)
    
    # Step 2: resolve seed paper
    raw_seed = _openalex_resolve_doi(doi)
    seed_paper = _normalise_openalex_work(raw_seed) if raw_seed else None
    if not seed_paper:
        logger.warning("[paper-seed] OpenAlex could not resolve DOI %s — keyword fallback", doi)
        return fetch_papers(url_or_doi, max_results=max_results,
                          use_decomposition=use_decomposition,
                          generate_breakdown=generate_breakdown)
    
    logger.info("[paper-seed] Seed paper: %s", seed_paper["title"])
    
    # Step 3 + 4: fetch related works AND multi-source keyword search in parallel
    openalex_id   = raw_seed.get("id", "")
    related_limit = min(max_results * 2, 30)
    kw_limit      = min(max_results, 10)
    
    # NEW: Paper titles are complex → use decomposition for keyword search
    # This significantly improves related paper discovery
    seed_title = seed_paper["title"]
    
    # If decomposition enabled, decompose the title for better coverage
    if use_decomposition:
        sub_topics = decompose_query(seed_title)
        logger.info("[paper-seed] Decomposed title into %d sub-topics for search", len(sub_topics))
    else:
        # Fallback: extract top terms from title
        title_terms = _parse_query_terms(seed_title)
        kw_query = " ".join(sorted(title_terms, key=len, reverse=True)[:6])
        sub_topics = [kw_query]
    
    related_results  = []
    all_oa_kw_results    = []
    all_crossref_results = []
    all_arxiv_results    = []
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        
        # Fetch related works from OpenAlex
        futures.append(("related", executor.submit(_openalex_fetch_related, openalex_id, related_limit)))
        
        # Fetch keyword search results for each sub-topic
        for topic in sub_topics:
            futures.append(("oa-kw", executor.submit(_openalex_search, topic, kw_limit)))
            futures.append(("crossref", executor.submit(_crossref_search, topic, kw_limit)))
            futures.append(("arxiv", executor.submit(_arxiv_search, topic, kw_limit)))
        
        for name, future in futures:
            try:
                result = future.result()
                if name == "related":
                    related_results = result
                elif name == "oa-kw":
                    all_oa_kw_results.extend(result)
                elif name == "crossref":
                    all_crossref_results.extend(result)
                else:
                    all_arxiv_results.extend(result)
                logger.info("[paper-seed] %s fetch done: %d results", name, len(result))
            except Exception as e:
                logger.warning("[paper-seed] %s fetch failed: %s", name, e)
    
    # Seed paper first, then OpenAlex related, then keyword results from all sources
    all_papers = _dedup_and_rank(
        [seed_paper] + related_results + all_oa_kw_results + all_crossref_results + all_arxiv_results,
        max_results,
        seed_title,   # rank by relevance to seed paper title
    )
    
    sources_used = ["openalex"]
    if all_crossref_results:  sources_used.append("crossref")
    if all_arxiv_results:     sources_used.append("arxiv")
    
    logger.info("[paper-seed] Total after dedup: %d papers | sources: %s | decomposed: %s",
                len(all_papers), sources_used, use_decomposition)
    
    result = {
        "papers":       all_papers,
        "api_worked":   len(all_papers) > 0,
        "sources_used": sources_used,
        "seed_paper":   seed_paper,
        "seed_title":   seed_title,  # NEW: Expose seed title for frontend
        "sub_topics":   sub_topics if len(sub_topics) > 1 else None,  # NEW: Expose decomposition
        "ss_failed":    False,
        "sort_by":      "relevance",
    }
    
    # NEW: Generate research breakdown if requested
    if generate_breakdown:
        result["breakdown"] = generate_research_breakdown(seed_title, clusters=None)
    
    return result


# ═══════════════════════════════════════════════════════════════
# Utility functions (unchanged)
# ═══════════════════════════════════════════════════════════════

def papers_to_llm_context(papers: list[dict], max_abstract_chars: int = 350) -> str:
    """Format papers for LLM context"""
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


def format_citation_apa(p: dict) -> str:
    """Format citation in APA style"""
    authors = p.get("authors") or "Unknown"
    year    = p.get("year") or "n.d."
    title   = p.get("title") or "Untitled"
    url     = p.get("url", "")
    doi     = p.get("doi")
    loc     = f"https://doi.org/{doi}" if doi else url
    return f"{authors} ({year}). {title}. {loc}"


def format_citation_ieee(p: dict, index: int) -> str:
    """Format citation in IEEE style"""
    authors = p.get("authors") or "Unknown"
    title   = p.get("title") or "Untitled"
    year    = p.get("year") or "n.d."
    url     = p.get("url", "")
    doi     = p.get("doi")
    loc     = f"doi: {doi}" if doi else f"[Online]. Available: {url}"
    return f"[{index}] {authors}, \"{title},\" {year}. {loc}"


def build_citation_list(papers: list[dict], style: str = "APA") -> str:
    """Build formatted citation list"""
    if style.upper() == "IEEE":
        lines = [format_citation_ieee(p, i+1) for i, p in enumerate(papers)]
    else:
        lines = [format_citation_apa(p) for p in papers]
    return "\n\n".join(lines)
