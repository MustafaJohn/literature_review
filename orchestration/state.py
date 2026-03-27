from typing import TypedDict, List, Dict, Any, Optional


class LitReviewState(TypedDict):
    # ── Input ──────────────────────────────────────────
    query:          str          # topic string OR paper title/DOI
    input_type:     str          # "topic" | "paper"
    citation_style: str          # "APA" | "IEEE"
    max_results:    int          # papers to fetch (10-50, default 14)

    # ── Researcher output ──────────────────────────────
    # Each dict: {title, authors, year, abstract, url,
    #              citations, is_open_access, source, doi}
    fetched_docs:   List[Dict[str, Any]]

    # ── Memory / retrieval ─────────────────────────────
    vector_results: List[Dict[str, Any]]
    graph_results:  List[Dict[str, Any]]

    # ── Thematic clusters (analyst output) ────────────
    # [{theme, description, paper_indices, contradictions}]
    clusters:       List[Dict[str, Any]]

    # ── Final outputs ──────────────────────────────────
    final_context:  str          # drafted narrative per cluster
    citation_list:  str          # formatted citations

    # ── Routing ───────────────────────────────────────
    next_step:         str
    analysis_decision: str

    # ── Metadata ──────────────────────────────────────
    sources:   List[Dict[str, Any]]
    logs:      List[str]
    ss_failed: bool              # True when SS was rate-limited / unavailable
