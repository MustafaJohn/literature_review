"""
api.py — FastAPI wrapper for the Literature Review agent.

Endpoints:
  POST /api/fetch      — fetch papers only (~15s)
  POST /api/cluster    — cluster selected papers (~8s, flash model)
  POST /api/summarize  — generate narrative from selected papers + clusters (~25s)
  GET  /api/health     — health check
  GET  /               — serves frontend/index.html

Run locally:
  uvicorn api:app --reload --port 8001

Deploy (Render/Railway):
  Set GEMINI_API_KEY, start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import time
import logging
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.researcher import research_agent
from agents.analyst import analyst_agent
from agents.summarizer import summarizer_agent
from tools.fetch_web import fetch_from_paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Literature Review Agent API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ── Models ────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    query:          str = Field(..., min_length=3, max_length=500)
    citation_style: str = Field(default="APA", pattern="^(APA|IEEE)$")
    max_results:    int = Field(default=14, ge=10, le=50)
    sort_by:        str = Field(default="relevance", pattern="^(relevance|recent|cited)$")

class PaperFetchRequest(BaseModel):
    url_or_doi:     str = Field(..., min_length=5, max_length=1000)
    citation_style: str = Field(default="APA", pattern="^(APA|IEEE)$")
    max_results:    int = Field(default=14, ge=10, le=50)

class SourceItem(BaseModel):
    title:          str
    authors:        str | None
    year:           int | None
    url:            str
    citations:      int | None
    is_open_access: bool
    doi:            str | None
    source:         str

class FetchResponse(BaseModel):
    query:           str
    input_type:      str
    sources:         list[SourceItem]   # no clusters — analyst runs later
    elapsed_seconds: float
    ss_failed:       bool

class ClusterRequest(BaseModel):
    query:   str
    papers:  list[dict[str, Any]]   # selected full paper dicts from frontend

class ClusterItem(BaseModel):
    theme:          str
    description:    str
    paper_indices:  list[int]
    contradictions: str | None

class ClusterResponse(BaseModel):
    clusters:        list[ClusterItem]
    elapsed_seconds: float

class SummarizeRequest(BaseModel):
    query:          str
    citation_style: str = Field(default="APA", pattern="^(APA|IEEE)$")
    papers:         list[dict[str, Any]]
    clusters:       list[dict[str, Any]]

class SummarizeResponse(BaseModel):
    narrative:       str
    citation_list:   str
    elapsed_seconds: float


# ── Routes ────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = frontend_path / "index.html"
    if not index.exists():
        return HTMLResponse("<h2>Frontend not found.</h2>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/fetch", response_model=FetchResponse)
def run_fetch(req: FetchRequest):
    """
    Step 1: fetch papers only. Fast (~15s).
    Returns all papers — user selects which ones to keep in Stage 2.
    Clustering happens in /api/cluster after user selection.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    logger.info("Fetch request: %s [%s] max_results=%d sort_by=%s",
                req.query, req.citation_style, req.max_results, req.sort_by)
    t0 = time.time()

    state: dict[str, Any] = {
        "query":             req.query,
        "input_type":        "",
        "citation_style":    req.citation_style,
        "max_results":       req.max_results,
        "sort_by":           req.sort_by,
        "fetched_docs":      [],
        "vector_results":    [],
        "graph_results":     [],
        "clusters":          [],
        "final_context":     "",
        "citation_list":     "",
        "next_step":         "",
        "analysis_decision": "",
        "sources":           [],
        "logs":              [],
        "ss_failed":         False,
    }

    try:
        state = research_agent(state)
    except Exception as exc:
        logger.exception("Fetch failed: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Fetch done in %.2fs | %d sources | ss_failed=%s",
                elapsed, len(state.get("sources", [])), state.get("ss_failed", False))

    return FetchResponse(
        query           = req.query,
        input_type      = state.get("input_type", "topic"),
        sources         = state.get("sources", []),
        elapsed_seconds = elapsed,
        ss_failed       = state.get("ss_failed", False),
    )


@app.post("/api/fetch_from_paper", response_model=FetchResponse)
def run_fetch_from_paper(req: PaperFetchRequest):
    """
    Paper-seeded fetch. Accepts a URL, DOI URL, or bare DOI.
    Extracts the DOI, resolves the seed paper via OpenAlex,
    then fetches its references + related works.
    Falls back to keyword search if DOI extraction fails.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    logger.info("Paper-seed fetch: %s max_results=%d", req.url_or_doi, req.max_results)
    t0 = time.time()

    try:
        result = fetch_from_paper(req.url_or_doi, max_results=req.max_results)
    except Exception as exc:
        logger.exception("Paper-seed fetch failed: %s", req.url_or_doi)
        raise HTTPException(status_code=500, detail=str(exc))

    # Build slim sources list for frontend (same shape as /api/fetch)
    seed   = result.get("seed_paper")
    papers = result.get("papers", [])

    sources = [
        {
            "title":          p["title"],
            "authors":        p.get("authors"),
            "year":           p.get("year"),
            "url":            p.get("url", ""),
            "citations":      p.get("citations"),
            "is_open_access": p.get("is_open_access", False),
            "source":         p.get("source", "openalex"),
            "doi":            p.get("doi"),
            # Keep abstract for summarizer — frontend sends these back
            "abstract":       p.get("abstract", ""),
            "text":           p.get("text", ""),
        }
        for p in papers
    ]

    elapsed = round(time.time() - t0, 2)
    logger.info("Paper-seed done in %.2fs | %d papers | seed: %s",
                elapsed, len(papers), seed["title"] if seed else "none")

    return FetchResponse(
        query           = seed["title"] if seed else req.url_or_doi,
        input_type      = "paper",
        sources         = sources,
        elapsed_seconds = elapsed,
        ss_failed       = False,
    )



@app.post("/api/cluster", response_model=ClusterResponse)
def run_cluster(req: ClusterRequest):
    """
    Step 2: cluster the user-selected papers into themes.
    Receives only the papers the user kept after Stage 2 selection.
    Uses gemini-2.0-flash for speed (~8s).
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    if not req.papers:
        raise HTTPException(status_code=400, detail="No papers provided.")

    logger.info("Cluster request: %s | %d papers", req.query, len(req.papers))
    t0 = time.time()

    state: dict[str, Any] = {
        "query":             req.query,
        "fetched_docs":      req.papers,
        "clusters":          [],
        "analysis_decision": "",
        "logs":              [],
    }

    try:
        state = analyst_agent(state)
    except Exception as exc:
        logger.exception("Clustering failed: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Cluster done in %.2fs | %d clusters", elapsed, len(state.get("clusters", [])))

    return ClusterResponse(
        clusters        = state.get("clusters", []),
        elapsed_seconds = elapsed,
    )


@app.post("/api/summarize", response_model=SummarizeResponse)
def run_summarize(req: SummarizeRequest):
    """
    Step 3: generate narrative + citations from selected papers and clusters.
    Receives full paper dicts and clusters from the frontend.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    if not req.papers:
        raise HTTPException(status_code=400, detail="No papers provided.")

    logger.info("Summarize request: %s [%s] | %d papers | %d clusters",
                req.query, req.citation_style, len(req.papers), len(req.clusters))
    t0 = time.time()

    state: dict[str, Any] = {
        "query":          req.query,
        "citation_style": req.citation_style,
        "fetched_docs":   req.papers,
        "clusters":       req.clusters,
        "final_context":  "",
        "citation_list":  "",
        "logs":           [],
    }

    try:
        state = summarizer_agent(state)
    except Exception as exc:
        logger.exception("Summarize failed: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Summarize done in %.2fs", elapsed)

    return SummarizeResponse(
        narrative       = state.get("final_context", ""),
        citation_list   = state.get("citation_list", ""),
        elapsed_seconds = elapsed,
    )
