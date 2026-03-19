"""
api.py — FastAPI wrapper for the Literature Review agent.

Endpoints:
  POST /api/fetch      — fetch papers + cluster (fast, ~20s)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Literature Review Agent API", version="2.0.0")

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

class ClusterItem(BaseModel):
    theme:          str
    description:    str
    paper_indices:  list[int]
    contradictions: str | None

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
    clusters:        list[ClusterItem]
    sources:         list[SourceItem]
    elapsed_seconds: float
    ss_failed:       bool

class SummarizeRequest(BaseModel):
    query:          str
    citation_style: str = Field(default="APA", pattern="^(APA|IEEE)$")
    papers:         list[dict[str, Any]]   # full paper dicts from frontend
    clusters:       list[dict[str, Any]]

class SummarizeResponse(BaseModel):
    narrative:       str
    citation_list:   str
    elapsed_seconds: float


# ── Routes ────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = frontend_path / "index.html"
    if not index.exists():
        return HTMLResponse("<h2>Frontend not found.</h2>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/fetch", response_model=FetchResponse)
def run_fetch(req: FetchRequest):
    """
    Step 1: fetch papers and cluster them.
    Fast (~20s). Returns papers + clusters so the user can review
    them before triggering the slower summarizer.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    logger.info("Fetch request: %s [%s] max_results=%d",
                req.query, req.citation_style, req.max_results)
    t0 = time.time()

    state: dict[str, Any] = {
        "query":             req.query,
        "input_type":        "",
        "citation_style":    req.citation_style,
        "max_results":       req.max_results,
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
        state = analyst_agent(state)
    except Exception as exc:
        logger.exception("Fetch pipeline failed: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Fetch done in %.2fs | %d sources | %d clusters | ss_failed=%s",
                elapsed, len(state.get("sources", [])),
                len(state.get("clusters", [])), state.get("ss_failed", False))

    return FetchResponse(
        query           = req.query,
        input_type      = state.get("input_type", "topic"),
        clusters        = state.get("clusters", []),
        sources         = state.get("sources", []),
        elapsed_seconds = elapsed,
        ss_failed       = state.get("ss_failed", False),
    )


@app.post("/api/summarize", response_model=SummarizeResponse)
def run_summarize(req: SummarizeRequest):
    """
    Step 2: generate narrative + citations from selected papers and clusters.
    Called only when user clicks 'Generate Draft' — after they have reviewed
    and optionally deselected papers in the frontend.
    Receives full paper dicts (with abstract/text) sent back from the frontend.
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
