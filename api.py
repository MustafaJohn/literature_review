"""
api.py — FastAPI wrapper for the Literature Review agent.

Endpoints:
  POST /api/review   — run the full pipeline
  GET  /api/health   — health check
  GET  /             — serves frontend/index.html

Run locally:
  uvicorn api:app --reload --port 8001

Deploy (Render/Railway):
  Set GEMINI_API_KEY, start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import time
import logging
from pathlib import Path

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

from orchestration.graph import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Literature Review Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ── Models ────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
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

class ReviewResponse(BaseModel):
    query:           str
    input_type:      str
    narrative:       str
    citation_list:   str
    clusters:        list[ClusterItem]
    sources:         list[SourceItem]
    elapsed_seconds: float
    ss_failed:       bool        # True when Semantic Scholar was rate-limited


# ── Routes ────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = frontend_path / "index.html"
    if not index.exists():
        return HTMLResponse("<h2>Frontend not found.</h2>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/review", response_model=ReviewResponse)
def run_review(req: ReviewRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")

    logger.info("Review request: %s [%s] max_results=%d",
                req.query, req.citation_style, req.max_results)
    t0 = time.time()

    try:
        graph = build_graph()
        result = graph.invoke({
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
        })
    except Exception as exc:
        logger.exception("Pipeline failed: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Done in %.2fs | %d sources | %d clusters | ss_failed=%s",
                elapsed, len(result.get("sources", [])),
                len(result.get("clusters", [])), result.get("ss_failed", False))

    return ReviewResponse(
        query           = req.query,
        input_type      = result.get("input_type", "topic"),
        narrative       = result.get("final_context", ""),
        citation_list   = result.get("citation_list", ""),
        clusters        = result.get("clusters", []),
        sources         = result.get("sources", []),
        elapsed_seconds = elapsed,
        ss_failed       = result.get("ss_failed", False),
    )
