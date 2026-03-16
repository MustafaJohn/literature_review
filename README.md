# LitDraft — Literature Review Agent

A multi-agent system that fetches real academic papers and drafts a structured literature review. Built on LangGraph, powered by Gemini, sources from Semantic Scholar and arXiv.

---

## What it does

Given a research topic or a specific paper (title or DOI), LitDraft:

1. Fetches real papers from Semantic Scholar and arXiv — no hallucinated citations
2. Chunks and indexes abstracts into a session-scoped FAISS vector store
3. Clusters papers into thematic groups using semantic retrieval + LLM reasoning
4. Drafts a structured literature review in continuous academic prose
5. Outputs a formatted citation list in APA or IEEE style

The student reviews and deselects papers before the draft is generated, so the output reflects exactly what they want included.

---

## Architecture

```
lit_review/
│
├── api.py                  ← FastAPI server (entry point for deployment)
├── main.py                 ← CLI entry point (for local testing)
│
├── frontend/
│   └── index.html          ← Single-file web UI (4-stage workflow)
│
├── agents/
│   ├── supervisor.py       ← LangGraph router — controls node transitions
│   ├── researcher.py       ← Detects input type, fetches papers from APIs
│   ├── memory_agent.py     ← Chunks abstracts, stores in vector memory
│   ├── analyst.py          ← Retrieves chunks, clusters papers by theme
│   └── summarizer.py       ← Drafts narrative + builds citation list
│
├── tools/
│   ├── fetch_web.py        ← Semantic Scholar + arXiv fetch logic
│   └── call_llm.py         ← Gemini API wrapper
│
├── memory/
│   ├── vector_memory.py    ← Session-scoped FAISS index (fastembed, no torch)
│   └── chunker.py          ← Splits abstracts into word-bounded chunks
│
└── orchestration/
    ├── graph.py            ← LangGraph DAG definition
    └── state.py            ← LitReviewState TypedDict
```

### Agent pipeline (LangGraph DAG)

```
supervisor → research → memory → analysis → supervisor → summarize → END
                                     ↑              |
                                     └──────────────┘
                                   (loops back if need_more_info)
```

The supervisor is the only router. It reads `analysis_decision` and `next_step` to decide whether to fetch, analyse, or summarize. Each request gets a fresh `VectorMemory` instance — no cross-query contamination.

---

## Input modes

The researcher agent auto-detects the input type:

| Input | Detection | Fetch strategy |
|---|---|---|
| `transformer attention NLP` | Free-text topic | Keyword search on Semantic Scholar + arXiv |
| `"Attention Is All You Need"` | Quoted string → paper title | Resolve via Semantic Scholar, expand via references + citations |
| `10.48550/arXiv.1706.03762` | Starts with `10.` → DOI | Direct DOI lookup, expand via references + citations |

Paper-seeded fetch gives richer results because it pulls papers that are directly connected in the citation graph, not just keyword-matched.

---

## Setup

### Prerequisites

- Python 3.11+
- A Gemini API key ([get one here](https://aistudio.google.com/))

### Install

```bash
git clone https://github.com/your-username/lit_review
cd lit_review

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Running locally

### Web UI (recommended)

```bash
uvicorn api:app --reload --port 8001
```

Open `http://localhost:8001` in your browser.

### CLI (for testing the pipeline directly)

```bash
python main.py
```

You'll be prompted for a query and citation style. The pipeline output prints to the terminal.

### Test the API directly

```bash
curl -X POST http://localhost:8001/api/review \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanisms", "citation_style": "APA"}'
```

API docs available at `http://localhost:8001/docs`.

---

## API reference

### `POST /api/review`

Run the full multi-agent pipeline.

**Request body:**

```json
{
  "query": "federated learning privacy",
  "citation_style": "APA"
}
```

| Field | Type | Description |
|---|---|---|
| `query` | string | Topic, paper title, or DOI |
| `citation_style` | `"APA"` \| `"IEEE"` | Citation format for output |

**Response:**

```json
{
  "query": "federated learning privacy",
  "input_type": "topic",
  "narrative": "The literature on federated learning...",
  "citation_list": "McMahan, B., et al. (2017)...",
  "clusters": [
    {
      "theme": "Privacy Guarantees in FL",
      "description": "...",
      "paper_indices": [1, 3, 5],
      "contradictions": "..."
    }
  ],
  "sources": [
    {
      "title": "Communication-Efficient Learning...",
      "authors": "McMahan, B., Moore, E., Ramage, D.",
      "year": 2017,
      "url": "https://arxiv.org/abs/1602.05629",
      "citations": 12453,
      "is_open_access": true,
      "doi": null,
      "source": "arxiv"
    }
  ],
  "elapsed_seconds": 18.4
}
```

### `GET /api/health`

Returns `{"status": "ok", "version": "1.0.0"}`.

---

## Deploying to Render

1. Push to a GitHub repository
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Set the environment variable `GEMINI_API_KEY` in the Render dashboard
4. Render will detect the `Procfile` automatically — start command is:
   ```
   uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
5. Deploy — your app will be live at `https://your-app.onrender.com`

> **Note on free tier:** The app uses `fastembed` (ONNX Runtime) instead of `sentence-transformers` + `torch`, keeping memory usage well within Render's 512MB free tier limit.

---

## Memory architecture

`VectorMemory` is intentionally **session-scoped** — it lives only for the duration of a single request and is not persisted to disk. This was a deliberate architectural decision to prevent cross-query contamination, where abstracts from a previous search would pollute retrieval for an unrelated topic.

Each call to `build_graph()` creates a fresh `VectorMemory` instance. If you need persistence across sessions (e.g. for a returning user's saved literature set), the right approach is to store paper metadata in a database and reconstruct the index on load — not to persist the FAISS index directly.

---

## Key design decisions

**Why fastembed instead of sentence-transformers?**
`sentence-transformers` pulls in `torch` (~500MB). `fastembed` uses ONNX Runtime and ships the `BAAI/bge-small-en-v1.5` model (~25MB). Same embedding quality, a fraction of the memory footprint. Essential for free-tier deployment.

**Why no DuckDuckGo/BeautifulSoup?**
Scraping academic sites is unreliable — ResearchGate and Springer return 403s or paywalled HTML. Semantic Scholar and arXiv are purpose-built APIs that return structured JSON with abstracts, authors, citation counts, and open-access PDF links. Zero scraping, zero 404s.

**Why session-scoped memory?**
Persistent FAISS indexes across queries caused the analyst to return chunks from previous, unrelated searches. The fix was to scope memory to the request lifecycle. This trades recall across sessions for correctness within a session — the right tradeoff for a tool where each query is an independent research task.

**Why LangGraph?**
The supervisor–worker pattern with conditional routing lets the analyst signal `need_more_info` back to the supervisor cleanly, without hardcoding retry logic in every agent. The graph is stateless and deterministic — the same input will always produce the same routing path.

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web server |
| `langgraph` | Multi-agent orchestration |
| `google-genai` | Gemini LLM calls |
| `faiss-cpu` | Vector similarity search |
| `fastembed` | Lightweight embeddings (no torch) |
| `feedparser` | arXiv Atom feed parsing |
| `requests` | Semantic Scholar API calls |
| `python-dotenv` | `.env` file loading |

---

## Limitations

- **No persistent user sessions** — the review is generated fresh each time. Saving sessions requires a database layer not included here.
- **Gemini rate limits** — on free-tier Gemini, requests with many papers may hit token limits. The summarizer caps abstracts at 300 characters to mitigate this.
- **Semantic Scholar rate limits** — the free API allows ~100 requests/5 minutes. The fetch layer includes 300ms delays between queries to stay within limits.
- **arXiv coverage** — arXiv is strong for CS, physics, economics, and quantitative biology. For humanities, medicine, or social sciences, Semantic Scholar alone may return better results.
