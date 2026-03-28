# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup environment (Python 3.11 required)
conda create -n LangGraph-Perfilador-Copywriter python=3.11 -y
conda activate LangGraph-Perfilador-Copywriter
pip install -r requirements.txt

# Run API server
uvicorn main:app --reload
# Server: http://localhost:8000, docs: http://localhost:8000/docs

# LangGraph development mode (for graph inspection/debugging)
pip install "langgraph-cli[inmem]"
langgraph dev --allow-blocking

# Docker build & push
docker buildx build -t fedecanesa/scraping-canesa:latest --push .
```

## Architecture

This is a **3-agent sequential Cold Email Generation Pipeline** built with FastAPI + LangGraph.

### Request Flow

```
POST /process  →  create run_id (timestamp)  →  background task
                                                       ↓
                                            LangGraph state machine
                                            (grafo.py: SalesState)
                                                       ↓
                                   DataEngineer → Profiler → Copywriter
```

- **`GET /process/{run_id}`** — polls run status; state is stored in memory via `services/run_manager.py`
- Runs are keyed by `run_id` format `YYYYmmdd_HHMMSS`

### Agent Chain (grafo.py)

1. **DataEngineer** (`agents/data_engineer.py`) — Scrapes target URL with Apify crawler, then optionally cleans/consolidates raw content with OpenAI LLM. Outputs `cleaned_data: list[dict]`.
2. **Profiler** (`agents/profiler.py`) — Analyzes `cleaned_data` with LLM to extract `business_summary`, `pain_points`, `technology`, `opportunities`, `ideal_customer`. Outputs `profile_data: dict`.
3. **Copywriter** (`agents/copywriter.py`) — Uses `profile_data` + user-provided `my_service_info` and `company_tone` to generate a personalized cold email. Outputs `final_email: str`.

### State Shape (SalesState TypedDict)

All inter-agent data flows through this shared state:
- Inputs: `run_id`, `target_url`, `max_crawl_pages`, `max_crawl_depth`, `skip_cleaning`, `my_service_info`, `company_tone`
- Agent 1 → 2: `cleaned_data`
- Agent 2 → 3: `profile_data`
- Agent 3 → API: `final_email`

### Key Files

| File | Role |
|------|------|
| `main.py` | FastAPI app (v2.2.1), CORS config, `/process` endpoints |
| `grafo.py` | LangGraph `StateGraph` definition; exports `app` |
| `schemas/process.py` | Pydantic models: `ProcessRequest`, `ProcessStatusResponse`, `ProcessResult` |
| `services/run_manager.py` | In-memory run store: `create_run`, `get_run`, `update_step`, `complete_run`, `fail_run` |
| `utils/file_storage.py` | Saves agent outputs to `Salidas de los Agentes/` (timestamped JSON + text) |
| `langgraph.json` | Registers `grafo:app` as both `agent` and `business_analysis` graphs |

### External APIs

- **OpenAI** (`OPENAI_API_KEY`) — GPT-4o for cleaning/profiling/copywriting
- **Apify** (`APIFY_API_TOKEN`) — `apify/website-content-crawler` actor for scraping

CORS is configured for `localhost:3000`, `localhost:5173`, and `*.vercel.app`.
