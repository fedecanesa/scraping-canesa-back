# main.py

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from config import settings
from grafo import app as pipeline
from schemas.process import (
    ProcessRequest,
    ProcessStartResponse,
    ProcessStatusResponse,
)
from schemas.search import SearchRequest, SearchResponse
from services.places_search import search_places
from services.run_manager import (
    PIPELINE_STEPS,
    complete_run,
    create_run,
    fail_run,
    get_run,
    update_step,
)


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address)


# --- App ---
app = FastAPI(
    title="DeepReacher API",
    description="Pipeline de inteligencia comercial: Scrape → Análisis profundo → Mensajes personalizados",
    version="3.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://v0-web-scraper-interface-gamma.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth dependency ---
def verify_api_key(x_api_key: str = Header(default="")) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Pipeline execution ---
def _execute_pipeline_sync(run_id: str, inputs: dict) -> dict:
    update_step(run_id, PIPELINE_STEPS[0], "running")
    final_state: dict = {}

    for event in pipeline.stream(inputs):
        for node_name, node_output in event.items():
            update_step(run_id, node_name, "completed")
            final_state.update(node_output)

            if node_name in PIPELINE_STEPS:
                idx = PIPELINE_STEPS.index(node_name)
                if idx + 1 < len(PIPELINE_STEPS):
                    update_step(run_id, PIPELINE_STEPS[idx + 1], "running")

            logger.info("[%s] Nodo completado: %s", run_id, node_name)

    return final_state


def run_pipeline(run_id: str, inputs: dict) -> None:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"pl_{run_id}")
    try:
        logger.info("[%s] Pipeline iniciado (timeout=%ss)", run_id, settings.pipeline_timeout_seconds)
        future = executor.submit(_execute_pipeline_sync, run_id, inputs)

        try:
            final_state = future.result(timeout=settings.pipeline_timeout_seconds)
            complete_run(run_id, final_state)
            logger.info("[%s] Pipeline completado", run_id)

        except FuturesTimeoutError:
            msg = f"Timeout: el pipeline excedió {settings.pipeline_timeout_seconds}s"
            logger.error("[%s] %s", run_id, msg)
            fail_run(run_id, msg)

    except Exception as e:
        logger.exception("[%s] Error ejecutando pipeline", run_id)
        fail_run(run_id, str(e))
    finally:
        executor.shutdown(wait=False)


# --- Endpoints ---
@app.post(
    "/process",
    response_model=ProcessStartResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def process_url(
    request: Request,
    body: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> ProcessStartResponse:
    """Inicia el pipeline de análisis en background y devuelve el run_id."""
    run_id = uuid.uuid4().hex[:12]

    inputs = {
        "run_id": run_id,
        "target_url": str(body.target_url),
        "max_crawl_pages": body.max_crawl_pages,
        "max_crawl_depth": body.max_crawl_depth,
        "skip_cleaning": body.skip_cleaning,
        "my_service_info": body.my_service_info or "Soluciones de IA para empresas",
        "company_tone": body.company_tone or "profesional y cercano",
        "objective": body.objective or "sell",
        "user_type": body.user_type or "other",
    }

    create_run(run_id, str(body.target_url))
    background_tasks.add_task(run_pipeline, run_id, inputs)

    return ProcessStartResponse(run_id=run_id, status="started")


@app.get("/process/{run_id}", response_model=ProcessStatusResponse)
async def get_process_status(run_id: str) -> ProcessStatusResponse:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return ProcessStatusResponse(**run)


@app.get("/process/{run_id}/stream")
async def stream_run_status(run_id: str, request: Request) -> EventSourceResponse:
    """Pushes run state as SSE events every 500ms until completed or failed."""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                logger.info("[%s] SSE client disconnected", run_id)
                break

            run = get_run(run_id)
            if run is None:
                yield {"data": json.dumps({"error": "Run not found", "status": "failed"})}
                break

            yield {"data": json.dumps(jsonable_encoder(run))}

            if run["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("20/minute")
async def search_businesses(
    request: Request,
    body: SearchRequest,
) -> SearchResponse:
    """Busca empresas por categoría + ciudad usando Google Places API."""
    if not settings.google_maps_api_key:
        raise HTTPException(status_code=503, detail="Google Maps API no configurada")

    try:
        results = search_places(
            category=body.category,
            city=body.city,
            max_results=body.max_results,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    query = f"{body.category} en {body.city}"
    return SearchResponse(results=results, total=len(results), query=query)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "DeepReacher API"}
