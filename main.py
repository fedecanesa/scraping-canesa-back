# main.py

# config.py usa pydantic-settings y lee .env directamente.
# load_dotenv() se mantiene como fallback para compatibilidad con os.environ.
from dotenv import load_dotenv
load_dotenv()

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings
from grafo import app as pipeline
from schemas.process import (
    ProcessRequest,
    ProcessStartResponse,
    ProcessStatusResponse,
)
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
    title="Cold Email Scraper API",
    description="Pipeline: Scrape → Profile → Copywriter",
    version="2.4.0",
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
    """
    Valida el header X-Api-Key si API_KEY está configurado en el entorno.
    Si API_KEY está vacío, el endpoint es público (útil para desarrollo).
    """
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# --- Pipeline execution ---
def _execute_pipeline_sync(run_id: str, inputs: dict) -> dict:
    """
    Ejecuta el pipeline LangGraph de forma síncrona.
    Corre en un thread separado para poder aplicar timeout.
    """
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
    """
    Background task: ejecuta el pipeline con timeout configurable.

    Usa un ThreadPoolExecutor interno para poder interrumpir la espera
    si el pipeline supera pipeline_timeout_seconds. El thread subyacente
    puede continuar corriendo (no cancelable en Python), pero el run
    queda marcado como 'failed' para el usuario.
    """
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
    """Inicia el pipeline en background y devuelve el run_id para polling."""
    run_id = uuid.uuid4().hex[:12]

    inputs = {
        "run_id": run_id,
        "target_url": str(body.target_url),
        "max_crawl_pages": body.max_crawl_pages,
        "max_crawl_depth": body.max_crawl_depth,
        "skip_cleaning": body.skip_cleaning,
        "my_service_info": body.my_service_info or "Soluciones de IA para empresas",
        "company_tone": body.company_tone or "profesional y cercano",
    }

    create_run(run_id, str(body.target_url))

    # BackgroundTasks de FastAPI corre en el thread pool de uvicorn.
    # Para alta concurrencia, reemplazar por Celery/ARQ + Redis.
    background_tasks.add_task(run_pipeline, run_id, inputs)

    return ProcessStartResponse(run_id=run_id, status="started")


@app.get("/process/{run_id}", response_model=ProcessStatusResponse)
async def get_process_status(run_id: str) -> ProcessStatusResponse:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return ProcessStatusResponse(**run)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
