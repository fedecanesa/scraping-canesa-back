# main.py

from datetime import datetime
import logging

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from grafo import app as pipeline
from schemas.process import (
    ProcessRequest,
    ProcessStartResponse,
    ProcessStatusResponse,
)
from services.run_manager import (
    complete_run,
    create_run,
    fail_run,
    get_run,
    update_step,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cold Email Scraper API",
    description="Pipeline: Scrape → Profile → Copywriter",
    version="2.2.1",
)

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


def run_pipeline(run_id: str, inputs: dict) -> None:
    try:
        logger.info("[%s] Pipeline iniciado", run_id)

        update_step(run_id, "DataEngineer", "running")
        result = pipeline.invoke(inputs)
        update_step(run_id, "DataEngineer", "completed")

        # Como LangGraph resuelve toda la cadena internamente,
        # acá dejamos los pasos siguientes como completados al final.
        update_step(run_id, "Profiler", "completed")
        update_step(run_id, "Copywriter", "completed")

        complete_run(run_id, result)
        logger.info("[%s] Pipeline completado", run_id)

    except Exception as e:
        logger.exception("[%s] Error ejecutando pipeline", run_id)
        fail_run(run_id, str(e))


@app.post("/process", response_model=ProcessStartResponse)
def process_url(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> ProcessStartResponse:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    inputs = {
        "run_id": run_id,
        "target_url": str(request.target_url),
        "max_crawl_pages": request.max_crawl_pages,
        "max_crawl_depth": request.max_crawl_depth,
        "skip_cleaning": request.skip_cleaning,
        "my_service_info": request.my_service_info or "Soluciones de IA para empresas",
        "company_tone": request.company_tone or "profesional y cercano",
    }

    create_run(run_id, str(request.target_url))
    background_tasks.add_task(run_pipeline, run_id, inputs)

    return ProcessStartResponse(run_id=run_id, status="started")


@app.get("/process/{run_id}", response_model=ProcessStatusResponse)
def get_process_status(run_id: str) -> ProcessStatusResponse:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return ProcessStatusResponse(**run)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}