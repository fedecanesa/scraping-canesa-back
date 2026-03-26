"""
API FastAPI para el pipeline de Cold Email.
Recibe URL desde el frontend y ejecuta el grafo de agentes.
"""

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Carga .env antes de importar el grafo/agentes

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from grafo import app as pipeline


app = FastAPI(
    title="Cold Email Scraper API",
    description="Pipeline: Scrape → Profile → Copywriter",
    version="1.0.0",
)

# CORS: Frontend en Vercel + localhost para desarrollo
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


class ProcessRequest(BaseModel):
    target_url: str = Field(..., description="URL del sitio web a analizar")
    max_crawl_pages: int = Field(10, ge=1, le=50, description="Máximo de páginas a rastrear")
    max_crawl_depth: int = Field(3, ge=1, le=10, description="Profundidad del crawl (1-10). Default: 3")
    skip_cleaning: bool = Field(True, description="Si True, salta limpieza LLM (más rápido). Si False, limpia con OpenAI")
    my_service_info: str | None = Field(
        None,
        description="Descripción de tu servicio (para el email). Default: 'Soluciones de IA para empresas'",
    )
    company_tone: str | None = Field(
        None,
        description="Tono deseado (ej: 'profesional y cercano'). Default: 'profesional y cercano'",
    )


class ProcessResponse(BaseModel):
    final_email: str
    profile_data: str | None = None
    target_url: str
    run_id: str | None = None


@app.post("/process", response_model=ProcessResponse)
def process_url(request: ProcessRequest):
    """
    Recibe la URL del frontend y ejecuta el pipeline completo:
    DataEngineer (scrape + limpieza) → Profiler (análisis) → Copywriter (cold email).
    """
    try:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        inputs = {
            "run_id": run_id,
            "target_url": request.target_url,
            "max_crawl_pages": request.max_crawl_pages,
            "max_crawl_depth": request.max_crawl_depth,
            "skip_cleaning": request.skip_cleaning,
        }
        if request.my_service_info:
            inputs["my_service_info"] = request.my_service_info
        if request.company_tone:
            inputs["company_tone"] = request.company_tone

        result = pipeline.invoke(inputs)

        return ProcessResponse(
            final_email=result.get("final_email", ""),
            profile_data=result.get("profile_data"),
            target_url=result.get("target_url", request.target_url),
            run_id=run_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el pipeline: {str(e)}")


@app.get("/health")
def health():
    """Health check para el frontend."""
    return {"status": "ok"}
