# agents/jobs_agent.py

"""
JobsAgent - Detecta señales de contratación en el sitio web del prospecto.

Busca páginas de empleo/careers del propio sitio del prospecto para identificar
qué áreas están creciendo. Un "Head of Marketing" buscado = no tienen marketing
in-house = pitch perfecto para una agencia de marketing.
"""

import json
import logging
from typing import Any
from urllib.parse import urlparse

from apify_client import ApifyClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings


logger = logging.getLogger(__name__)

APIFY_ACTOR_ID = "apify/website-content-crawler"

CAREERS_PATHS = [
    "/careers",
    "/jobs",
    "/empleo",
    "/trabaja-con-nosotros",
    "/work-with-us",
    "/join-us",
    "/vacantes",
    "/about/careers",
    "/company/jobs",
]


def _build_careers_urls(target_url: str) -> list[dict[str, str]]:
    """Genera posibles URLs de páginas de empleo del prospecto."""
    parsed = urlparse(target_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return [{"url": f"{base}{path}"} for path in CAREERS_PATHS]


def _scrape_careers_pages(target_url: str) -> str:
    """Scrapea posibles páginas de empleo del prospecto. Devuelve el texto encontrado."""
    client = ApifyClient(settings.apify_api_token)

    run_input: dict[str, Any] = {
        "startUrls": _build_careers_urls(target_url),
        "maxCrawlPages": 5,
        "maxCrawlDepth": 1,
    }

    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
    dataset = client.dataset(run["defaultDatasetId"])

    texts: list[str] = []
    for item in dataset.iterate_items():
        text = item.get("markdown") or item.get("text") or ""
        if text.strip():
            texts.append(f"[{item.get('url', '')}]\n{text[:3000]}")

    return "\n\n---\n\n".join(texts)


def _extract_job_signals(raw_text: str) -> dict[str, Any]:
    """Usa un LLM para extraer señales de contratación del texto scrapeado."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=settings.openai_api_key)

    system = """Eres un analista de inteligencia comercial. Analizás páginas de empleo de empresas y extraés señales de negocio accionables.

Devuelve SOLO JSON válido con esta estructura:
{
  "jobs": [
    {
      "role": "Nombre del puesto",
      "department": "Área: Marketing, Tech, Ventas, Operaciones, Finanzas, RRHH, etc.",
      "signal": "Qué revela este puesto sobre el negocio en 1 oración concreta"
    }
  ],
  "hiring_summary": "Resumen de la situación de contratación. Qué áreas están creciendo y qué señala sobre sus prioridades actuales. 1-2 oraciones."
}

Si no encontrás información de empleo, devolvé: {"jobs": [], "hiring_summary": ""}

Ejemplos de señales valiosas:
- "Head of Marketing" → No tienen marketing in-house, lo están armando desde cero
- "CTO" o "Tech Lead" → Transformación digital en curso, necesitan infraestructura
- "Sales Manager" → Están escalando ventas, probablemente necesiten herramientas de CRM o automatización
- "Customer Success Manager" → Crecimiento de clientes, posible problema de churn
- "Data Analyst" → Maduración del negocio, necesitan inteligencia de datos"""

    user_msg = f"Analizá esta página de empleo y extraé señales de negocio:\n\n{raw_text[:6000]}"
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])

    try:
        parsed = json.loads(str(response.content).strip())
        return {
            "jobs": parsed.get("jobs", []),
            "hiring_summary": parsed.get("hiring_summary", ""),
        }
    except Exception:
        logger.exception("[JobsAgent] No se pudo parsear el JSON de señales")
        return {"jobs": [], "hiring_summary": ""}


def jobs_node(state: dict[str, Any]) -> dict[str, Any]:
    target_url = state.get("target_url", "")

    logger.info("[JobsAgent] Buscando señales de contratación en: %s", target_url)

    jobs_data: dict[str, Any] = {"jobs": [], "hiring_summary": ""}
    try:
        if not settings.apify_api_token:
            logger.warning("[JobsAgent] APIFY_API_TOKEN no configurado, saltando")
        elif not target_url:
            logger.warning("[JobsAgent] Sin target_url, saltando")
        else:
            raw_text = _scrape_careers_pages(target_url)
            if raw_text.strip():
                jobs_data = _extract_job_signals(raw_text)
                logger.info("[JobsAgent] %s puestos detectados", len(jobs_data.get("jobs", [])))
            else:
                logger.info("[JobsAgent] No se encontraron páginas de empleo accesibles")
    except Exception:
        logger.exception("[JobsAgent] Error al buscar empleos, continuando sin datos")

    return {"jobs_data": jobs_data}
