# agents/profiler.py

"""
Agente 2 - The Profiler

Analiza el contenido limpio del Agente 1 y devuelve
un perfil estructurado en JSON para que el Copywriter
no dependa de regex frágiles.
"""

import json
import logging
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.file_storage import save_json_output


logger = logging.getLogger(__name__)


class ProfilerState(TypedDict, total=False):
    run_id: str
    cleaned_data: list[dict[str, Any]]
    profile_data: dict[str, Any]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

prompt_profiler = ChatPromptTemplate.from_template(
    """Analiza el siguiente contenido extraído de una website y devuelve JSON válido con esta estructura exacta:

{{
  "business_summary": "Resumen breve del negocio",
  "pain_points": ["punto 1", "punto 2"],
  "technology": ["tech 1", "tech 2"],
  "opportunities": ["oportunidad 1", "oportunidad 2"],
  "ideal_customer": "descripción del cliente ideal"
}}

Reglas:
- Usa únicamente la información presente en el contenido
- No inventes datos
- Si algo no surge con claridad, devuelve una lista vacía o un string vacío
- Devuelve SOLO JSON válido, sin markdown, sin explicación adicional

Contenido del sitio:
{website_content}
"""
)

profiler_chain = prompt_profiler | llm | StrOutputParser()


def _format_cleaned_data_for_prompt(cleaned_data: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for item in cleaned_data:
        url = item.get("url", "(sin URL)")
        text = item.get("cleaned_text", "")

        if text:
            parts.append(f"### Página: {url}\n{text}")

    return "\n\n---\n\n".join(parts) if parts else "(No hay contenido disponible)"


def _safe_parse_profile(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)

        return {
            "business_summary": parsed.get("business_summary", ""),
            "pain_points": parsed.get("pain_points", []),
            "technology": parsed.get("technology", []),
            "opportunities": parsed.get("opportunities", []),
            "ideal_customer": parsed.get("ideal_customer", ""),
        }
    except Exception:
        logger.exception("[Agente 2] No se pudo parsear el JSON del profiler")
        return {
            "business_summary": "",
            "pain_points": [],
            "technology": [],
            "opportunities": [],
            "ideal_customer": "",
        }


def profiler_node(state: ProfilerState) -> dict[str, Any]:
    cleaned_data = state.get("cleaned_data", [])
    website_content = _format_cleaned_data_for_prompt(cleaned_data)

    logger.info("[Agente 2] Profiler ejecutando análisis...")
    raw_result = profiler_chain.invoke({"website_content": website_content})
    profile_result = _safe_parse_profile(raw_result)
    logger.info("[Agente 2] Profiler completado")

    run_id = state.get("run_id")
    if run_id:
        out_path = save_json_output(run_id, "agente2_perfil.json", profile_result)
        logger.info("[Agente 2] Guardado: %s", out_path)

    return {"profile_data": profile_result}