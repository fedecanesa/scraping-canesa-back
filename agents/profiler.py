# agents/profiler.py

"""
Agente 2 - The Profiler (DeepReacher edition)

Analiza el contenido limpio del Agente 1 y devuelve inteligencia comercial
profunda: resumen del negocio, modelo, issues estructurados, oportunidades
accionables y lead score.
"""

import json
import logging
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from utils.file_storage import save_json_output


logger = logging.getLogger(__name__)


class ProfilerState(TypedDict, total=False):
    run_id: str
    cleaned_data: list[dict[str, Any]]
    profile_data: dict[str, Any]
    objective: str
    user_type: str


_OBJECTIVE_LABELS = {
    "sell": "Vender nuestros servicios al prospecto",
    "partnership": "Explorar una alianza o colaboración",
}

_USER_TYPE_LABELS = {
    "marketing_agency": "Agencia de marketing digital",
    "dev_agency": "Agencia de desarrollo web",
    "other": "Profesional / empresa de servicios",
}

_PROMPT = ChatPromptTemplate.from_template(
    """Eres un analista de negocios senior especializado en inteligencia comercial B2B.

Contexto del análisis:
- Tipo de usuario: {user_type_label}
- Objetivo del contacto: {objective_label}

Analiza el contenido extraído del sitio web y devuelve SOLO JSON válido con esta estructura exacta:

{{
  "business_summary": "Resumen ejecutivo del negocio en 2-3 oraciones concretas",
  "what_they_do": "¿A qué se dedica realmente esta empresa? Core del negocio en 1-2 oraciones",
  "business_model": "¿Cómo genera ingresos? Describe brevemente su modelo de negocio",
  "what_doing_well": ["cosa positiva 1", "cosa positiva 2", "cosa positiva 3"],
  "pain_points": ["punto de dolor 1", "punto de dolor 2"],
  "technology": ["tecnología detectada 1", "tecnología detectada 2"],
  "issues": [
    {{
      "title": "Nombre corto del problema detectado",
      "description": "Impacto concreto de este problema en el negocio"
    }}
  ],
  "opportunities": [
    {{
      "title": "Nombre corto de la oportunidad",
      "explanation": "Por qué existe esta oportunidad y por qué es relevante ahora",
      "impact": "Qué pierde o deja de ganar el negocio si no lo resuelve",
      "solution": "Propuesta concreta de solución alineada con el objetivo del analizador"
    }}
  ],
  "ideal_customer": "¿A quién le vende este negocio? Descripción del cliente ideal",
  "lead_score": 72,
  "lead_score_reason": "Una oración explicando el score asignado"
}}

Criterios para lead_score (0-100):
- 80-100: Múltiples oportunidades claras y concretas, negocio activo y en crecimiento, alta alineación con el objetivo del analizador
- 60-79: Algunas oportunidades identificables, negocio establecido, alineación media
- 40-59: Pocas oportunidades claras, negocio básico o muy genérico
- 0-39: Sitio precario, pocas señales de negocio real, muy fuera de target o sin información suficiente

Reglas:
- Usa únicamente la información presente en el contenido extraído
- No inventes datos ni supongas cosas que no estén en el texto
- Genera entre 2 y 4 issues y entre 2 y 4 oportunidades concretas y accionables
- Las oportunidades deben estar orientadas al objetivo del analizador
- Si el sitio tiene poco contenido, ajusta el score hacia abajo y sé conservador
- Devuelve SOLO JSON válido, sin markdown, sin texto adicional

Contenido del sitio:
{website_content}
"""
)


def _get_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=settings.openai_api_key)
    return _PROMPT | llm | StrOutputParser()


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
            "what_they_do": parsed.get("what_they_do", ""),
            "business_model": parsed.get("business_model", ""),
            "what_doing_well": parsed.get("what_doing_well", []),
            "pain_points": parsed.get("pain_points", []),
            "technology": parsed.get("technology", []),
            "issues": parsed.get("issues", []),
            "opportunities": parsed.get("opportunities", []),
            "ideal_customer": parsed.get("ideal_customer", ""),
            "lead_score": int(parsed.get("lead_score", 0)),
            "lead_score_reason": parsed.get("lead_score_reason", ""),
        }
    except Exception:
        logger.exception("[Agente 2] No se pudo parsear el JSON del profiler")
        return {
            "business_summary": "",
            "what_they_do": "",
            "business_model": "",
            "what_doing_well": [],
            "pain_points": [],
            "technology": [],
            "issues": [],
            "opportunities": [],
            "ideal_customer": "",
            "lead_score": 0,
            "lead_score_reason": "",
        }


def profiler_node(state: ProfilerState) -> dict[str, Any]:
    cleaned_data = state.get("cleaned_data", [])
    objective = state.get("objective", "sell")
    user_type = state.get("user_type", "other")

    website_content = _format_cleaned_data_for_prompt(cleaned_data)
    objective_label = _OBJECTIVE_LABELS.get(objective, _OBJECTIVE_LABELS["sell"])
    user_type_label = _USER_TYPE_LABELS.get(user_type, _USER_TYPE_LABELS["other"])

    logger.info("[Agente 2] Profiler ejecutando análisis (objetivo=%s)...", objective)

    raw_result = _get_chain().invoke({
        "website_content": website_content,
        "objective_label": objective_label,
        "user_type_label": user_type_label,
    })

    profile_result = _safe_parse_profile(raw_result)
    logger.info("[Agente 2] Profiler completado — score=%s", profile_result.get("lead_score"))

    run_id = state.get("run_id")
    if run_id:
        out_path = save_json_output(run_id, "agente2_perfil.json", profile_result)
        logger.info("[Agente 2] Guardado: %s", out_path)

    return {"profile_data": profile_result}
