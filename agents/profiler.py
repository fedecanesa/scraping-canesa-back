# agents/profiler.py

"""
Agente 2 - The Profiler (DeepReacher edition)

Dos modos completamente diferenciados:
- sell: detecta brechas, urgencias y señales de compra
- partnership: detecta complementariedad, reputación y potencial de alianza
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
    reviews_data: list[dict[str, Any]]
    jobs_data: dict[str, Any]
    business_name: str
    profile_data: dict[str, Any]
    objective: str
    user_type: str


_USER_TYPE_LABELS = {
    "marketing_agency": "Agencia de marketing digital",
    "dev_agency": "Agencia de desarrollo web",
    "other": "Profesional / empresa de servicios",
}

# ─── Prompt SELL ──────────────────────────────────────────────────────────────
# Foco: brechas, urgencias, señales de compra, potencial de ROI

_PROMPT_SELL = ChatPromptTemplate.from_template(
    """Eres un analista de inteligencia comercial B2B especializado en identificar oportunidades de venta.

Contexto del analizador:
- Tipo de negocio: {user_type_label}
- Objetivo: Vender servicios a este prospecto

Tu misión: analizar TODAS las fuentes disponibles con ojos de vendedor experto. Detectá brechas reales, señales de urgencia y oportunidades concretas donde nuestros servicios pueden generar valor inmediato.

FUENTES DE ANÁLISIS:

1. CONTENIDO DEL SITIO WEB:
{website_content}

2. RESEÑAS DE GOOGLE ({reviews_count} reseñas):
{reviews_context}

3. SEÑALES DE CONTRATACIÓN (job postings detectados):
{jobs_context}

---

Devuelve SOLO JSON válido con esta estructura exacta:

{{
  "business_summary": "Resumen ejecutivo del negocio en 2-3 oraciones. Quién son, a qué se dedican, a quién le venden.",
  "what_they_do": "Core del negocio en 1 oración muy concreta. Sin rodeos.",
  "business_model": "Cómo generan dinero. Servicios, productos, suscripciones, etc.",
  "what_doing_well": ["fortaleza real 1", "fortaleza real 2"],
  "pain_points": ["dolor específico 1", "dolor específico 2", "dolor específico 3"],
  "technology": ["tech detectada 1", "tech detectada 2"],
  "issues": [
    {{
      "title": "Nombre corto del problema (max 6 palabras)",
      "description": "Impacto concreto en su negocio. No genérico. Qué están perdiendo HOY por esto."
    }}
  ],
  "opportunities": [
    {{
      "title": "Oportunidad concreta (max 6 palabras)",
      "explanation": "Por qué existe esta brecha ahora. Señales específicas que la evidencian (podés citar reseñas o job postings).",
      "impact": "Cuánto les cuesta no resolverlo: clientes perdidos, tiempo, dinero, reputación.",
      "solution": "Cómo nuestro tipo de servicio lo resuelve. Concreto, no genérico."
    }}
  ],
  "ideal_customer": "A quién le vende esta empresa. Su buyer persona.",
  "buying_signals": ["señal de compra 1 (puede venir de reseñas o job postings)", "señal de compra 2"],
  "top_review_quote": "La reseña más reveladora (textual, positiva o negativa) para usar en outreach. Vacío si no hay reseñas.",
  "lead_score": 72,
  "lead_score_reason": "Una oración justificando el score considerando las 3 fuentes: web, reseñas y señales de contratación."
}}

Criterios del lead_score para VENTA (0-100):
- 80-100: Dolor urgente y evidente en múltiples fuentes, señales de crecimiento o presión, brecha grande, alta alineación con nuestro tipo de servicio
- 60-79: Oportunidades identificables pero menos urgentes, negocio activo, alineación media
- 40-59: Pocas señales claras, negocio genérico o sin brechas evidentes
- 0-39: Sin información suficiente, sitio precario, o empresa claramente fuera de target

Señales de compra a buscar:
- Web: sitio desactualizado, sin blog, sin pixel de tracking, sin chat, formularios rotos, sin testimonios, stack tecnológico viejo
- Reseñas: quejas recurrentes en reviews revelan problemas no resueltos = oportunidad de venta directa
- Job postings: buscan "Head of Marketing" = no tienen marketing in-house; buscan "CTO" = en transformación digital; etc.

Reglas:
- Usá SOLO información presente en las fuentes. No inventes.
- Si hay reseñas, incluí insights de ellas en issues y oportunidades.
- Si hay job postings, úsalos como señal de compra prioritaria.
- Generá entre 2 y 4 issues y entre 2 y 4 oportunidades.
- Sé específico. "No tienen automatización" no sirve. "No tienen flujo automatizado post-compra, lo que les genera abandono de clientes en los primeros 30 días" sí sirve.
- Devuelve SOLO JSON válido, sin markdown, sin texto adicional.
"""
)

# ─── Prompt PARTNERSHIP ───────────────────────────────────────────────────────
# Foco: complementariedad, reputación, fit para colaborar, modelo de alianza

_PROMPT_PARTNERSHIP = ChatPromptTemplate.from_template(
    """Eres un analista de desarrollo de negocios especializado en identificar oportunidades de alianzas estratégicas B2B.

Contexto del analizador:
- Tipo de negocio del analizador: {user_type_label}
- Objetivo: Explorar si esta empresa es un buen candidato para una alianza, joint venture, referidos mutuos o colaboración en proyectos

Tu misión: analizar TODAS las fuentes disponibles con ojos de potencial socio de negocios. No buscás venderles algo — buscás entender si tienen lo que a vos te falta y si vos tenés lo que a ellos les falta. La pregunta clave es: ¿nos complementamos?

FUENTES DE ANÁLISIS:

1. CONTENIDO DEL SITIO WEB:
{website_content}

2. RESEÑAS DE GOOGLE ({reviews_count} reseñas):
{reviews_context}

3. SEÑALES DE CONTRATACIÓN (job postings detectados):
{jobs_context}

---

Devuelve SOLO JSON válido con esta estructura exacta:

{{
  "business_summary": "Resumen ejecutivo: qué hacen, a quién le sirven, en qué mercado operan.",
  "what_they_do": "Su especialidad principal en 1 oración. El núcleo de su propuesta de valor.",
  "business_model": "Cómo generan ingresos y a qué tipo de clientes atienden.",
  "what_doing_well": ["fortaleza 1 (relevante para una alianza)", "fortaleza 2", "fortaleza 3"],
  "pain_points": ["brecha o limitación que un socio podría cubrir 1", "brecha 2"],
  "technology": ["tech detectada 1", "tech detectada 2"],
  "issues": [
    {{
      "title": "Limitación o brecha detectada (max 6 palabras)",
      "description": "Qué les falta o qué no pueden hacer solos, que un socio podría complementar."
    }}
  ],
  "opportunities": [
    {{
      "title": "Tipo de alianza posible (max 6 palabras)",
      "explanation": "Por qué esta alianza tiene sentido. Qué se complementa entre ambas empresas (podés usar señales de reseñas o job postings).",
      "impact": "Qué gana cada parte. Clientes compartidos, capacidad adicional, mercados nuevos.",
      "solution": "Formato concreto de la alianza: referidos, white-label, co-ejecución de proyectos, joint venture, etc."
    }}
  ],
  "ideal_customer": "A quién le vende esta empresa. Esto ayuda a ver si hay solapamiento o complemento de audiencias.",
  "buying_signals": ["señal de apertura a colaborar 1 (puede venir de reseñas o job postings)", "señal 2"],
  "top_review_quote": "La reseña más reveladora para entender su relación con clientes. Vacío si no hay reseñas.",
  "lead_score": 72,
  "lead_score_reason": "Una oración: qué tan complementarios son considerando las 3 fuentes y qué tan probable es que una alianza funcione."
}}

Criterios del lead_score para PARTNERSHIP (0-100):
- 80-100: Servicios altamente complementarios (no compiten, se complementan), cliente ideal similar o adyacente, tamaño y reputación compatibles, señales de apertura a colaborar
- 60-79: Complementariedad parcial, algo de solapamiento de clientes, posible pero requiere más conversación
- 40-59: Complementariedad débil o competencia directa, fit dudoso
- 0-39: Competidores directos, sin fit, o información insuficiente para evaluar

Tipos de alianzas a considerar:
- Referidos mutuos: se mandan clientes que el otro puede atender mejor
- White-label: uno ejecuta bajo la marca del otro
- Co-ejecución: proyectos grandes que requieren ambas especialidades
- Joint venture: oferta conjunta al mercado
- Subcontratación: uno es proveedor del otro en áreas específicas

Señales de apertura a la alianza:
- Reseñas: clientes que mencionan servicios que la empresa no ofrece → oportunidad de complemento
- Job postings: buscan roles en áreas donde un socio ya tiene capacidad → alianza vs contratar

Reglas:
- Evaluá complementariedad, NO oportunidades de venta. Cambio de mentalidad total.
- Si hay reseñas, usalas para entender la reputación y las brechas de servicio.
- Si hay job postings, úsalos para identificar áreas donde un socio puede complementar en vez de que contraten.
- Generá entre 2 y 3 issues y entre 2 y 3 oportunidades de alianza concretas.
- Sé específico sobre el formato de alianza. No digas "podrían colaborar" — decí exactamente cómo.
- Devuelve SOLO JSON válido, sin markdown, sin texto adicional.
"""
)


def _get_chain(objective: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=settings.openai_api_key)
    prompt = _PROMPT_SELL if objective == "sell" else _PROMPT_PARTNERSHIP
    return prompt | llm | StrOutputParser()


def _format_cleaned_data_for_prompt(cleaned_data: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in cleaned_data:
        url = item.get("url", "(sin URL)")
        text = item.get("cleaned_text", "")
        if text:
            parts.append(f"### Página: {url}\n{text}")
    return "\n\n---\n\n".join(parts) if parts else "(No hay contenido disponible)"


def _format_reviews_context(reviews_data: list[dict[str, Any]]) -> tuple[str, int]:
    """Formatea las reviews para el prompt. Devuelve (texto, cantidad)."""
    if not reviews_data:
        return "No hay reseñas disponibles.", 0
    lines: list[str] = []
    for r in reviews_data:
        rating = r.get("rating")
        text = r.get("text", "")
        stars = f"{'★' * int(rating)}{'☆' * (5 - int(rating))}" if rating else ""
        lines.append(f"- {stars} \"{text}\"")
    return "\n".join(lines), len(reviews_data)


def _format_jobs_context(jobs_data: dict[str, Any]) -> str:
    """Formatea las señales de contratación para el prompt."""
    if not jobs_data:
        return "No se detectaron señales de contratación."
    jobs = jobs_data.get("jobs", [])
    hiring_summary = jobs_data.get("hiring_summary", "")
    if not jobs and not hiring_summary:
        return "No se detectaron señales de contratación en el sitio."
    lines: list[str] = []
    if hiring_summary:
        lines.append(f"Resumen: {hiring_summary}")
    for job in jobs:
        role = job.get("role", "")
        dept = job.get("department", "")
        signal = job.get("signal", "")
        lines.append(f"- Puesto: {role} ({dept}) → {signal}")
    return "\n".join(lines)


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
            "buying_signals": parsed.get("buying_signals", []),
            "top_review_quote": parsed.get("top_review_quote", ""),
            "lead_score": int(parsed.get("lead_score", 0)),
            "lead_score_reason": parsed.get("lead_score_reason", ""),
        }
    except Exception:
        logger.exception("[Agente 2] No se pudo parsear el JSON del profiler")
        return {
            "business_summary": "", "what_they_do": "", "business_model": "",
            "what_doing_well": [], "pain_points": [], "technology": [],
            "issues": [], "opportunities": [], "ideal_customer": "",
            "buying_signals": [], "top_review_quote": "",
            "lead_score": 0, "lead_score_reason": "",
        }


def profiler_node(state: ProfilerState) -> dict[str, Any]:
    cleaned_data = state.get("cleaned_data", [])
    reviews_data = state.get("reviews_data", [])
    jobs_data = state.get("jobs_data", {})
    objective = state.get("objective", "sell")
    user_type = state.get("user_type", "other")

    website_content = _format_cleaned_data_for_prompt(cleaned_data)
    user_type_label = _USER_TYPE_LABELS.get(user_type, _USER_TYPE_LABELS["other"])
    reviews_context, reviews_count = _format_reviews_context(reviews_data)
    jobs_context = _format_jobs_context(jobs_data)

    logger.info(
        "[Agente 2] Profiler ejecutando análisis (objetivo=%s, reviews=%s, jobs=%s)...",
        objective, reviews_count, len((jobs_data or {}).get("jobs", [])),
    )

    raw_result = _get_chain(objective).invoke({
        "website_content": website_content,
        "user_type_label": user_type_label,
        "reviews_context": reviews_context,
        "reviews_count": reviews_count,
        "jobs_context": jobs_context,
    })

    profile_result = _safe_parse_profile(raw_result)
    logger.info("[Agente 2] Profiler completado — score=%s", profile_result.get("lead_score"))

    run_id = state.get("run_id")
    if run_id:
        out_path = save_json_output(run_id, "agente2_perfil.json", profile_result)
        logger.info("[Agente 2] Guardado: %s", out_path)

    return {"profile_data": profile_result}
