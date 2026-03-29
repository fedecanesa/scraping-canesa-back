# agents/copywriter.py

"""
Agente 3 - The Copywriter (DeepReacher edition)

Genera 3 variantes de mensaje de contacto B2B hiper personalizadas
a partir del perfil estructurado del Agente 2.
"""

import json
import logging
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from utils.file_storage import save_json_output, save_text_output


logger = logging.getLogger(__name__)


class CopywriterState(TypedDict, total=False):
    run_id: str
    profile_data: dict[str, Any]
    my_service_info: str
    company_tone: str
    objective: str
    final_email: str
    message_variants: list[dict[str, str]]


_OBJECTIVE_LABELS = {
    "sell": "Vender nuestros servicios y generar una reunión o respuesta",
    "partnership": "Proponer una alianza o colaboración de negocios",
}

_PROMPT = ChatPromptTemplate.from_template(
    """Eres un experto Copywriter B2B especializado en cold outreach con alta tasa de respuesta.

PERFIL DEL PROSPECTO:
Negocio: {what_they_do}
Resumen: {business_summary}
Problemas detectados:
{issues_list}
Oportunidades detectadas:
{opportunities_list}
Tecnología: {technology}

NOSOTROS OFRECEMOS: {my_service_info}
OBJETIVO DEL MENSAJE: {objective_label}
TONO: {company_tone}

Genera 3 variantes de mensaje de contacto. Devuelve SOLO JSON válido, sin markdown ni texto adicional:

{{
  "main": "Mensaje principal aquí",
  "variant_a": "Variante A aquí",
  "variant_b": "Variante B aquí"
}}

Descripción de cada variante:
- main: Equilibrado y personalizado. Menciona algo específico del negocio del prospecto, conecta con una oportunidad o problema detectado, propone valor claro y termina con CTA de bajo compromiso (ej: "¿Tiene 15 minutos esta semana?").
- variant_a: Más directo y orientado a resultados. Va al punto rápido, menciona un beneficio concreto o potencial impacto específico, CTA proactivo (propone fecha concreta).
- variant_b: Más consultivo. Empieza con una pregunta de diagnóstico inteligente que posiciona al remitente como experto, termina con CTA de consulta gratuita o conversación sin compromiso.

Reglas para TODOS los mensajes:
- Máximo 120 palabras por mensaje
- Icebreaker genuino y específico del prospecto (jamás genérico como "Espero que estés bien")
- Referencia explícita a una oportunidad o problema real detectado
- Propuesta de valor alineada al objetivo del mensaje
- Tono: {company_tone}
- NO empieces con "Hola" genérico ni frases de spam
- Devuelve SOLO JSON, sin markdown, sin explicaciones
"""
)


def _get_chain():
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=settings.openai_api_key)
    return _PROMPT | llm | StrOutputParser()


def _as_bullets(items: list[Any], key: str | None = None) -> str:
    if not items:
        return "- No detectado"
    if key:
        return "\n".join([f"- {item.get(key, item)}" for item in items])
    return "\n".join([f"- {item}" for item in items])


def _parse_variants(raw: str) -> list[dict[str, str]]:
    try:
        parsed = json.loads(raw)
        return [
            {"id": "main", "label": "Principal", "content": parsed.get("main", "").strip()},
            {"id": "variant_a", "label": "Variante A", "content": parsed.get("variant_a", "").strip()},
            {"id": "variant_b", "label": "Variante B", "content": parsed.get("variant_b", "").strip()},
        ]
    except Exception:
        logger.exception("[Agente 3] No se pudo parsear variantes JSON, usando fallback")
        return [{"id": "main", "label": "Principal", "content": raw.strip()}]


def copywriter_node(state: CopywriterState) -> dict[str, Any]:
    profile_data = state.get("profile_data", {})
    my_service_info = state.get("my_service_info", "Soluciones de IA para empresas")
    company_tone = state.get("company_tone", "profesional y cercano")
    objective = state.get("objective", "sell")

    objective_label = _OBJECTIVE_LABELS.get(objective, _OBJECTIVE_LABELS["sell"])

    opportunities = profile_data.get("opportunities", [])
    issues = profile_data.get("issues", [])

    logger.info("[Agente 3] Copywriter generando 3 variantes (objetivo=%s)...", objective)

    raw_result = _get_chain().invoke({
        "what_they_do": profile_data.get("what_they_do", profile_data.get("business_summary", "")),
        "business_summary": profile_data.get("business_summary", ""),
        "issues_list": _as_bullets(issues, key="title"),
        "opportunities_list": _as_bullets(opportunities, key="title"),
        "technology": ", ".join(profile_data.get("technology", [])) or "No detectada",
        "my_service_info": my_service_info,
        "objective_label": objective_label,
        "company_tone": company_tone,
    })

    message_variants = _parse_variants(raw_result)
    final_email = message_variants[0]["content"] if message_variants else raw_result.strip()

    logger.info("[Agente 3] Copywriter completado — %s variantes generadas", len(message_variants))

    run_id = state.get("run_id")
    if run_id:
        out_path = save_json_output(run_id, "agente3_variantes.json", message_variants)
        logger.info("[Agente 3] Guardado: %s", out_path)
        save_text_output(run_id, "agente3_email.txt", final_email)

    return {
        "final_email": final_email,
        "message_variants": message_variants,
    }
