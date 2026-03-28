# agents/copywriter.py

"""
Agente 3 - The Copywriter

Toma el perfil estructurado del Agente 2 y redacta un cold email
más robusto, sin depender de regex ni texto libre ambiguo.
"""

import logging
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from utils.file_storage import save_text_output


logger = logging.getLogger(__name__)


class CopywriterState(TypedDict, total=False):
    run_id: str
    profile_data: dict[str, Any]
    my_service_info: str
    company_tone: str
    final_email: str


_PROMPT = ChatPromptTemplate.from_template(
    """Eres un experto Copywriter de ventas B2B especializado en cold emailing.

Tienes esta información del prospecto:

Resumen del negocio:
{business_summary}

Puntos de dolor:
{pain_points}

Tecnología detectada:
{technology}

Oportunidades:
{opportunities}

Cliente ideal:
{ideal_customer}

Nosotros vendemos:
{my_service_info}

Escribe un email al CEO o responsable comercial. Reglas:
- Empieza con un icebreaker genuino sobre algo específico del negocio
- Conecta con uno o más puntos de dolor detectados
- Presenta nuestra solución como una ayuda natural, no agresiva
- Usa un CTA de bajo compromiso
- Mantén un tono {company_tone}
- Sé concreto, humano y breve
- Devuelve ÚNICAMENTE el email, sin introducciones ni explicaciones
"""
)


def _get_chain():
    """Instancia el LLM en tiempo de ejecución, leyendo la key desde settings."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=settings.openai_api_key)
    return _PROMPT | llm | StrOutputParser()


def _as_bullets(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join([f"- {item}" for item in value]) if value else "- No detectado"
    return str(value) if value else "No detectado"


def copywriter_node(state: CopywriterState) -> dict[str, str]:
    profile_data = state.get("profile_data", {})
    my_service_info = state.get("my_service_info", "Soluciones de IA para empresas")
    company_tone = state.get("company_tone", "profesional y cercano")

    logger.info("[Agente 3] Copywriter generando email...")

    email_result = _get_chain().invoke(
        {
            "business_summary": profile_data.get("business_summary", ""),
            "pain_points": _as_bullets(profile_data.get("pain_points", [])),
            "technology": _as_bullets(profile_data.get("technology", [])),
            "opportunities": _as_bullets(profile_data.get("opportunities", [])),
            "ideal_customer": profile_data.get("ideal_customer", ""),
            "my_service_info": my_service_info,
            "company_tone": company_tone,
        }
    )

    logger.info("[Agente 3] Copywriter completado")

    run_id = state.get("run_id")
    if run_id:
        out_path = save_text_output(run_id, "agente3_email.txt", email_result)
        logger.info("[Agente 3] Guardado: %s", out_path)

    return {"final_email": email_result}
