# agents/copywriter.py

"""
Agente 3 - The Copywriter (DeepReacher edition)

Dos estrategias completamente diferenciadas:
- sell: vendedor experto, conecta dolor con solución, CTA directo
- partnership: par a par, propuesta de alianza, tono de colega no de vendedor
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


# ─── Prompt SELL ──────────────────────────────────────────────────────────────

_PROMPT_SELL = ChatPromptTemplate.from_template(
    """Eres un experto en cold outreach B2B con más de 10 años generando reuniones para agencias y consultoras.

PERFIL DEL PROSPECTO:
Negocio: {what_they_do}
Problemas detectados:
{issues_list}
Oportunidades:
{opportunities_list}
Señales de compra: {buying_signals}
Tecnología: {technology}

NOSOTROS OFRECEMOS: {my_service_info}
TONO: {company_tone}

Generá 3 mensajes de venta B2B. Devuelve SOLO JSON válido:

{{
  "main": "...",
  "variant_a": "...",
  "variant_b": "..."
}}

ESTRATEGIA DE CADA VARIANTE:

**main — El equilibrado que convierte:**
- Icebreaker genuino: mencioná algo MUY específico del negocio del prospecto (algo que viste en su web, no genérico)
- Conectá directamente con el problema más urgente detectado
- Presentá nuestro servicio como la solución natural, no como una venta forzada
- CTA de bajo compromiso: "¿Tiene 15 minutos esta semana?" o similar
- Máximo 100 palabras

**variant_a — El directo orientado a resultado:**
- Empieza con el impacto/resultado concreto que podemos generar (ej: "Empresas como la tuya aumentan un 30% sus leads en 60 días con...")
- Muy corto, muy al grano
- CTA proactivo: proponé una fecha o un demo específico
- Máximo 80 palabras

**variant_b — El consultivo que genera confianza:**
- Empieza con una pregunta de diagnóstico inteligente que demuestra que entendés su negocio (ej: "¿Cuánto tiempo invierte tu equipo en X sin resultados claros?")
- Posicionarte como experto que puede ayudar, no como vendedor
- CTA de consulta gratuita o conversación sin compromiso
- Máximo 100 palabras

REGLAS PARA LOS 3 MENSAJES:
- NUNCA empieces con "Hola, espero que estés bien" o frases genéricas de spam
- NUNCA digas "Me permito escribirte" o "Mi nombre es X de empresa Y"
- El prospecto tiene que sentir que este mensaje fue escrito SOLO para él
- Referenciá algo específico de su negocio o web en cada mensaje
- Tono: {company_tone}
- Devuelve SOLO JSON, sin markdown, sin explicaciones
"""
)

# ─── Prompt PARTNERSHIP ───────────────────────────────────────────────────────

_PROMPT_PARTNERSHIP = ChatPromptTemplate.from_template(
    """Eres un experto en desarrollo de alianzas estratégicas B2B. Tu comunicación es de par a par: no vendés, proponés construir algo juntos.

PERFIL DEL POTENCIAL SOCIO:
Negocio: {what_they_do}
Sus fortalezas: {what_doing_well}
Oportunidades de alianza detectadas:
{opportunities_list}
Sus brechas (donde nosotros podríamos complementar):
{issues_list}
Tecnología: {technology}

NOSOTROS SOMOS: {my_service_info}
TONO: {company_tone}

Generá 3 mensajes de propuesta de alianza. Devuelve SOLO JSON válido:

{{
  "main": "...",
  "variant_a": "...",
  "variant_b": "..."
}}

ESTRATEGIA DE CADA VARIANTE:

**main — La propuesta de alianza directa:**
- Icebreaker: algo concreto que admirás o reconocés de su trabajo (genuino, basado en lo que viste)
- Explicá brevemente quiénes somos y qué hacemos
- Identificá el punto de complementariedad específico: "Vos hacés X, nosotros hacemos Y. Juntos podríamos..."
- Proponé un formato concreto de colaboración (referidos, co-ejecución, white-label)
- CTA: "¿Tenés 20 minutos para explorar si tiene sentido?"
- Máximo 120 palabras

**variant_a — El enfoque en cliente compartido:**
- Empieza identificando el tipo de cliente que ambos atendemos (sin revelar datos de clientes)
- Planteá la oportunidad de generar más valor juntos para ese cliente
- Proponé un acuerdo de referidos o co-ejecución específico
- CTA: una llamada corta o un café virtual
- Máximo 100 palabras

**variant_b — El enfoque en proyecto concreto:**
- Mencioná un tipo de proyecto donde claramente se necesitan las dos especialidades
- Proponé arrancar con algo pequeño: un proyecto piloto o una propuesta conjunta para un cliente
- Tono muy colega, muy informal si el tono lo permite
- CTA: compartir un caso reciente donde esto hubiera servido
- Máximo 100 palabras

REGLAS PARA LOS 3 MENSAJES:
- El tono es de IGUAL A IGUAL. No sos un proveedor buscando clientes, sos un potencial socio
- NUNCA uses lenguaje de ventas: "oferta", "servicio", "precio", "propuesta comercial"
- Usá "colaborar", "construir juntos", "sumar fuerzas", "complementarnos"
- El prospecto tiene que sentir que los elegiste a ellos específicamente, no que mandaste esto a 100 empresas
- Tono: {company_tone}
- Devuelve SOLO JSON, sin markdown, sin explicaciones
"""
)


def _get_chain(objective: str):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=settings.openai_api_key)
    prompt = _PROMPT_SELL if objective == "sell" else _PROMPT_PARTNERSHIP
    return prompt | llm | StrOutputParser()


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

    opportunities = profile_data.get("opportunities", [])
    issues = profile_data.get("issues", [])
    buying_signals = profile_data.get("buying_signals", [])
    what_doing_well = profile_data.get("what_doing_well", [])

    logger.info("[Agente 3] Copywriter generando 3 variantes (objetivo=%s)...", objective)

    invoke_params = {
        "what_they_do": profile_data.get("what_they_do", profile_data.get("business_summary", "")),
        "issues_list": _as_bullets(issues, key="title"),
        "opportunities_list": _as_bullets(opportunities, key="title"),
        "technology": ", ".join(profile_data.get("technology", [])) or "No detectada",
        "my_service_info": my_service_info,
        "company_tone": company_tone,
    }

    if objective == "sell":
        invoke_params["buying_signals"] = ", ".join(buying_signals) or "No detectadas"
    else:
        invoke_params["what_doing_well"] = _as_bullets(what_doing_well)

    raw_result = _get_chain(objective).invoke(invoke_params)

    message_variants = _parse_variants(raw_result)
    final_email = message_variants[0]["content"] if message_variants else raw_result.strip()

    logger.info("[Agente 3] Copywriter completado — %s variantes", len(message_variants))

    run_id = state.get("run_id")
    if run_id:
        save_json_output(run_id, "agente3_variantes.json", message_variants)
        save_text_output(run_id, "agente3_email.txt", final_email)

    return {
        "final_email": final_email,
        "message_variants": message_variants,
    }
