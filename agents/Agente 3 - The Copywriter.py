"""
Agente 3 - The Copywriter

Toma los "dolores" detectados por el Agente 2 y redacta un Cold Email
ofreciendo servicios, conectando específicamente con lo que hace la empresa.

Misión: "La Conexión" — Unir punto A (Cliente) y punto B (Tu servicio).
"""

import os
import re
from typing import Any, TypedDict

SALIDAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Salidas de los Agentes")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# --- Tipos para el grafo ---
class CopywriterState(TypedDict, total=False):
    """Estado: profile_data del Agente 2 + config de tu servicio."""
    profile_data: str
    my_service_info: str
    company_tone: str
    final_email: str


# --- Cadena LCEL (definida fuera del nodo) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

prompt_copywriter = ChatPromptTemplate.from_template("""Eres un experto Copywriter de ventas B2B especializado en 'Cold Emailing'.

Tienes esta información del prospecto:
Negocio: {business_summary}
Puntos de dolor: {pain_points}

Nosotros vendemos: {my_service_info}

Escribe un email al CEO. Reglas:
- Empieza con un 'Icebreaker' genuino sobre algo específico que encontraste en su web (demuestra que los investigaste).
- Menciona el problema (Punto de dolor) usando sus propias palabras si es posible.
- Presenta nuestra solución de IA como el alivio natural a ese dolor.
- Call to Action (CTA) de bajo compromiso (ej: '¿Te envío un video de 3 min?').
- Tono: {company_tone} (Adáptate a su estilo).

Devuelve ÚNICAMENTE el email, sin introducciones ni explicaciones.
""")

copywriter_chain = prompt_copywriter | llm | StrOutputParser()


def _extract_pain_points(profile_data: str) -> str:
    """Extrae la sección 'Puntos de Dolor' del perfil del Agente 2."""
    if not profile_data:
        return "(No se detectaron puntos de dolor)"
    # Busca "Puntos de Dolor" o similar hasta el siguiente numeral/sección
    pattern = r"\*\*Puntos de Dolor[^*]*\*\*[:\s]*(.*?)(?=\n\n\d\.|\n\n\*\*|\n\n##|\Z)"
    match = re.search(pattern, profile_data, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: primeros párrafos o todo el perfil
    return profile_data[:1500] if len(profile_data) > 1500 else profile_data


def copywriter_node(state: CopywriterState) -> dict[str, str]:
    """
    Nodo que genera el Cold Email a partir del perfil del Agente 2.
    Recibe profile_data, my_service_info, company_tone y devuelve final_email.
    """
    profile_data = state.get("profile_data", "")
    my_service_info = state.get("my_service_info", "Soluciones de IA para empresas")
    company_tone = state.get("company_tone", "profesional y cercano")

    pain_points = _extract_pain_points(profile_data)

    print("[Agente 3] Copywriter generando email...", flush=True)
    email_result = copywriter_chain.invoke({
        "business_summary": profile_data,
        "pain_points": pain_points,
        "my_service_info": my_service_info,
        "company_tone": company_tone,
    })
    print("[Agente 3] Copywriter completado", flush=True)

    # Guardar salida en documento para revisión
    run_id = state.get("run_id")
    if run_id:
        run_dir = os.path.join(SALIDAS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "agente3_email.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(email_result)
        print(f"[Agente 3] Guardado: {out_path}", flush=True)

    return {"final_email": email_result}
