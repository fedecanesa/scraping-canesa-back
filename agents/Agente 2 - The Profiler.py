"""
Agente 2 - The Profiler

Analiza el modelo de negocio a partir de los datos scrapeados y limpios del Agente 1.
Enfocado en: Puntos de Dolor, tecnología usada, carencias y cliente ideal.

Recibe: cleaned_data (salida del Agente 1)
Devuelve: profile_data (perfil de negocio para el Agente 3)
"""

import os
from typing import Any, TypedDict

SALIDAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Salidas de los Agentes")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# --- Tipos para el grafo ---
class ProfilerState(TypedDict, total=False):
    """Estado esperado: cleaned_data viene del Agente 1."""
    cleaned_data: list[dict[str, Any]]
    profile_data: str
    url: str


# --- Cadena LCEL (definida fuera del nodo) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

prompt_profiler = ChatPromptTemplate.from_template("""Analiza el siguiente contenido extraído de una website y genera un **Perfil de Negocio** estructurado.

## Contenido de la website:
{website_content}

## Tu análisis debe cubrir:

1. **Puntos de Dolor / Necesidades**: ¿Qué problemas intenta resolver? ¿Qué necesidades cubre?

2. **Tecnología**: ¿Qué tecnologías mencionan o utilizan? (frameworks, herramientas, integraciones, etc.)

3. **Carencias / Oportunidades**: ¿Parece que les falta algo? ¿Qué podría mejorar o complementar su oferta?

4. **Cliente ideal**: ¿Quién es su público objetivo? Describe el perfil del cliente ideal basándote en el contenido.

Responde en formato claro y estructurado, con secciones para cada punto. Usa únicamente la información presente en el contenido; no inventes datos.
""")

profiler_chain = prompt_profiler | llm | StrOutputParser()


def _format_cleaned_data_for_prompt(cleaned_data: list[dict[str, Any]]) -> str:
    """Convierte la salida del Agente 1 en texto para el prompt."""
    parts: list[str] = []
    for item in cleaned_data:
        url = item.get("url", "(sin URL)")
        text = item.get("cleaned_text", "")
        if text:
            parts.append(f"### Página: {url}\n{text}")
    return "\n\n---\n\n".join(parts) if parts else "(No hay contenido disponible)"


def profiler_node(state: ProfilerState) -> dict[str, str]:
    """
    Nodo que analiza el modelo de negocio usando la cadena LCEL.
    Recibe cleaned_data del Agente 1 y devuelve profile_data.
    """
    cleaned_data = state.get("cleaned_data", [])
    website_content = _format_cleaned_data_for_prompt(cleaned_data)

    print("[Agente 2] Profiler ejecutando análisis...", flush=True)
    profile_result = profiler_chain.invoke({
        "website_content": website_content,
    })
    print("[Agente 2] Profiler completado", flush=True)

    # Guardar salida en documento para revisión
    run_id = state.get("run_id")
    if run_id:
        run_dir = os.path.join(SALIDAS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "agente2_perfil.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(profile_result)
        print(f"[Agente 2] Guardado: {out_path}", flush=True)

    return {"profile_data": profile_result}
