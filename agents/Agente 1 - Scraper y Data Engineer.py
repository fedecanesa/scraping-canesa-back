"""
Agente 1 - Scraper y Data Engineer

Este agente realiza dos funciones principales:
1. Scrapeo: Utiliza el Website Content Crawler de Apify (apify/website-content-crawler)
   para extraer contenido de una website.
2. Limpieza: Pasa los datos scrapeados por una LLM (OpenAI) vía LangChain v1
   para limpiar y normalizar el contenido.

Documentación LangChain v1: https://docs.langchain.com/oss/python/releases/langchain-v1
Website Content Crawler: https://apify.com/apify/website-content-crawler
"""

import json
import os
from typing import Any

from apify_client import ApifyClient

SALIDAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Salidas de los Agentes")
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage


# --- Configuración ---
APIFY_ACTOR_ID = "apify/website-content-crawler"
MAX_CHUNK_CHARS = 8000  # Límite aproximado para evitar exceder contexto del LLM


def scrape_website(
    url: str,
    max_crawl_pages: int = 1,
    max_crawl_depth: int | None = None,
    use_markdown: bool = True,
    apify_token: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extrae contenido de una website usando el Website Content Crawler de Apify.

    Args:
        url: URL de inicio para el crawl (ej: "https://docs.apify.com/")
        max_crawl_pages: Número máximo de páginas a rastrear (default: 10)
        max_crawl_depth: Profundidad máxima del crawl (opcional)
        use_markdown: Si True, incluye el campo markdown; si False, solo text
        apify_token: Token de Apify (o APIFY_API_TOKEN en env)

    Returns:
        Lista de diccionarios con: url, text, markdown (opcional), metadata

    Raises:
        ValueError: Si no hay APIFY_API_TOKEN configurado
    """
    token = apify_token or os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise ValueError(
            "Se requiere APIFY_API_TOKEN. "
            "Configúralo en el entorno o pásalo como apify_token."
        )

    client = ApifyClient(token)

    run_input: dict[str, Any] = {
        "startUrls": [{"url": url}],
        "maxCrawlPages": max_crawl_pages,
    }
    if max_crawl_depth is not None:
        run_input["maxCrawlDepth"] = max_crawl_depth

    print("[Agente 1] Iniciando scrape con Apify...", flush=True)
    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
    dataset = client.dataset(run["defaultDatasetId"])

    results: list[dict[str, Any]] = []
    for item in dataset.iterate_items():
        entry: dict[str, Any] = {
            "url": item.get("url", ""),
            "text": item.get("text") or "",
            "metadata": item.get("metadata", {}),
        }
        if use_markdown and item.get("markdown"):
            entry["markdown"] = item["markdown"]
        results.append(entry)

    print(f"[Agente 1] Scrape completado: {len(results)} páginas", flush=True)
    return results


def _split_content_for_llm(content: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Divide contenido largo en chunks para no exceder el contexto del LLM."""
    if len(content) <= max_chars:
        return [content] if content.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = start + max_chars
        if end < len(content):
            # Intentar cortar en un salto de línea
            last_newline = content.rfind("\n", start, end)
            if last_newline > start:
                end = last_newline + 1
        chunks.append(content[start:end].strip())
        start = end

    return [c for c in chunks if c]


def clean_scraped_data(
    scraped_items: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    openai_api_key: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Limpia los datos scrapeados usando una LLM de OpenAI vía LangChain v1.

    La LLM elimina ruido, normaliza formato y preserva el contenido sustancial.

    Args:
        scraped_items: Salida de scrape_website()
        model: Modelo de OpenAI (default: gpt-4o-mini)
        temperature: Temperatura para la generación (0.0 para resultados deterministas)
        openai_api_key: API key de OpenAI (o OPENAI_API_KEY en env)
        system_prompt: Prompt de sistema personalizado para la limpieza

    Returns:
        Lista de diccionarios con url, cleaned_text, metadata (y markdown si existía)
    """
    if not scraped_items:
        return []

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Se requiere OPENAI_API_KEY. "
            "Configúralo en el entorno o pásalo como openai_api_key."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    default_system = """Eres un experto en limpieza de datos web. Tu tarea es limpiar el contenido scrapeado de una página web.

Reglas:
- Elimina elementos de interfaz residuales (ej: "Skip to main content", "On this page", enlaces de navegación)
- Quita repeticiones innecesarias y ruido
- Conserva el contenido sustancial: títulos, párrafos, listas, tablas en texto
- Normaliza espacios en blanco y saltos de línea
- Mantén la estructura lógica del contenido (títulos, subtítulos, párrafos)
- NO inventes ni añadas información que no esté en el texto original
- Devuelve ÚNICAMENTE el texto limpio, sin comentarios ni explicaciones"""

    sys_prompt = system_prompt or default_system

    cleaned: list[dict[str, Any]] = []

    for i, item in enumerate(scraped_items):
        url = item.get("url", "")
        # Preferir markdown si existe, es más estructurado
        content = item.get("markdown") or item.get("text", "")
        metadata = item.get("metadata", {})

        print(f"[Agente 1] Limpiando página {i + 1}/{len(scraped_items)}...", flush=True)
        if not content or not content.strip():
            cleaned.append({
                "url": url,
                "cleaned_text": "",
                "metadata": metadata,
                **({"markdown": item.get("markdown")} if "markdown" in item else {}),
            })
            continue

        chunks = _split_content_for_llm(content)
        cleaned_parts: list[str] = []

        for chunk in chunks:
            user_msg = f"Limpia el siguiente contenido extraído de {url}:\n\n{chunk}"
            response = llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_msg),
            ])
            if hasattr(response, "content") and response.content:
                cleaned_parts.append(response.content.strip())

        cleaned_text = "\n\n".join(cleaned_parts) if cleaned_parts else ""

        out: dict[str, Any] = {
            "url": url,
            "cleaned_text": cleaned_text,
            "metadata": metadata,
        }
        if "markdown" in item:
            out["markdown"] = item["markdown"]
        cleaned.append(out)

    print(f"[Agente 1] Limpieza completada: {len(cleaned)} páginas", flush=True)
    return cleaned


def data_engineer_node(state: dict) -> dict:
    """
    Nodo para LangGraph: scrapea y limpia la URL del estado.
    Espera state con 'target_url'; opcional: max_crawl_pages, my_service_info, company_tone.
    Devuelve {'cleaned_data': list[dict]}.
    """
    url = state.get("target_url", "")
    if not url:
        return {"cleaned_data": []}
    max_pages = state.get("max_crawl_pages", 10)
    max_depth = state.get("max_crawl_depth", 3)
    skip_cleaning = state.get("skip_cleaning", True)  # True por defecto para evitar timeout
    results = run_scraper_and_clean(url=url, max_crawl_pages=max_pages, max_crawl_depth=max_depth, skip_cleaning=skip_cleaning)

    # Guardar salida en documento para revisión
    run_id = state.get("run_id")
    if run_id:
        run_dir = os.path.join(SALIDAS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "agente1_datos_limpios.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Agente 1] Guardado: {out_path}", flush=True)

    return {"cleaned_data": results}


def run_scraper_and_clean(
    url: str,
    max_crawl_pages: int = 10,
    max_crawl_depth: int = 3,
    model: str = "gpt-4o-mini",
    skip_cleaning: bool = False,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Pipeline completo: scrapea una website y opcionalmente limpia los datos con LLM.

    Args:
        url: URL de la website a scrapear
        max_crawl_pages: Máximo de páginas a rastrear
        model: Modelo OpenAI para la limpieza
        skip_cleaning: Si True, salta la limpieza con LLM (más rápido, usa texto raw)
        **kwargs: Argumentos adicionales para scrape_website y clean_scraped_data

    Returns:
        Lista de diccionarios con url, cleaned_text, metadata
    """
    scraped = scrape_website(url=url, max_crawl_pages=max_crawl_pages, max_crawl_depth=max_crawl_depth, **kwargs)
    if skip_cleaning:
        print("[Agente 1] Modo rápido: omitiendo limpieza LLM, pasando a Profiler...", flush=True)
        # Pasar texto raw al Profiler (mucho más rápido, evita timeout)
        return [
            {
                "url": item.get("url", ""),
                "cleaned_text": item.get("markdown") or item.get("text", ""),
                "metadata": item.get("metadata", {}),
            }
            for item in scraped
        ]
    return clean_scraped_data(scraped_items=scraped, model=model, **kwargs)


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Requiere: APIFY_API_TOKEN y OPENAI_API_KEY en el entorno
    url_ejemplo = "https://docs.apify.com/"
    print("Scrapeando y limpiando...")
    resultados = run_scraper_and_clean(url_ejemplo, max_crawl_pages=3)
    for r in resultados:
        print(f"\n--- {r['url']} ---")
        print(r["cleaned_text"][:500] + "..." if len(r["cleaned_text"]) > 500 else r["cleaned_text"])
