# agents/data_engineer.py

"""
Agente 1 - Scraper y Data Engineer

Funciones:
1. Scrapea una website con Apify
2. Opcionalmente limpia el contenido con OpenAI
3. Devuelve cleaned_data para el siguiente agente
"""

import logging
import os
from typing import Any

from apify_client import ApifyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils.file_storage import save_json_output


logger = logging.getLogger(__name__)

APIFY_ACTOR_ID = "apify/website-content-crawler"
MAX_CHUNK_CHARS = 8000


def scrape_website(
    url: str,
    max_crawl_pages: int = 10,
    max_crawl_depth: int | None = None,
    use_markdown: bool = True,
    apify_token: str | None = None,
) -> list[dict[str, Any]]:
    token = apify_token or os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise ValueError(
            "Se requiere APIFY_API_TOKEN. Configúralo en el entorno o pásalo manualmente."
        )

    client = ApifyClient(token)

    run_input: dict[str, Any] = {
        "startUrls": [{"url": url}],
        "maxCrawlPages": max_crawl_pages,
    }

    if max_crawl_depth is not None:
        run_input["maxCrawlDepth"] = max_crawl_depth

    logger.info("[Agente 1] Iniciando scrape con Apify...")
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

    logger.info("[Agente 1] Scrape completado: %s páginas", len(results))
    return results


def _split_content_for_llm(content: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(content) <= max_chars:
        return [content] if content.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(content):
        end = start + max_chars

        if end < len(content):
            last_newline = content.rfind("\n", start, end)
            if last_newline > start:
                end = last_newline + 1

        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


def clean_scraped_data(
    scraped_items: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    openai_api_key: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    if not scraped_items:
        return []

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Se requiere OPENAI_API_KEY. Configúralo en el entorno o pásalo manualmente."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    default_system = """Eres un experto en limpieza de datos web. Tu tarea es limpiar el contenido scrapeado de una página web.

Reglas:
- Elimina elementos de interfaz residuales
- Quita repeticiones innecesarias y ruido
- Conserva el contenido sustancial: títulos, párrafos, listas y tablas en texto
- Normaliza espacios en blanco y saltos de línea
- Mantén la estructura lógica del contenido
- NO inventes ni añadas información que no esté en el texto original
- Devuelve ÚNICAMENTE el texto limpio, sin comentarios ni explicaciones"""

    sys_prompt = system_prompt or default_system
    cleaned: list[dict[str, Any]] = []

    total_items = len(scraped_items)

    for i, item in enumerate(scraped_items, start=1):
        url = item.get("url", "")
        content = item.get("markdown") or item.get("text", "")
        metadata = item.get("metadata", {})

        logger.info("[Agente 1] Limpiando página %s/%s...", i, total_items)

        if not content or not content.strip():
            cleaned.append(
                {
                    "url": url,
                    "cleaned_text": "",
                    "metadata": metadata,
                    **({"markdown": item.get("markdown")} if "markdown" in item else {}),
                }
            )
            continue

        chunks = _split_content_for_llm(content)
        cleaned_parts: list[str] = []

        for chunk in chunks:
            user_msg = f"Limpia el siguiente contenido extraído de {url}:\n\n{chunk}"

            response = llm.invoke(
                [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=user_msg),
                ]
            )

            if hasattr(response, "content") and response.content:
                cleaned_parts.append(str(response.content).strip())

        cleaned_text = "\n\n".join(cleaned_parts) if cleaned_parts else ""

        out: dict[str, Any] = {
            "url": url,
            "cleaned_text": cleaned_text,
            "metadata": metadata,
        }

        if "markdown" in item:
            out["markdown"] = item["markdown"]

        cleaned.append(out)

    logger.info("[Agente 1] Limpieza completada: %s páginas", len(cleaned))
    return cleaned


def run_scraper_and_clean(
    url: str,
    max_crawl_pages: int = 10,
    max_crawl_depth: int = 3,
    model: str = "gpt-4o-mini",
    skip_cleaning: bool = False,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    scraped = scrape_website(
        url=url,
        max_crawl_pages=max_crawl_pages,
        max_crawl_depth=max_crawl_depth,
        **kwargs,
    )

    if skip_cleaning:
        logger.info("[Agente 1] Modo rápido: omitiendo limpieza LLM")

        return [
            {
                "url": item.get("url", ""),
                "cleaned_text": item.get("markdown") or item.get("text", ""),
                "metadata": item.get("metadata", {}),
            }
            for item in scraped
        ]

    return clean_scraped_data(scraped_items=scraped, model=model, **kwargs)


def data_engineer_node(state: dict[str, Any]) -> dict[str, Any]:
    url = state.get("target_url", "")
    if not url:
        return {"cleaned_data": []}

    run_id = state.get("run_id")
    max_pages = state.get("max_crawl_pages", 10)
    max_depth = state.get("max_crawl_depth", 3)
    skip_cleaning = state.get("skip_cleaning", True)

    results = run_scraper_and_clean(
        url=url,
        max_crawl_pages=max_pages,
        max_crawl_depth=max_depth,
        skip_cleaning=skip_cleaning,
    )

    if run_id:
        out_path = save_json_output(run_id, "agente1_datos_limpios.json", results)
        logger.info("[Agente 1] Guardado: %s", out_path)

    return {"cleaned_data": results}