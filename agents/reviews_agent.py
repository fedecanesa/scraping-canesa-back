# agents/reviews_agent.py

"""
ReviewsAgent - Obtiene reseñas de Google Maps via Apify.
Enriquece el análisis del prospecto con la voz real de sus clientes.
"""

import logging
from typing import Any
from urllib.parse import urlparse

from apify_client import ApifyClient

from config import settings


logger = logging.getLogger(__name__)

APIFY_ACTOR_ID = "compass/crawler-google-places"
MAX_REVIEWS = 10


def _extract_business_name(target_url: str) -> str:
    """Extrae el nombre probable del negocio desde el dominio de la URL."""
    try:
        domain = urlparse(target_url).netloc.replace("www.", "")
        name = domain.split(".")[0]
        return name.replace("-", " ").replace("_", " ").title()
    except Exception:
        return ""


def _fetch_google_reviews(business_name: str) -> list[dict[str, Any]]:
    """Busca reviews en Google Maps para el nombre del negocio dado."""
    client = ApifyClient(settings.apify_api_token)

    run_input: dict[str, Any] = {
        "searchStringsArray": [business_name],
        "maxCrawledPlacesPerSearch": 1,
        "maxReviews": MAX_REVIEWS,
        "language": "es",
        "includeWebResults": False,
        "reviewsSort": "newest",
    }

    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
    dataset = client.dataset(run["defaultDatasetId"])

    reviews: list[dict[str, Any]] = []
    for item in dataset.iterate_items():
        for review in item.get("reviews") or []:
            text = review.get("text") or review.get("body") or ""
            if not text.strip():
                continue
            reviews.append({
                "rating": review.get("stars") or review.get("rating"),
                "text": text.strip(),
                "date": review.get("publishAt") or review.get("publishedAtDate") or "",
            })
            if len(reviews) >= MAX_REVIEWS:
                break
        if reviews:
            break  # Solo usamos el primer lugar encontrado

    return reviews


def reviews_node(state: dict[str, Any]) -> dict[str, Any]:
    target_url = state.get("target_url", "")
    business_name = _extract_business_name(target_url)

    logger.info("[ReviewsAgent] Buscando reviews de Google Maps para: %s", business_name)

    reviews: list[dict[str, Any]] = []
    try:
        if not settings.apify_api_token:
            logger.warning("[ReviewsAgent] APIFY_API_TOKEN no configurado, saltando")
        elif not business_name:
            logger.warning("[ReviewsAgent] No se pudo extraer nombre del negocio de: %s", target_url)
        else:
            reviews = _fetch_google_reviews(business_name)
            logger.info("[ReviewsAgent] %s reviews obtenidas", len(reviews))
    except Exception:
        logger.exception("[ReviewsAgent] Error al obtener reviews, continuando sin ellas")

    return {
        "reviews_data": reviews,
        "business_name": business_name,
    }
