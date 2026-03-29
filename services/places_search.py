# services/places_search.py

"""
Búsqueda de empresas por categoría + ciudad usando Apify
actor: compass/crawler-google-places

Reutiliza el token de Apify ya configurado en settings.
"""

import logging
from typing import Any

from apify_client import ApifyClient

from config import settings
from schemas.search import PlaceResult


logger = logging.getLogger(__name__)

APIFY_ACTOR_ID = "compass/crawler-google-places"


def search_places(category: str, city: str, max_results: int = 20) -> list[PlaceResult]:
    """
    Busca empresas en Google Maps vía Apify.

    Args:
        category: Tipo de negocio (ej: "agencias de marketing")
        city: Ciudad o región (ej: "Buenos Aires")
        max_results: Máximo de resultados a devolver

    Returns:
        Lista de PlaceResult con name, website, address, rating
    """
    token = settings.apify_api_token
    if not token:
        raise ValueError("APIFY_API_TOKEN no está configurado")

    query = f"{category} en {city}"
    logger.info("[Places] Buscando en Apify Google Maps: %s", query)

    client = ApifyClient(token)

    run_input: dict[str, Any] = {
        "searchStringsArray": [query],
        "maxCrawledPlacesPerSearch": min(max_results, 20),
        "language": "es",
        "countryCode": "",
        "includeWebResults": False,
    }

    run = client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
    dataset = client.dataset(run["defaultDatasetId"])

    results: list[PlaceResult] = []

    for item in dataset.iterate_items():
        website = item.get("website")

        # Sin website no tiene caso en DeepReacher
        if not website:
            continue

        # Normalizar website — a veces viene sin esquema
        if website and not website.startswith("http"):
            website = f"https://{website}"

        place_id = item.get("placeId") or item.get("id") or ""
        name = item.get("title") or item.get("name") or ""
        address = item.get("address") or item.get("street") or ""
        rating = item.get("totalScore") or item.get("rating")
        rating_count = item.get("reviewsCount") or item.get("userRatingsTotal")

        results.append(PlaceResult(
            place_id=place_id,
            name=name,
            website=website,
            address=address if address else None,
            rating=float(rating) if rating else None,
            rating_count=int(rating_count) if rating_count else None,
        ))

        if len(results) >= max_results:
            break

    logger.info("[Places] %s resultados con website", len(results))
    return results
