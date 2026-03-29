# services/places_search.py

"""
Búsqueda de empresas por categoría + ciudad usando Google Places API v1 (New).

Usa un solo request con field mask para obtener nombre, website, dirección,
rating y cantidad de reviews en una sola llamada.
"""

import logging
from typing import Any

import httpx

from config import settings
from schemas.search import PlaceResult


logger = logging.getLogger(__name__)

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Campos que pedimos — solo los necesarios para minimizar costo
FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.websiteUri",
    "places.formattedAddress",
    "places.rating",
    "places.userRatingCount",
])


def search_places(category: str, city: str, max_results: int = 20) -> list[PlaceResult]:
    """
    Busca empresas en Google Places usando la API v1 (New).

    Args:
        category: Tipo de negocio a buscar (ej: "agencias de marketing")
        city: Ciudad o región (ej: "Buenos Aires")
        max_results: Máximo de resultados a devolver (máx 20 por la API)

    Returns:
        Lista de PlaceResult con name, website, address, rating
    """
    api_key = settings.google_maps_api_key
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY no está configurado")

    query = f"{category} en {city}"
    logger.info("[Places] Buscando: %s", query)

    payload: dict[str, Any] = {
        "textQuery": query,
        "pageSize": min(max_results, 20),
        "languageCode": "es",
    }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    with httpx.Client(timeout=15.0) as client:
        response = client.post(PLACES_SEARCH_URL, json=payload, headers=headers)

    if response.status_code != 200:
        logger.error("[Places] Error %s: %s", response.status_code, response.text)
        raise RuntimeError(f"Google Places API error {response.status_code}: {response.text}")

    data = response.json()
    places_raw = data.get("places", [])

    results: list[PlaceResult] = []
    for place in places_raw:
        website = place.get("websiteUri")

        # Ignorar resultados sin website — no tienen caso en DeepReacher
        if not website:
            continue

        results.append(PlaceResult(
            place_id=place.get("id", ""),
            name=place.get("displayName", {}).get("text", ""),
            website=website,
            address=place.get("formattedAddress"),
            rating=place.get("rating"),
            rating_count=place.get("userRatingCount"),
        ))

    logger.info("[Places] %s resultados con website (de %s totales)", len(results), len(places_raw))
    return results
