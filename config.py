# config.py

"""
Configuración centralizada usando pydantic-settings.

Lee variables de entorno directamente (y de .env si existe).
Esto hace que load_dotenv() sea redundante pero no rompe nada si se mantiene.

Uso:
    from config import settings
    settings.openai_api_key
    settings.pipeline_timeout_seconds
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- APIs externas ---
    openai_api_key: str = ""
    apify_api_token: str = ""
    google_maps_api_key: str = ""

    # --- Seguridad ---
    # Si se define, el header X-Api-Key es obligatorio en POST /process.
    # Dejar vacío para deshabilitar auth (útil en desarrollo local).
    api_key: str = ""

    # --- Pipeline ---
    pipeline_timeout_seconds: int = 300  # 5 minutos

    # --- Rate limiting ---
    # Requests por minuto por IP para POST /process
    rate_limit_per_minute: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignora vars de entorno que no están definidas aquí
    )


settings = Settings()
