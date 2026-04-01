# services/run_manager.py

"""
In-memory run store para el pipeline de agentes.

Thread-safe: usa un lock para proteger acceso concurrente desde background tasks.

Limitación conocida: los runs se pierden al reiniciar el servidor.
Para producción, reemplazar _runs por Redis o una DB.
"""

import threading
from datetime import datetime, timezone
from typing import Any


_runs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

PIPELINE_STEPS = ["DataEngineer", "ReviewsAgent", "JobsAgent", "Profiler", "Copywriter"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run(run_id: str, target_url: str) -> None:
    """Registra un nuevo run con estado inicial 'running'."""
    with _lock:
        _runs[run_id] = {
            "run_id": run_id,
            "target_url": target_url,
            "status": "running",
            "current_step": PIPELINE_STEPS[0],
            "steps": {step: "pending" for step in PIPELINE_STEPS},
            "created_at": _now_iso(),
            "finished_at": None,
            "result": None,
            "error": None,
        }


def get_run(run_id: str) -> dict[str, Any] | None:
    """Devuelve una copia del estado del run, o None si no existe."""
    with _lock:
        run = _runs.get(run_id)
        return dict(run) if run else None


def update_step(run_id: str, step: str, status: str) -> None:
    """Actualiza el estado de un paso y el current_step del run."""
    with _lock:
        run = _runs.get(run_id)
        if run:
            run["steps"][step] = status
            run["current_step"] = step


def complete_run(run_id: str, pipeline_output: dict[str, Any]) -> None:
    """
    Marca el run como completado y extrae el resultado del output del pipeline.

    No sobreescribe si el run ya está en estado 'failed' (ej: por timeout).
    """
    with _lock:
        run = _runs.get(run_id)
        if run and run["status"] != "failed":
            run["status"] = "completed"
            run["current_step"] = "completed"
            run["finished_at"] = _now_iso()
            run["result"] = {
                "final_email": pipeline_output.get("final_email"),
                "message_variants": pipeline_output.get("message_variants"),
                "profile_data": pipeline_output.get("profile_data"),
                "target_url": pipeline_output.get("target_url"),
                "run_id": run_id,
            }
            for step in PIPELINE_STEPS:
                run["steps"][step] = "completed"


def fail_run(run_id: str, error: str) -> None:
    """Marca el run como fallido con el mensaje de error."""
    with _lock:
        run = _runs.get(run_id)
        if run:
            run["status"] = "failed"
            run["finished_at"] = _now_iso()
            run["error"] = error
