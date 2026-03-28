# tests/test_run_manager.py

"""
Tests unitarios para services/run_manager.py.

No requieren API keys ni red — testean solo el store in-memory.
"""

import pytest

from services.run_manager import (
    PIPELINE_STEPS,
    complete_run,
    create_run,
    fail_run,
    get_run,
    update_step,
)


@pytest.fixture(autouse=True)
def clean_runs():
    """Limpia el store entre tests para evitar interferencias."""
    from services import run_manager
    run_manager._runs.clear()
    yield
    run_manager._runs.clear()


class TestCreateRun:
    def test_creates_run_with_correct_fields(self):
        create_run("abc123", "https://example.com")
        run = get_run("abc123")

        assert run is not None
        assert run["run_id"] == "abc123"
        assert run["target_url"] == "https://example.com"
        assert run["status"] == "running"
        assert run["current_step"] == PIPELINE_STEPS[0]
        assert run["result"] is None
        assert run["error"] is None
        assert run["finished_at"] is None

    def test_all_steps_start_as_pending(self):
        create_run("abc123", "https://example.com")
        run = get_run("abc123")
        assert all(v == "pending" for v in run["steps"].values())

    def test_get_run_returns_copy(self):
        """Modificar el resultado no debe afectar el store."""
        create_run("abc123", "https://example.com")
        run = get_run("abc123")
        run["status"] = "hacked"
        assert get_run("abc123")["status"] == "running"


class TestGetRun:
    def test_returns_none_for_unknown_run(self):
        assert get_run("no-existe") is None


class TestUpdateStep:
    def test_updates_step_status_and_current_step(self):
        create_run("abc123", "https://example.com")
        update_step("abc123", "DataEngineer", "running")

        run = get_run("abc123")
        assert run["steps"]["DataEngineer"] == "running"
        assert run["current_step"] == "DataEngineer"

    def test_noop_on_unknown_run(self):
        # No debe lanzar excepción
        update_step("no-existe", "DataEngineer", "running")


class TestCompleteRun:
    def test_marks_completed_and_extracts_result(self):
        create_run("abc123", "https://example.com")
        pipeline_output = {
            "final_email": "Hola CEO...",
            "profile_data": {"business_summary": "Una empresa", "pain_points": []},
            "target_url": "https://example.com",
        }
        complete_run("abc123", pipeline_output)
        run = get_run("abc123")

        assert run["status"] == "completed"
        assert run["current_step"] == "completed"
        assert run["finished_at"] is not None
        assert run["result"]["final_email"] == "Hola CEO..."
        assert run["result"]["run_id"] == "abc123"
        assert all(v == "completed" for v in run["steps"].values())

    def test_does_not_overwrite_failed_run(self):
        """Si ya falló (timeout), complete_run no debe sobreescribir."""
        create_run("abc123", "https://example.com")
        fail_run("abc123", "Timeout")
        complete_run("abc123", {"final_email": "tarde"})

        run = get_run("abc123")
        assert run["status"] == "failed"
        assert run["result"] is None


class TestFailRun:
    def test_marks_failed_with_error(self):
        create_run("abc123", "https://example.com")
        fail_run("abc123", "Apify error 500")
        run = get_run("abc123")

        assert run["status"] == "failed"
        assert run["error"] == "Apify error 500"
        assert run["finished_at"] is not None

    def test_noop_on_unknown_run(self):
        fail_run("no-existe", "error")
