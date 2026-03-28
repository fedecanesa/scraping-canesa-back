# tests/test_schemas.py

"""
Tests de validación para los schemas Pydantic.
"""

import pytest
from pydantic import ValidationError

from schemas.process import ProcessRequest, ProcessStatusResponse


class TestProcessRequest:
    def test_valid_minimal(self):
        req = ProcessRequest(target_url="https://example.com")
        assert req.skip_cleaning is True
        assert req.max_crawl_pages == 1
        assert req.max_crawl_depth == 1

    def test_valid_full(self):
        req = ProcessRequest(
            target_url="https://example.com",
            max_crawl_pages=5,
            max_crawl_depth=2,
            skip_cleaning=False,
            my_service_info="Agencia de marketing",
            company_tone="formal",
        )
        assert req.max_crawl_pages == 5
        assert req.my_service_info == "Agencia de marketing"

    def test_rejects_url_without_scheme(self):
        with pytest.raises(ValidationError) as exc_info:
            ProcessRequest(target_url="example.com")
        assert "http" in str(exc_info.value).lower()

    def test_strips_whitespace_from_url(self):
        req = ProcessRequest(target_url="  https://example.com  ")
        assert req.target_url == "https://example.com"

    def test_rejects_missing_url(self):
        with pytest.raises(ValidationError):
            ProcessRequest()


class TestProcessStatusResponse:
    def test_valid_running_response(self):
        resp = ProcessStatusResponse(
            run_id="abc123",
            target_url="https://example.com",
            status="running",
            current_step="DataEngineer",
            steps={"DataEngineer": "running", "Profiler": "pending", "Copywriter": "pending"},
            created_at="2026-03-28T10:00:00+00:00",
        )
        assert resp.finished_at is None
        assert resp.result is None
        assert resp.error is None

    def test_valid_completed_response(self):
        resp = ProcessStatusResponse(
            run_id="abc123",
            target_url="https://example.com",
            status="completed",
            current_step="completed",
            steps={"DataEngineer": "completed", "Profiler": "completed", "Copywriter": "completed"},
            created_at="2026-03-28T10:00:00+00:00",
            finished_at="2026-03-28T10:02:00+00:00",
            result={
                "final_email": "Hola...",
                "profile_data": None,
                "target_url": "https://example.com",
                "run_id": "abc123",
            },
        )
        assert resp.result.final_email == "Hola..."
