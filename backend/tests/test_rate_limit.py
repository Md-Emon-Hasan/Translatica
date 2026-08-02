"""
Rate Limiting Tests
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.core.rate_limit import get_client_ip


def make_request(headers=None, client_host="10.0.0.1"):
    """Build a minimal request-like object for key function tests."""
    request = MagicMock()
    request.headers = headers or {}
    request.client = MagicMock()
    request.client.host = client_host
    return request


class TestClientIpResolution:
    """The key function must see through the reverse proxy."""

    def test_prefers_first_forwarded_for_entry(self):
        request = make_request({"X-Forwarded-For": "203.0.113.7, 70.41.3.18"})
        assert get_client_ip(request) == "203.0.113.7"

    def test_falls_back_to_remote_address(self):
        assert get_client_ip(make_request()) == "10.0.0.1"

    def test_ignores_empty_forwarded_for(self):
        request = make_request({"X-Forwarded-For": " "})
        assert get_client_ip(request) == "10.0.0.1"


@pytest.mark.asyncio
class TestTranslateRateLimit:
    """Limits on the translate endpoint."""

    async def test_under_the_limit_succeeds(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_TRANSLATE", "5/minute")

        for _ in range(5):
            response = await client.post("/translate", json={"text": "Hello world"})
            assert response.status_code == 200

    async def test_over_the_limit_returns_429(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_TRANSLATE", "2/minute")

        for _ in range(2):
            assert (
                await client.post("/translate", json={"text": "Hello world"})
            ).status_code == 200

        response = await client.post("/translate", json={"text": "Hello world"})

        assert response.status_code == 429
        body = response.json()
        assert "Rate limit exceeded" in body["detail"]
        assert body["error"] == body["detail"]
        assert body["retry_after"] >= 1
        assert int(response.headers["Retry-After"]) >= 1

    async def test_forwarded_for_values_get_separate_buckets(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_TRANSLATE", "1/minute")
        payload = {"text": "Hello world"}

        first = await client.post(
            "/translate", json=payload, headers={"X-Forwarded-For": "1.1.1.1"}
        )
        other = await client.post(
            "/translate", json=payload, headers={"X-Forwarded-For": "2.2.2.2"}
        )
        repeat = await client.post(
            "/translate", json=payload, headers={"X-Forwarded-For": "1.1.1.1"}
        )

        assert first.status_code == 200
        assert other.status_code == 200
        assert repeat.status_code == 429

    async def test_disabling_rate_limiting_removes_all_limits(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(settings, "RATE_LIMIT_TRANSLATE", "1/minute")
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)

        for _ in range(5):
            response = await client.post("/translate", json={"text": "Hello world"})
            assert response.status_code == 200


@pytest.mark.asyncio
class TestUnlimitedEndpoints:
    """Health and model-info are never rate limited."""

    async def test_health_is_never_limited(self, client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_TRANSLATE", "1/minute")

        for _ in range(25):
            assert (await client.get("/health")).status_code == 200

    async def test_model_info_is_never_limited(self, client):
        for _ in range(25):
            assert (await client.get("/model-info")).status_code == 200


@pytest.mark.asyncio
class TestReadEndpointLimits:
    """History and stats share the read-endpoint limits."""

    async def test_history_is_limited(self, temp_db_client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_HISTORY", "1/minute")

        assert (await temp_db_client.get("/history")).status_code == 200
        assert (await temp_db_client.get("/history")).status_code == 429

    async def test_stats_is_limited(self, temp_db_client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_STATS", "1/minute")

        assert (await temp_db_client.get("/stats")).status_code == 200
        assert (await temp_db_client.get("/stats")).status_code == 429

    async def test_batch_has_its_own_bucket(self, pipeline_client, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_BATCH", "1/minute")

        payload = {"texts": ["Hello world."]}
        assert (
            await pipeline_client.post("/translate/batch", json=payload)
        ).status_code == 200
        assert (
            await pipeline_client.post("/translate/batch", json=payload)
        ).status_code == 429
        assert (
            await pipeline_client.post("/translate", json={"text": "Hello world."})
        ).status_code == 200
