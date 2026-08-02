"""
History and Statistics Endpoint Tests
"""

from datetime import datetime, timedelta

import pytest

from app.core.config import settings
from app.models.translation import Translation

pytestmark = pytest.mark.asyncio


async def seed(client, rows):
    """Insert (source, translated, timestamp) rows into the temporary database."""
    async with client._session_factory() as session:
        for source, translated, timestamp in rows:
            session.add(
                Translation(
                    source_text=source,
                    translated_text=translated,
                    timestamp=timestamp,
                )
            )
        await session.commit()


class TestHistoryEmptyDatabase:
    """An empty log is a valid state, not an error."""

    async def test_history_returns_empty_page(self, temp_db_client):
        response = await temp_db_client.get("/history")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["items"] == []
        assert body["limit"] == settings.HISTORY_DEFAULT_LIMIT
        assert body["offset"] == 0

    async def test_stats_returns_zeros(self, temp_db_client):
        response = await temp_db_client.get("/stats")

        assert response.status_code == 200
        body = response.json()
        assert body["total_translations"] == 0
        assert body["translations_today"] == 0
        assert body["translations_last_7_days"] == 0
        assert body["translations_last_30_days"] == 0
        assert body["avg_source_chars"] == 0.0
        assert body["avg_translated_chars"] == 0.0
        assert body["daily_counts"] == []


class TestHistoryPagination:
    """Pagination bounds and validation."""

    async def test_pagination_and_total_count(self, temp_db_client):
        now = datetime.utcnow()
        await seed(
            temp_db_client,
            [
                (f"Source {i}", f"Fuente {i}", now - timedelta(minutes=i))
                for i in range(25)
            ],
        )

        first = (await temp_db_client.get("/history?limit=10")).json()
        assert first["total"] == 25
        assert len(first["items"]) == 10
        assert first["items"][0]["source_text"] == "Source 0"

        second = (await temp_db_client.get("/history?limit=10&offset=10")).json()
        assert second["total"] == 25
        assert len(second["items"]) == 10
        assert second["items"][0]["source_text"] == "Source 10"

        last = (await temp_db_client.get("/history?limit=10&offset=20")).json()
        assert len(last["items"]) == 5

    async def test_oldest_first_is_available(self, temp_db_client):
        now = datetime.utcnow()
        await seed(
            temp_db_client,
            [
                ("Older", "Viejo", now - timedelta(hours=2)),
                ("Newer", "Nuevo", now),
            ],
        )

        body = (await temp_db_client.get("/history?newest_first=false")).json()
        assert body["items"][0]["source_text"] == "Older"

    async def test_limit_above_maximum_is_rejected(self, temp_db_client):
        response = await temp_db_client.get(
            f"/history?limit={settings.HISTORY_MAX_LIMIT + 1}"
        )
        assert response.status_code == 422

    async def test_zero_limit_is_rejected(self, temp_db_client):
        assert (await temp_db_client.get("/history?limit=0")).status_code == 422

    async def test_negative_offset_is_rejected(self, temp_db_client):
        assert (await temp_db_client.get("/history?offset=-1")).status_code == 422


class TestHistoryFilters:
    """Search and date-range filters."""

    async def test_substring_search_on_source_text(self, temp_db_client):
        now = datetime.utcnow()
        await seed(
            temp_db_client,
            [
                ("The lighthouse keeper", "El farero", now),
                ("A quiet harbour", "Un puerto tranquilo", now),
            ],
        )

        body = (await temp_db_client.get("/history?search=lighthouse")).json()
        assert body["total"] == 1
        assert body["items"][0]["source_text"] == "The lighthouse keeper"

    async def test_date_range_filter(self, temp_db_client):
        now = datetime.utcnow()
        await seed(
            temp_db_client,
            [
                ("Old row", "Fila vieja", now - timedelta(days=10)),
                ("New row", "Fila nueva", now),
            ],
        )

        start = (now - timedelta(days=1)).isoformat()
        body = (await temp_db_client.get(f"/history?start_date={start}")).json()
        assert body["total"] == 1
        assert body["items"][0]["source_text"] == "New row"

        end = (now - timedelta(days=5)).isoformat()
        body = (await temp_db_client.get(f"/history?end_date={end}")).json()
        assert body["total"] == 1
        assert body["items"][0]["source_text"] == "Old row"


class TestHistoryTruncation:
    """Long texts are shortened in list responses."""

    async def test_long_text_is_truncated_and_flagged(self, temp_db_client):
        limit = settings.HISTORY_TEXT_PREVIEW_CHARS
        await seed(
            temp_db_client,
            [("a" * (limit + 50), "b" * (limit + 50), datetime.utcnow())],
        )

        item = (await temp_db_client.get("/history")).json()["items"][0]
        assert len(item["source_text"]) == limit
        assert len(item["translated_text"]) == limit
        assert item["source_truncated"] is True
        assert item["translated_truncated"] is True

    async def test_short_text_is_not_flagged(self, temp_db_client):
        await seed(temp_db_client, [("Short", "Corto", datetime.utcnow())])

        item = (await temp_db_client.get("/history")).json()["items"][0]
        assert item["source_text"] == "Short"
        assert item["source_truncated"] is False
        assert item["translated_truncated"] is False


class TestStats:
    """Aggregate statistics."""

    async def test_counts_and_averages(self, temp_db_client):
        now = datetime.utcnow()
        await seed(
            temp_db_client,
            [
                ("abcd", "efghij", now),
                ("abcd", "efghij", now - timedelta(days=3)),
                ("abcd", "efghij", now - timedelta(days=10)),
                ("abcd", "efghij", now - timedelta(days=45)),
            ],
        )

        body = (await temp_db_client.get("/stats")).json()

        assert body["total_translations"] == 4
        assert body["translations_today"] == 1
        assert body["translations_last_7_days"] == 2
        assert body["translations_last_30_days"] == 3
        assert body["avg_source_chars"] == 4.0
        assert body["avg_translated_chars"] == 6.0

    async def test_daily_series_covers_the_last_30_days(self, temp_db_client):
        now = datetime.utcnow()
        await seed(
            temp_db_client,
            [
                ("one", "uno", now),
                ("two", "dos", now),
                ("three", "tres", now - timedelta(days=2)),
                ("old", "viejo", now - timedelta(days=40)),
            ],
        )

        series = (await temp_db_client.get("/stats")).json()["daily_counts"]

        assert len(series) == 2
        assert sum(day["count"] for day in series) == 3
        assert series[-1]["date"] == now.strftime("%Y-%m-%d")

    async def test_cache_statistics_are_included(self, temp_db_client):
        cache = (await temp_db_client.get("/stats")).json()["cache"]

        assert set(cache) == {"size", "maxsize", "ttl", "hits", "misses", "hit_rate"}
