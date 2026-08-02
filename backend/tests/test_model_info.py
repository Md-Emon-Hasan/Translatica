"""
Model Info Endpoint Tests
"""

import pytest

from app.core.config import settings

pytestmark = pytest.mark.asyncio


class TestModelInfoEndpoint:
    """Tests for GET /model-info."""

    async def test_all_expected_fields_are_present(self, client):
        response = await client.get("/model-info")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "base_model",
            "peft_type",
            "task_type",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "trainable_parameters",
            "generation",
            "limits",
            "model_loaded",
        }

    async def test_adapter_metadata_comes_from_the_adapter_config(self, client):
        body = (await client.get("/model-info")).json()

        assert body["base_model"] == "Helsinki-NLP/opus-mt-en-es"
        assert body["peft_type"] == "LORA"
        assert body["task_type"] == "SEQ_2_SEQ_LM"
        assert body["lora_rank"] == 8
        assert body["lora_alpha"] == 32
        assert body["target_modules"] == ["q_proj", "v_proj"]

    async def test_generation_values_come_from_config(self, client, monkeypatch):
        monkeypatch.setattr(settings, "GEN_NUM_BEAMS", 7)
        monkeypatch.setattr(settings, "GEN_MAX_NEW_TOKENS", 99)

        generation = (await client.get("/model-info")).json()["generation"]

        assert generation["num_beams"] == 7
        assert generation["max_new_tokens"] == 99
        assert generation["do_sample"] is False

    async def test_limits_come_from_config(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MAX_INPUT_CHARS", 4321)
        monkeypatch.setattr(settings, "CHUNK_TOKEN_BUDGET", 111)

        limits = (await client.get("/model-info")).json()["limits"]

        assert limits["max_input_chars"] == 4321
        assert limits["chunk_token_budget"] == 111
        assert limits["max_input_tokens"] == settings.MAX_INPUT_TOKENS
        assert limits["min_input_chars"] == settings.MIN_INPUT_CHARS
        assert limits["max_batch_size"] == settings.MAX_BATCH_SIZE
        assert limits["max_batch_chars"] == settings.MAX_BATCH_CHARS
        assert limits["max_output_length"] == settings.MAX_OUTPUT_LENGTH

    async def test_reports_whether_the_model_is_loaded(self, client):
        assert (await client.get("/model-info")).json()["model_loaded"] is True

        client._mock_manager.is_loaded = False
        try:
            assert (await client.get("/model-info")).json()["model_loaded"] is False
        finally:
            client._mock_manager.is_loaded = True

    async def test_missing_adapter_config_is_tolerated(
        self, client, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "missing")

        body = (await client.get("/model-info")).json()

        assert body["base_model"] is None
        assert body["target_modules"] == []
        assert body["generation"]["num_beams"] == settings.GEN_NUM_BEAMS

    async def test_no_metric_claims_are_reported(self, client):
        body = (await client.get("/model-info")).json()

        assert "bleu" not in {key.lower() for key in body}
