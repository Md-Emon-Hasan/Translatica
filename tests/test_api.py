"""
API Endpoint Tests
"""


class TestIndexEndpoint:
    """Tests for the index endpoint."""

    def test_index_returns_html(self, client):
        """Test that index returns HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Translatica" in response.text

    def test_index_contains_form(self, client):
        """Test that index contains the translation form."""
        response = client.get("/")
        assert response.status_code == 200
        assert "inputText" in response.text
        assert "Translate" in response.text


class TestTranslateEndpoint:
    """Tests for the translate endpoint."""

    def test_translate_success(self, client):
        """Test successful translation."""
        response = client.post("/translate", json={"text": "Hello world"})

        assert response.status_code == 200
        data = response.json()
        assert "translation" in data

    def test_translate_empty_text(self, client):
        """Test translation with empty text."""
        response = client.post("/translate", json={"text": ""})

        assert response.status_code == 422  # Validation error

    def test_translate_whitespace_only(self, client):
        """Test translation with whitespace only."""
        response = client.post("/translate", json={"text": "   "})

        # Should either be 400 or 200 depending on service handling
        assert response.status_code in [200, 400]

    def test_translate_missing_text_field(self, client):
        """Test translation with missing text field."""
        response = client.post("/translate", json={})

        assert response.status_code == 422

    def test_translate_long_text(self, client, sample_texts):
        """Test translation with long text."""
        response = client.post("/translate", json={"text": sample_texts["long_text"]})

        assert response.status_code == 200
        data = response.json()
        assert "translation" in data

    def test_translate_special_characters(self, client):
        """Test translation with special characters."""
        response = client.post("/translate", json={"text": "Hello! How are you?"})

        assert response.status_code == 200
        data = response.json()
        assert "translation" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestAPIDocumentation:
    """Tests for API documentation."""

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_available(self, client):
        """Test that Swagger UI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """Test that ReDoc documentation is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
