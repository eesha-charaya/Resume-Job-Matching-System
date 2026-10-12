"""Tests for /health and /ready endpoints."""

import time
import numpy as np


def test_health_endpoint_returns_200(api_client):
    """GET /health returns 200 with {"status": "ok"}."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_response_time_under_50ms(api_client):
    """GET /health response time < 50ms (no I/O)."""
    start_time = time.time()
    response = api_client.get("/health")
    elapsed_ms = (time.time() - start_time) * 1000
    assert elapsed_ms < 50, f"Response time {elapsed_ms:.2f}ms exceeds 50ms limit"
    assert response.status_code == 200


def test_ready_all_healthy_returns_200(
    api_client, mock_db, mock_collection, mock_embedding, mock_ollama_health
):
    """GET /ready returns 200 when all dependencies are healthy."""
    mock_db.cursor().fetchone.return_value = (1,)
    mock_collection.count.return_value = 5
    mock_embedding.encode.return_value = np.asarray([[0.1] * 1024], dtype=np.float32)
    mock_ollama_health.return_value = (True, "Ollama is healthy")

    response = api_client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    checks = data["checks"]
    assert len(checks) == 4
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is True
    assert checks["embedding_model"]["healthy"] is True
    assert checks["ollama"]["healthy"] is True


def test_ready_db_unreachable_returns_503(api_client, mock_db, mock_ollama_health):
    """GET /ready returns 503 when DB is unreachable."""
    mock_db.cursor().execute.side_effect = Exception("Connection failed")
    mock_ollama_health.return_value = (True, "Ollama is healthy")

    response = api_client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    checks = data["checks"]
    assert data["status"] == "not ready"
    assert checks["database"]["healthy"] is False
    assert "error" in checks["database"]["message"].lower()
    assert checks["chroma"]["healthy"] is True
    assert checks["embedding_model"]["healthy"] is True
    assert checks["ollama"]["healthy"] is True


def test_ready_chroma_unreachable_returns_503(api_client, mock_collection, mock_ollama_health):
    """GET /ready returns 503 when ChromaDB is unreachable."""
    mock_collection.count.side_effect = Exception("Collection not found")
    mock_ollama_health.return_value = (True, "Ollama is healthy")

    response = api_client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is False
    assert "error" in checks["chroma"]["message"].lower()
    assert checks["embedding_model"]["healthy"] is True
    assert checks["ollama"]["healthy"] is True


def test_ready_embedding_model_failure_returns_503(
    api_client, mock_embedding, mock_ollama_health
):
    """GET /ready returns 503 when the local embedding model fails."""
    mock_embedding.encode.side_effect = Exception("Embedding model failure")
    mock_ollama_health.return_value = (True, "Ollama is healthy")

    response = api_client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is True
    assert checks["embedding_model"]["healthy"] is False
    assert "error" in checks["embedding_model"]["message"].lower()
    assert checks["ollama"]["healthy"] is True


def test_ready_ollama_unreachable_returns_503(api_client, mock_ollama_health):
    """GET /ready returns 503 when Ollama is unreachable."""
    mock_ollama_health.return_value = (False, "Connection failed")

    response = api_client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["database"]["healthy"] is True
    assert checks["chroma"]["healthy"] is True
    assert checks["embedding_model"]["healthy"] is True
    assert checks["ollama"]["healthy"] is False
    assert "connection" in checks["ollama"]["message"].lower()


def test_ready_ollama_missing_model_returns_503(api_client, mock_ollama_health):
    """GET /ready returns 503 when Ollama is missing the required model."""
    mock_ollama_health.return_value = (
        False,
        "Required model 'llama3.2:3b-instruct-q4_K_M' not found in Ollama",
    )

    response = api_client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["ollama"]["healthy"] is False
    assert "model" in checks["ollama"]["message"].lower()


def test_ready_partial_failure_shows_only_failing_checks(
    api_client, mock_db, mock_embedding, mock_ollama_health
):
    """GET /ready with partial failure shows only failing checks as errors."""
    mock_db.cursor().execute.side_effect = Exception("DB connection failed")
    mock_embedding.encode.side_effect = Exception("Embedding model failure")
    mock_ollama_health.return_value = (True, "Ollama is healthy")

    response = api_client.get("/ready")

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["database"]["healthy"] is False
    assert checks["chroma"]["healthy"] is True
    assert checks["embedding_model"]["healthy"] is False
    assert checks["ollama"]["healthy"] is True


def test_ready_checks_always_has_four_keys(api_client):
    """GET /ready always returns checks with exactly four keys."""
    response = api_client.get("/ready")
    checks = response.json()["checks"]

    assert set(checks.keys()) == {"database", "chroma", "embedding_model", "ollama"}
    assert len(checks) == 4
    for check in checks.values():
        assert "healthy" in check
        assert isinstance(check["healthy"], bool)
        assert "message" in check
        assert isinstance(check["message"], str)


def test_ready_response_time_under_2s(api_client):
    """GET /ready response time < 2s with mocked dependencies."""
    start_time = time.time()
    response = api_client.get("/ready")
    elapsed_s = time.time() - start_time
    assert elapsed_s < 2, f"Response time {elapsed_s:.2f}s exceeds 2s limit"
    assert response.status_code in [200, 503]
