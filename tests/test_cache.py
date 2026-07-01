import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)

@pytest.fixture
def mock_get_embedding():
    with patch("app.main.get_embedding", new_callable=AsyncMock) as mock:
        # Mock returning a 768-dimensional float list
        mock.return_value = [0.1] * 768
        yield mock

@pytest.fixture
def mock_generate_llm_response():
    with patch("app.main.generate_llm_response", new_callable=AsyncMock) as mock:
        mock.return_value = "Mocked Ollama LLM Response"
        yield mock

@pytest.fixture
def mock_search_cache():
    with patch("app.main.search_cache", new_callable=AsyncMock) as mock:
        yield mock

@pytest.fixture
def mock_store_cache():
    with patch("app.main.store_cache", new_callable=AsyncMock) as mock:
        yield mock

@pytest.fixture
def mock_invalidate_cache():
    with patch("app.main.invalidate_cache", new_callable=AsyncMock) as mock:
        yield mock

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_cache_hit(mock_get_embedding, mock_search_cache, mock_generate_llm_response):
    # Set up cache hit response
    mock_search_cache.return_value = {
        "query": "test query",
        "response": "Cached response here",
        "similarity": 0.95,
        "context_hash": "test_hash"
    }

    response = client.post(
        "/api/v1/query",
        json={"query": "test query", "context_hash": "test_hash"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Cached response here"
    assert data["status"] == "HIT"
    assert data["similarity"] == 0.95
    # Verify that LLM fallback was NOT called
    mock_generate_llm_response.assert_not_called()

def test_query_cache_miss(mock_get_embedding, mock_search_cache, mock_generate_llm_response, mock_store_cache):
    # Set up cache miss response
    mock_search_cache.return_value = None

    response = client.post(
        "/api/v1/query",
        json={"query": "test query", "context_hash": "test_hash"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Mocked Ollama LLM Response"
    assert data["status"] == "MISS"
    assert data["similarity"] is None
    
    # Verify that LLM fallback was called
    mock_generate_llm_response.assert_called_once()
    # Verify that result was stored in the cache
    mock_store_cache.assert_called_once()

def test_query_cache_context_hash_mismatch(mock_get_embedding, mock_search_cache, mock_generate_llm_response, mock_store_cache):
    # Set up cache hit response with a different context_hash
    mock_search_cache.return_value = {
        "query": "test query",
        "response": "Old cached response",
        "similarity": 0.95,
        "context_hash": "old_hash"
    }

    response = client.post(
        "/api/v1/query",
        json={"query": "test query", "context_hash": "new_hash"}
    )
    
    assert response.status_code == 200
    data = response.json()
    # Mismatch leads to a Cache Miss, meaning it queries the LLM and gets the fresh response
    assert data["response"] == "Mocked Ollama LLM Response"
    assert data["status"] == "MISS"
    
    # Verify LLM was called and stored
    mock_generate_llm_response.assert_called_once()
    mock_store_cache.assert_called_once()

def test_invalidate_cache_endpoint(mock_invalidate_cache):
    mock_invalidate_cache.return_value = 5

    response = client.post(
        "/api/v1/invalidate",
        json={"context_hash": "test_hash"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["invalidated_count"] == 5
    mock_invalidate_cache.assert_called_once_with("test_hash")

@patch("app.main.get_redis_client")
def test_list_cache_keys(mock_get_redis, mock_search_cache):
    # Mock Redis client response
    mock_client = MagicMock()
    mock_get_redis.return_value = mock_client
    
    # Mock keys method returning bytes keys
    mock_client.keys = AsyncMock(return_value=[b"cache:key_1"])
    mock_client.hgetall = AsyncMock(return_value={
        b"query": b"test query",
        b"response": b"test response",
        b"context_hash": b"test_hash"
    })
    
    response = client.get("/api/v1/cache/keys")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["key"] == "cache:key_1"
    assert data[0]["query"] == "test query"
    assert data[0]["response"] == "test response"
    assert data[0]["context_hash"] == "test_hash"

@patch("app.main.get_redis_client")
def test_delete_cache_key(mock_get_redis):
    mock_client = MagicMock()
    mock_get_redis.return_value = mock_client
    mock_client.delete = AsyncMock(return_value=1)
    
    response = client.delete("/api/v1/cache/keys/cache:key_1")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_client.delete.assert_called_once_with("cache:key_1")

