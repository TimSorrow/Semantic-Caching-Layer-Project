# Semantic Caching Layer: Architectural Specification

This document defines the strict architectural specification, directory layout, dependency requirements, and module signatures for the Semantic Caching Layer MVP.

---

## 1. Technical Stack & Core Libraries

- **Framework**: FastAPI (Asynchronous API endpoints)
- **Database**: Redis (Redis Stack with RediSearch and Vector Similarity Search capabilities)
- **SDK**: Google GenAI SDK (`google-genai`)
- **Embedding Model**: `text-embedding-004` (strictly 768 dimensions)
- **Fallback LLM**: `gemini-3.5-flash`

---

## 2. Project Directory Structure

```
semantic-caching-layer/
├── devbox.json
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── cache.py
│   └── services/
│       ├── __init__.py
│       ├── cache_service.py
│       └── embedding_service.py
└── tests/
    ├── __init__.py
    └── test_cache.py
```

---

## 3. Required Dependencies (`requirements.txt`)

```text
fastapi>=0.110.0
uvicorn>=0.28.0
redis[hiredis]>=5.0.1
google-genai>=0.1.1
pydantic-settings>=2.2.1
pydantic>=2.6.4
numpy>=1.26.4
python-dotenv>=1.0.1
```

---

## 4. Architectural Components & Function Signatures

### `app/core/config.py`
Defines configuration validation via Pydantic.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    REDIS_URL: str = "redis://localhost:6379"
    EMBEDDING_MODEL: str = "text-embedding-004"
    LLM_MODEL: str = "gemini-3.5-flash"
    SIMILARITY_THRESHOLD: float = 0.90
    CACHE_TTL: int = 3600
    VECTOR_DIMENSION: int = 768

    class Config:
        env_file = ".env"

settings = Settings()
```

### `app/services/embedding_service.py`
Asynchronously generates query embeddings using the Google GenAI SDK.

```python
from google import genai

async def get_embedding(text: str) -> list[float]:
    """
    Generates a 768-dimensional float vector for the input text using 'text-embedding-004'.
    Uses the asynchronous Google GenAI SDK client.
    """
    pass
```

### `app/services/cache_service.py`
Manages Redis connections, search index creation, KNN similarity search, and data persistence.

```python
from redis.asyncio import Redis

async def get_redis_client() -> Redis:
    """Returns an active Redis client connection from the pool."""
    pass

async def init_cache_index() -> None:
    """
    Checks if the Redis index (e.g., 'idx:semantic_cache') exists.
    If not, creates it using FT.CREATE with a vector field (HNSW, Cosine similarity, 768 dimensions).
    """
    pass

async def search_cache(vector: list[float], threshold: float) -> dict | None:
    """
    Performs a vector search (KNN) via FT.SEARCH using the 768-dimensional vector.
    Calculates cosine similarity and returns the cached response if the score >= threshold.
    Returns None if no matching entry is found.
    """
    pass

async def store_cache(query: str, response: str, vector: list[float], context_hash: str = None) -> None:
    """
    Saves the query, response, embedding vector, and context hash to Redis.
    Applies the configured CACHE_TTL.
    """
    pass

async def invalidate_cache(context_hash: str) -> int:
    """
    Deletes cache entries that match a specific context hash.
    Returns the count of invalidated records.
    """
    pass
```

### `app/main.py`
The FastAPI application wrapper implementing the request handling logic flow.

```python
from fastapi import FastAPI
from app.schemas.cache import QueryRequest, QueryResponse

app = FastAPI(title="Semantic Caching Layer API")

@app.on_event("startup")
async def startup_event():
    # Initialize Redis search index
    pass

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    """
    Logic Flow:
    1. Generate embedding vector for payload.query using embedding_service.
    2. Search the Redis vector cache.
    3. If HIT:
       - Verify context_hash matches (if provided).
       - Return response with metadata status="HIT".
    4. If MISS:
       - Query fallback LLM (gemini-3.5-flash) using Google GenAI SDK.
       - Store result, query, vector, and context_hash in cache.
       - Return response with metadata status="MISS".
    """
    pass
```

---

## 5. Strict Constraints

- **Async-Only Interface**: All network I/O operations (Redis connections, Google GenAI SDK calls, and FastAPI endpoints) must run asynchronously.
- **Embedding Alignment**: The embedding dimension is strictly **768** to align with `text-embedding-004`. Vector schemas in Redis must use `768` dimensions with type `FLOAT32`.
- **Redis Error Handling**: If Redis is offline or fails, the application must catch the exception, log the failure, bypass the cache (Cache Miss fallback), and directly serve requests via the LLM API (graceful degradation).
- **Context Validation**: Cached values must match the client-supplied `context_hash` to ensure data consistency. A mismatch must trigger a cache miss.