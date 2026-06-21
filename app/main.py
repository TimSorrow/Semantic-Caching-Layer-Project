import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status

from app.core.config import settings
from app.schemas.cache import (
    QueryRequest,
    QueryResponse,
    InvalidateRequest,
    InvalidateResponse
)
from app.services.embedding_service import get_embedding
from app.services.cache_service import (
    init_cache_index,
    search_cache,
    store_cache,
    invalidate_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Redis search index on startup
    logger.info("Initializing application and Redis index...")
    try:
        await init_cache_index()
    except Exception as e:
        logger.error(f"Could not initialize Redis search index on startup: {e}")
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="Semantic Caching Layer API (Local Stack)",
    description="MVP for a Semantic Caching Layer using FastAPI, Redis Vector Search, and Ollama (Gemma 4).",
    version="1.0.0",
    lifespan=lifespan
)

async def generate_llm_response(prompt: str) -> str:
    """
    Sends a chat generation query to the local Ollama instance using the configured model.
    """
    url = f"{settings.OLLAMA_URL}/api/chat"
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "")
            if not content:
                raise ValueError("Ollama response did not contain message content.")
            return content
    except Exception as e:
        logger.error(f"Ollama chat generation failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate text from Ollama LLM: [{type(e).__name__}] {str(e)}")

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    """
    Query endpoint that implements the semantic caching flow:
    1. Generate embedding vector for the query via local Ollama.
    2. Search the Redis vector cache.
    3. If HIT and context_hash matches (if provided):
       - Return the cached response with metadata status="HIT".
    4. If MISS (or context_hash mismatch):
       - Query local LLM (Gemma 4) using Ollama.
       - Store the query, response, vector, and context_hash in the cache.
       - Return the response with metadata status="MISS".
    """
    query = payload.query
    context_hash = payload.context_hash

    # Step 1: Generate embedding
    try:
        vector = await get_embedding(query)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}. Falling back to direct LLM call without caching.")
        # Fallback to direct LLM call on embedding failures
        try:
            response_text = await generate_llm_response(query)
            return QueryResponse(
                response=response_text,
                status="MISS",
                similarity=None
            )
        except Exception as llm_err:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Both embedding generation and LLM generation failed: {llm_err}"
            )

    # Step-2: Search Cache
    cache_hit = await search_cache(vector, settings.SIMILARITY_THRESHOLD, context_hash)

    if cache_hit:
        cached_context_hash = cache_hit.get("context_hash")
        
        # Verify context_hash matches if it was provided
        if context_hash is None or context_hash == cached_context_hash:
            logger.info("Semantic cache HIT!")
            return QueryResponse(
                response=cache_hit["response"],
                status="HIT",
                similarity=float(cache_hit["similarity"])
            )
        else:
            logger.info("Semantic cache hit, but context_hash mismatched. Treating as MISS.")

    # Step 3: Cache Miss -> Fallback to Ollama LLM
    logger.info("Semantic cache MISS. Querying local fallback LLM...")
    try:
        response_text = await generate_llm_response(query)
    except Exception as e:
        logger.error(f"Failed to query local LLM: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to query local LLM: {e}"
        )

    # Step 4: Store result in Cache
    await store_cache(query, response_text, vector, context_hash)

    return QueryResponse(
        response=response_text,
        status="MISS",
        similarity=None
    )

@app.post("/api/v1/invalidate", response_model=InvalidateResponse)
async def invalidate_endpoint(payload: InvalidateRequest):
    """
    Invalidation endpoint that clears cache entries associated with a specific context_hash.
    """
    invalidated_count = await invalidate_cache(payload.context_hash)
    return InvalidateResponse(invalidated_count=invalidated_count)

@app.get("/health")
async def health_endpoint():
    """
    Verify application health.
    """
    return {"status": "ok"}
