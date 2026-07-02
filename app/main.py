import os
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
    invalidate_cache,
    get_redis_client
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure static directory exists
    os.makedirs("app/static", exist_ok=True)
    
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

# Mount static files (will create dashboard assets here)
# Make sure we mount it AFTER defining routes or handle fallback correctly.
# But we can also mount it at /static and serve / directly with FileResponse.

import asyncio

# Global flag to track Vertex AI initialization status
_vertex_initialized = False

def _generate_vertex_gemini_sync(prompt: str) -> str:
    global _vertex_initialized
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    if not _vertex_initialized:
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION)
        _vertex_initialized = True
        
    model = GenerativeModel(settings.VERTEX_LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text

async def generate_llm_response(prompt: str) -> str:
    """
    Sends a chat generation query to the configured model.
    Uses Google Vertex AI (Gemini 1.5 Flash) if settings.USE_VERTEX_AI is True,
    otherwise queries the local Ollama instance (Llama 3.2).
    """
    if settings.USE_VERTEX_AI:
        try:
            return await asyncio.to_thread(_generate_vertex_gemini_sync, prompt)
        except Exception as e:
            logger.error(f"Vertex AI Gemini chat generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate text from Vertex AI Gemini: [{type(e).__name__}] {str(e)}")

    # Fallback to local Ollama
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

async def check_rate_limit(request: Request) -> None:
    """
    Simple IP-based rate limiter using Redis to protect Vertex AI / Ollama APIs.
    Limits clients to 15 requests per minute.
    """
    client_ip = request.client.host if request.client else "unknown"
    limit = 15
    window = 60
    
    try:
        redis_client = get_redis_client()
        if redis_client:
            key = f"rate_limit:{client_ip}"
            current = await redis_client.incr(key)
            if current == 1:
                await redis_client.expire(key, window)
            if current > limit:
                logger.warning(f"Rate limit exceeded for IP: {client_ip} (Requests: {current})")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests from this IP. Please try again in a minute."
                )
    except HTTPException:
        raise
    except Exception as e:
        # Gracefully degrade: if Redis fails, do not block the pipeline
        logger.error(f"Rate limiter failed: {e}")
        pass

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest, request: Request):
    """
    Query endpoint that implements the semantic caching flow:
    1. Apply IP-based rate limiting.
    2. Generate embedding vector for the query via local Ollama.
    3. Search the Redis vector cache.
    4. If HIT and context_hash matches (if provided):
       - Return the cached response with metadata status="HIT".
    5. If MISS (or context_hash mismatch):
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

    # Step-2: Search Cache (use custom threshold if passed, fallback to settings default)
    threshold = payload.threshold if payload.threshold is not None else settings.SIMILARITY_THRESHOLD
    cache_hit, closest_similarity = await search_cache(vector, threshold, context_hash)
    
    if cache_hit:
        cached_context_hash = cache_hit.get("context_hash")
        
        # Verify context_hash matches if it was provided
        if context_hash is None or context_hash == cached_context_hash:
            logger.info("Semantic cache HIT!")
            return QueryResponse(
                response=cache_hit["response"],
                status="HIT",
                similarity=closest_similarity
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
        similarity=closest_similarity
    )

@app.post("/api/v1/invalidate", response_model=InvalidateResponse)
async def invalidate_endpoint(payload: InvalidateRequest):
    """
    Invalidation endpoint that clears cache entries associated with a specific context_hash.
    """
    invalidated_count = await invalidate_cache(payload.context_hash)
    return InvalidateResponse(invalidated_count=invalidated_count)

@app.get("/api/v1/cache/keys")
async def list_cache_keys():
    """
    Retrieves metadata of all entries currently stored in Redis.
    """
    client = get_redis_client()
    try:
        keys = await client.keys("cache:*")
        details = []
        for key in keys:
            mapping = await client.hgetall(key)
            decoded = {"key": key.decode('utf-8') if isinstance(key, bytes) else key}
            for k, v in mapping.items():
                k_str = k.decode('utf-8') if isinstance(k, bytes) else k
                if k_str != "embedding":
                    decoded[k_str] = v.decode('utf-8') if isinstance(v, bytes) else v
            details.append(decoded)
        return details
    except Exception as e:
        logger.error(f"Failed to retrieve cache keys: {e}")
        return []

@app.delete("/api/v1/cache/keys/{key_id}")
async def delete_cache_key(key_id: str):
    """
    Deletes a specific cache key from Redis.
    """
    client = get_redis_client()
    try:
        # Key in database format is cache:<uuid>
        res = await client.delete(key_id)
        return {"status": "ok" if res else "not_found"}
    except Exception as e:
        logger.error(f"Failed to delete cache key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_endpoint():
    """
    Verify application health, including checking Redis connection.
    """
    client = get_redis_client()
    try:
        await client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
        
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "online" if redis_ok else "offline"
    }

# Serve Frontend SPA
@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse("app/static/index.html")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

