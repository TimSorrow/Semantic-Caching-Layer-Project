import logging
import time
import uuid
import numpy as np
from redis.asyncio import Redis
from app.core.config import settings

logger = logging.getLogger(__name__)

# Single instance of the Redis client (no decoding globally to avoid corrupting binary vectors)
redis_client: Redis | None = None

def get_redis_client() -> Redis:
    """Returns an active Redis client connection from the pool."""
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=False)
    return redis_client

async def init_cache_index() -> None:
    """
    Checks if the Redis index (e.g., 'idx:semantic_cache') exists.
    If not, creates it using FT.CREATE with a vector field (HNSW, Cosine similarity, 768 dimensions).
    """
    client = get_redis_client()
    try:
        # Check if index exists by querying it
        await client.execute_command("FT.INFO", "idx:semantic_cache")
        logger.info("Redis search index 'idx:semantic_cache' already exists.")
    except Exception as e:
        if "Unknown index name" in str(e) or "not found" in str(e).lower():
            logger.info("Creating Redis search index 'idx:semantic_cache'...")
            try:
                await client.execute_command(
                    "FT.CREATE", "idx:semantic_cache",
                    "ON", "HASH",
                    "PREFIX", "1", "cache:",
                    "SCHEMA",
                    "query", "TEXT",
                    "response", "TEXT",
                    "context_hash", "TAG",
                    "embedding", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(settings.VECTOR_DIMENSION),
                    "DISTANCE_METRIC", "COSINE"
                )
                logger.info("Redis search index 'idx:semantic_cache' created successfully.")
            except Exception as create_err:
                logger.error(f"Failed to create Redis search index: {create_err}")
                raise create_err
        else:
            logger.error(f"Error checking Redis index status: {e}")
            # Do not raise here to allow application to degrade gracefully if Redis is offline

def parse_search_results(res) -> list[dict]:
    """
    Parses RediSearch FT.SEARCH response into a list of parsed document dictionaries.
    Supports both dictionary (new redis-py) and list (old redis-py) response formats.
    """
    parsed_docs = []
    
    if isinstance(res, dict):
        results = res.get(b'results') or res.get('results') or []
        for doc in results:
            doc_id = doc.get(b'id') or doc.get('id')
            if isinstance(doc_id, bytes):
                doc_id = doc_id.decode('utf-8')
                
            fields_source = doc.get(b'extra_attributes') or doc.get('extra_attributes')
            if not fields_source:
                fields_source = doc.get(b'values') or doc.get('values')
                
            parsed_fields = {"id": doc_id}
            
            if isinstance(fields_source, dict):
                for k, v in fields_source.items():
                    k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                    if isinstance(v, bytes):
                        if k_str != "embedding":
                            v = v.decode('utf-8')
                    parsed_fields[k_str] = v
            elif isinstance(fields_source, list):
                for i in range(0, len(fields_source), 2):
                    k = fields_source[i]
                    k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                    v = fields_source[i+1]
                    if isinstance(v, bytes):
                        if k_str != "embedding":
                            v = v.decode('utf-8')
                    parsed_fields[k_str] = v
            parsed_docs.append(parsed_fields)
            
    elif isinstance(res, list):
        if len(res) > 1:
            total = res[0]
            if total > 0:
                i = 1
                while i < len(res):
                    doc_id = res[i]
                    if isinstance(doc_id, bytes):
                        doc_id = doc_id.decode('utf-8')
                    
                    fields_source = res[i+1]
                    parsed_fields = {"id": doc_id}
                    
                    if isinstance(fields_source, list):
                        for j in range(0, len(fields_source), 2):
                            k = fields_source[j]
                            k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                            v = fields_source[j+1]
                            if isinstance(v, bytes):
                                if k_str != "embedding":
                                    v = v.decode('utf-8')
                            parsed_fields[k_str] = v
                    i += 2
                    parsed_docs.append(parsed_fields)
    return parsed_docs

async def search_cache(vector: list[float], threshold: float, context_hash: str = None) -> dict | None:
    """
    Performs a vector search (KNN) via FT.SEARCH using the 768-dimensional vector.
    Applies pre-filtering by context_hash if provided, otherwise searches globally.
    Calculates cosine similarity and returns the cached response if the score >= threshold.
    Returns None if no matching entry is found.
    """
    client = get_redis_client()
    try:
        # Convert vector to float32 binary blob
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        # Construct hybrid vector search query with pre-filtering
        if context_hash:
            query_str = f"@context_hash:{{{context_hash}}} => [KNN 1 @embedding $vector AS similarity_score]"
        else:
            query_str = f"* => [KNN 1 @embedding $vector AS similarity_score]"
            
        res = await client.execute_command(
            "FT.SEARCH", "idx:semantic_cache",
            query_str,
            "PARAMS", "2", "vector", vector_bytes,
            "SORTBY", "similarity_score", "ASC",
            "DIALECT", "2"
        )
        
        # Parse search results using the robust helper function
        parsed_docs = parse_search_results(res)
        
        if not parsed_docs:
            return None
            
        best_doc = parsed_docs[0]
        
        # The similarity score returned is the distance. 
        # Let's extract and calculate true cosine similarity.
        distance = float(best_doc.get("similarity_score", 1.0))
        similarity = 1.0 - distance
        
        if similarity >= threshold:
            best_doc["similarity"] = similarity
            return best_doc
            
        return None
    except Exception as e:
        logger.error(f"Redis cache search failed (gracefully degrading): {e}", exc_info=True)
        return None

async def store_cache(query: str, response: str, vector: list[float], context_hash: str = None) -> None:
    """
    Saves the query, response, embedding vector, and context hash to Redis.
    Applies the configured CACHE_TTL.
    """
    client = get_redis_client()
    try:
        cache_id = str(uuid.uuid4())
        key = f"cache:{cache_id}"
        
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        mapping = {
            "query": query.encode('utf-8'),
            "response": response.encode('utf-8'),
            "embedding": vector_bytes,
            "created_at": str(time.time()).encode('utf-8')
        }
        if context_hash:
            mapping["context_hash"] = context_hash.encode('utf-8')
            
        async with client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, settings.CACHE_TTL)
            await pipe.execute()
            
        logger.info(f"Cached query successfully with key: {key}")
    except Exception as e:
        logger.error(f"Failed to write to Redis cache: {e}")

async def invalidate_cache(context_hash: str) -> int:
    """
    Deletes cache entries that match a specific context hash.
    Returns the count of invalidated records.
    """
    client = get_redis_client()
    try:
        # Query matching context_hash TAG
        query_str = f"@context_hash:{{{context_hash}}}"
        res = await client.execute_command(
            "FT.SEARCH", "idx:semantic_cache",
            query_str, "NOCONTENT", "DIALECT", "2"
        )
        
        # Parse count and keys based on response type
        keys = []
        count = 0
        if isinstance(res, dict):
            # Dict format
            total = res.get(b'total_results') or res.get('total_results') or 0
            count = int(total)
            results = res.get(b'results') or res.get('results') or []
            for doc in results:
                doc_id = doc.get(b'id') or doc.get('id')
                if doc_id:
                    keys.append(doc_id)
        elif isinstance(res, list):
            # List format
            if len(res) > 0:
                count = int(res[0])
                keys = [key for key in res[1:]]

        if keys:
            await client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} keys matching context_hash: {context_hash}")
        return count
    except Exception as e:
        logger.error(f"Redis cache invalidation failed: {e}", exc_info=True)
        return 0
