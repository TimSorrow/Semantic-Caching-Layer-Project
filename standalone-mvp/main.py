import os
import time
import hashlib
import asyncio
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery

app = FastAPI()

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.85
DISTANCE_THRESHOLD = 1.0 - SIMILARITY_THRESHOLD

# Initialize Redis client
redis_client = Redis.from_url(REDIS_URL)

# Initialize embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
VECTOR_DIMENSIONS = model.get_sentence_embedding_dimension()

# Define RedisVL schema
schema_dict = {
    "index": {
        "name": "semantic_cache",
        "prefix": "cache:semantic",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "prompt", "type": "text"},
        {"name": "response", "type": "text"},
        {
            "name": "prompt_embedding",
            "type": "vector",
            "attrs": {
                "dims": VECTOR_DIMENSIONS,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32"
            }
        }
    ]
}
schema = IndexSchema.from_dict(schema_dict)
index = SearchIndex(schema, redis_client)

@app.on_event("startup")
async def startup_event():
    # Create the index if it doesn't exist
    index.create(overwrite=False, drop=False)

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    response: str
    cache_tier: str

def get_exact_match(prompt: str) -> str | None:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    key = f"cache:exact:{prompt_hash}"
    result = redis_client.get(key)
    if result:
        return result.decode("utf-8")
    return None

def set_exact_match(prompt: str, response: str):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    key = f"cache:exact:{prompt_hash}"
    redis_client.set(key, response)

def get_semantic_match(prompt: str) -> str | None:
    # Generate embedding
    embedding = model.encode(prompt).tolist()
    
    # Query RedisVL
    query = VectorQuery(
        vector=embedding,
        vector_field_name="prompt_embedding",
        return_fields=["response", "vector_distance"],
        num_results=1
    )
    
    results = index.query(query)
    
    if results:
        best_match = results[0]
        # RedisVL returns distance
        distance = float(best_match["vector_distance"])
        if distance <= DISTANCE_THRESHOLD:
            return best_match["response"]
            
    return None

def set_semantic_match(prompt: str, response: str):
    embedding = model.encode(prompt)
    data = {
        "prompt": prompt,
        "response": response,
        "prompt_embedding": embedding.astype(np.float32).tobytes()
    }
    index.load([data])

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    prompt = request.prompt
    
    # Tier 1: Exact Match
    exact_hit = get_exact_match(prompt)
    if exact_hit:
        return GenerateResponse(response=exact_hit, cache_tier="tier1_exact")
        
    # Tier 2: Semantic Match
    semantic_hit = get_semantic_match(prompt)
    if semantic_hit:
        return GenerateResponse(response=semantic_hit, cache_tier="tier2_semantic")
        
    # Cache Miss: Simulate LLM Generation
    await asyncio.sleep(2)
    response_text = f"Generated response for: {prompt}"
    
    # Save to both tiers
    set_exact_match(prompt, response_text)
    set_semantic_match(prompt, response_text)
    
    return GenerateResponse(response=response_text, cache_tier="miss_generated")
