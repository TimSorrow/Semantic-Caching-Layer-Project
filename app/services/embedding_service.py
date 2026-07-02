import httpx
import logging
import asyncio
from app.core.config import settings

logger = logging.getLogger(__name__)

_vertex_initialized = False

def _get_vertex_embedding_sync(text: str) -> list[float]:
    global _vertex_initialized
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    
    if not _vertex_initialized:
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION)
        _vertex_initialized = True
        
    model = TextEmbeddingModel.from_pretrained(settings.VERTEX_EMBEDDING_MODEL)
    # The text-embedding-004 model supports dimensionality up to 768 natively
    embeddings = model.get_embeddings([text])
    return embeddings[0].values

async def get_embedding(text: str) -> list[float]:
    """
    Generates a 768-dimensional float vector for the input text.
    Uses Google Vertex AI (text-embedding-004) if settings.USE_VERTEX_AI is True,
    otherwise falls back to local Ollama (nomic-embed-text).
    """
    if settings.USE_VERTEX_AI:
        try:
            return await asyncio.to_thread(_get_vertex_embedding_sync, text)
        except Exception as e:
            logger.error(f"Failed to fetch embedding from Vertex AI: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate embedding via Vertex AI: [{type(e).__name__}] {str(e)}")
            
    # Fallback to local Ollama
    url = f"{settings.OLLAMA_URL}/api/embed"
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "input": text
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            embeddings = data.get("embeddings")
            if not embeddings or not isinstance(embeddings, list):
                raise ValueError("Ollama response did not contain expected 'embeddings' list.")
            
            # Ollama returns a list of embeddings. Since we passed a single string,
            # we extract the first list of floats.
            return embeddings[0]
    except Exception as e:
        logger.error(f"Failed to fetch embedding from Ollama: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate embedding via Ollama: [{type(e).__name__}] {str(e)}")
