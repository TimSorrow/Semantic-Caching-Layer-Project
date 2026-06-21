import httpx
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

async def get_embedding(text: str) -> list[float]:
    """
    Generates a 768-dimensional float vector for the input text using Ollama's local embeddings API.
    Uses the model specified in settings (default: 'nomic-embed-text').
    """
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
