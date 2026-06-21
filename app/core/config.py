from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "gemma4:latest"
    SIMILARITY_THRESHOLD: float = 0.90
    CACHE_TTL: int = 3600
    VECTOR_DIMENSION: int = 768
    REDIS_URL: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
