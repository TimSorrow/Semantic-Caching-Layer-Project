from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "llama3.2:latest"
    SIMILARITY_THRESHOLD: float = 0.90
    CACHE_TTL: int = 3600
    VECTOR_DIMENSION: int = 768
    
    # GCP Vertex AI Configurations
    USE_VERTEX_AI: bool = False
    GCP_PROJECT_ID: str = ""
    GCP_REGION: str = "us-central1"
    VERTEX_EMBEDDING_MODEL: str = "text-multilingual-embedding-002"
    VERTEX_LLM_MODEL: str = "gemini-2.5-flash"
    REDIS_URL: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
