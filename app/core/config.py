from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Advanced RAG Backend"
    DEBUG: bool = True

    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b-instruct-q4_0"

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Groq Settings
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    EVALUATOR_MODEL: str = "llama-3.3-70b-versatile"
    OLLAMA_EVAL_MODEL: str = "llama3:8b-instruct-q4_0"

    # Comparison Models
    OLLAMA_MODELS: str = "llama3,llama3:8b-instruct-q4_0,llama3.2:1b"

    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PATH: str = "./storage"
    COLLECTION_NAME: str = "advanced_rag_collection"

    # Models
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    
    # RAG Settings
    CHUNKING_SIZE: int = 500
    CHUNKING_OVERLAP: int = 100
    USE_SEMANTIC_CHUNKING: bool = True
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
