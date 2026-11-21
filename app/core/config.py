from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "DocIntel-RAG"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Literal["dev", "staging", "prod"] = "dev"
    MAX_UPLOAD_MB: int = 25
    UPLOAD_DIR: Path = Path("./data/uploads")
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_URL: str | None = "http://localhost:6333"
    QDRANT_LOCAL_PATH: Path = Path("./data/qdrant_db")
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_API_KEY: str | None = None
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
