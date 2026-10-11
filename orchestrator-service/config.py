"""
config.py

Centralised settings for the orchestrator gateway. Values come from environment
variables or ``orchestrator-service/.env`` (see ``.env.example``), with defaults for
the single-machine profile (Profile A in ARCHITECTURE_V2_PLAN.md).
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

SERVICE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""  # set in .env (never committed)

    # ML engine (stateless scoring service)
    ML_SERVICE_URL: str = "http://localhost:8001"
    ML_TIMEOUT_SECONDS: float = 60.0
    ML_BATCH_SIZE: int = 400

    # Local LLM(s): comma-separated, tried in order (Profile B: Windows first, Mac fallback).
    LLM_BASE_URLS: Annotated[list[str], NoDecode] = Field(default=["http://localhost:8080"])
    LLM_MODEL: str = "qwen3.5-9b-instruct"
    LLM_TIMEOUT_SECONDS: float = 180.0
    LLM_INLINE_WAIT_SECONDS: float = 10.0

    # Case assistant (chatbot): RTX first (fast CUDA prefill for long case context), Mac 9B as fallback.
    CHAT_LLM_URLS: Annotated[list[str], NoDecode] = Field(default=["http://localhost:8080"])
    CHAT_LLM_MODEL: str = "qwen3.5-9b-chat"
    CHAT_TIMEOUT_SECONDS: float = 90.0

    # Detection tuning
    INSIDER_WINDOW_HOURS: int = 48
    REPORTING_THRESHOLD: float = 10_000.0
    STRUCTURING_MARGIN: float = 1_000.0
    STRUCTURING_WINDOW_HOURS: int = 72
    STRUCTURING_MIN_COUNT: int = 3
    CYCLE_MAX_HOPS: int = 4
    CYCLE_MAX_DAYS: int = 7
    MAX_GRAPH_ACCOUNTS: int = 15

    # Ingestion
    INGEST_MAX_BATCH: int = 2_000

    # Gateway
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(default=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"])
    ALLOW_ADMIN: bool = False  # /admin/reset etc. Enable in .env only for local demos and tests.

    model_config = SettingsConfigDict(env_file=SERVICE_DIR / ".env", env_file_encoding="utf-8",
                                      extra="ignore", case_sensitive=True)

    @field_validator("LLM_BASE_URLS", "CHAT_LLM_URLS", "CORS_ORIGINS", mode="before")
    @classmethod
    def _split_csv(cls, value):
        if isinstance(value, str) and not value.strip().startswith("["):
            return [part.strip().rstrip("/") for part in value.split(",") if part.strip()]
        return value


settings = Settings()

if not settings.NEO4J_PASSWORD:
    raise SystemExit("NEO4J_PASSWORD is not set. Copy orchestrator-service/.env.example to .env and fill it in.")
