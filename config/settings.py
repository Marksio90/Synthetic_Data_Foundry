"""
config/settings.py — Centralised, type-safe configuration via pydantic-settings.
All values come from environment variables (see .env.example).
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(
        ...,
        description="Sync SQLAlchemy URL (psycopg2)",
    )
    async_database_url: str = Field(
        ...,
        description="Async SQLAlchemy URL (asyncpg)",
    )

    # ------------------------------------------------------------------
    # Local LLM (vLLM)
    # ------------------------------------------------------------------
    vllm_base_url: str = Field("http://vllm:8000/v1")
    vllm_model: str = Field("llama3")
    vllm_api_key: str = Field("not-needed", description="'not-needed' for local vLLM; set to openai_api_key when routing through OpenAI")
    vllm_temperature: float = Field(0.7)
    vllm_max_tokens: int = Field(1024)

    # ------------------------------------------------------------------
    # OpenAI (Judge + Embeddings)
    # ------------------------------------------------------------------
    openai_api_key: str = Field(..., description="OpenAI secret key")
    openai_primary_model: str = Field("gpt-4o-mini")
    openai_fallback_model: str = Field("gpt-4o")
    openai_embedding_model: str = Field("text-embedding-3-small")
    openai_embedding_dims: int = Field(1536)

    # ------------------------------------------------------------------
    # LlamaParse (Ingestor)
    # ------------------------------------------------------------------
    llama_parse_api_key: str = Field("", description="LlamaParse key (optional)")

    # ------------------------------------------------------------------
    # Pipeline behaviour
    # ------------------------------------------------------------------
    adversarial_ratio: float = Field(0.10, ge=0.0, le=1.0)
    quality_threshold: float = Field(0.90, ge=0.0, le=1.0)
    max_retries_per_chunk: int = Field(3, ge=1)
    max_refusal_ratio: float = Field(0.10, ge=0.0, le=1.0,
        description="Max fraction of 'Brak danych' records in output (0.10 = 10%)")
    chunk_overlap_chars: int = Field(150, ge=0)
    batch_size: int = Field(10, ge=1)
    watermark_interval: int = Field(50, ge=1)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_file: str = Field("/app/output/dataset_esg_v1.jsonl")
    client_id: str = Field("dev_client")

    # ------------------------------------------------------------------
    # Tenacity (Self-Check 2.0 — API rate-limit backoff)
    # ------------------------------------------------------------------
    tenacity_max_attempts: int = Field(6)
    tenacity_initial_wait: float = Field(2.0)    # seconds
    tenacity_max_wait: float = Field(64.0)       # seconds

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field("INFO")

    @field_validator("adversarial_ratio", "quality_threshold", "max_refusal_ratio", mode="before")
    @classmethod
    def parse_float(cls, v: object) -> float:
        return float(v)


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
