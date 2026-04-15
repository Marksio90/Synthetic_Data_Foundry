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
    vllm_base_url: str = Field(
        "https://api.openai.com/v1",
        description=(
            "OpenAI-compatible endpoint. Opcje:\n"
            "  OpenAI:  https://api.openai.com/v1 (model: gpt-4o-mini)\n"
            "  Gemini:  https://generativelanguage.googleapis.com/v1beta/openai/ (model: gemini-2.0-flash-lite)\n"
            "  Together: https://api.together.xyz/v1 (model: meta-llama/Llama-3.1-8B-Instruct-Turbo)\n"
            "  Local:   http://ollama:11434/v1 (model: llama3.1:8b)"
        ),
    )
    vllm_model: str = Field("llama3")
    vllm_api_key: str = Field("not-needed", description="'not-needed' for local vLLM; set to openai_api_key when routing through OpenAI")
    vllm_temperature: float = Field(0.7)
    vllm_max_tokens: int = Field(1536,
        description="Max tokens for expert generation (increased to accommodate CoT reasoning block)")

    # ------------------------------------------------------------------
    # Groq (Llama 3.3 70B — primary question/answer generator)
    # ------------------------------------------------------------------
    groq_api_key: str = Field("", description="Groq API key — https://console.groq.com")
    groq_base_url: str = Field("https://api.groq.com/openai/v1")
    groq_model: str = Field("llama-3.3-70b-versatile")

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
    max_turns: int = Field(3, ge=1, le=5, description="Conversation turns per chunk")
    max_refusal_ratio: float = Field(0.10, ge=0.0, le=1.0,
        description="Max fraction of 'Brak danych' records in output (0.10 = 10%)")
    chunk_overlap_chars: int = Field(150, ge=0)
    batch_size: int = Field(10, ge=1)
    watermark_interval: int = Field(50, ge=1)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_file: str = Field("/app/output/dataset_esg_v1.jsonl")
    dpo_output_file: str = Field("/app/output/dataset_esg_v1_dpo.jsonl",
        description="DPO preference pairs (TRL DPOTrainer format)")
    client_id: str = Field("dev_client")

    # ------------------------------------------------------------------
    # Near-duplicate detection (MinHash)
    # ------------------------------------------------------------------
    dedup_threshold: float = Field(0.85, ge=0.0, le=1.0,
        description="Jaccard similarity threshold for duplicate question detection")

    # ------------------------------------------------------------------
    # Cross-document synthesis pass
    # ------------------------------------------------------------------
    cross_doc_samples: int = Field(50, ge=0,
        description="Number of cross-document Q&A pairs to generate after main pass (0 = skip)")

    # ------------------------------------------------------------------
    # Tenacity (Self-Check 2.0 — API rate-limit backoff)
    # ------------------------------------------------------------------
    tenacity_max_attempts: int = Field(6)
    tenacity_initial_wait: float = Field(2.0)    # seconds
    tenacity_max_wait: float = Field(64.0)       # seconds

    # ------------------------------------------------------------------
    # Auto-calibration
    # ------------------------------------------------------------------
    calibration_samples: int = Field(
        50, ge=10, le=500,
        description="Number of chunks to analyse for auto-calibrating quality_threshold",
    )
    calibration_target_accept_rate: float = Field(
        0.85, ge=0.5, le=1.0,
        description="Fraction of samples that should pass quality gate (calibrator targets this)",
    )

    # ------------------------------------------------------------------
    # Translation (non-PL source documents)
    # ------------------------------------------------------------------
    source_language: str = Field(
        "auto",
        description="Source document language ('auto'=detect, 'pl', 'en', 'de', 'fr')",
    )
    deepl_api_key: str = Field(
        "",
        description="DeepL API key for high-quality translation (optional; falls back to Groq)",
    )

    # ------------------------------------------------------------------
    # API / UI service
    # ------------------------------------------------------------------
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8080)
    data_dir: str = Field("/app/data", description="Directory where input PDFs are stored")
    ollama_url: str = Field(
        "http://localhost:11434",
        description="Ollama API endpoint (override to http://ollama:11434 in Docker)",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field("INFO")

    @field_validator("adversarial_ratio", "quality_threshold", "max_refusal_ratio", "dedup_threshold", mode="before")
    @classmethod
    def parse_float(cls, v: object) -> float:
        return float(v)


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
