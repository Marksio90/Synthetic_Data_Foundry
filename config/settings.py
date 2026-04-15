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
    # Ollama — LOCAL model (primary, zero cost)
    # Działa na 32 GB RAM. Zalecane modele:
    #   llama3.2        (~2 GB)  — szybki, dobry do Q&A
    #   llama3.1:8b     (~4.7 GB) — wyższa jakość
    #   mistral         (~4.1 GB) — dobry do długich dokumentów
    # Uruchom: ollama pull llama3.1:8b
    # ------------------------------------------------------------------
    ollama_model: str = Field(
        "llama3.1:8b",
        description=(
            "Model Ollama do generowania Q&A (PRIMARY, darmowy, lokalny). "
            "Ustaw pusty string '' aby pominąć Ollama i przejść do LLaMA API."
        ),
    )
    ollama_embed_model: str = Field(
        "nomic-embed-text",
        description=(
            "Model Ollama do embeddingów (darmowy, 0.3 GB RAM). "
            "Ustaw '' aby używać OpenAI text-embedding-3-small."
        ),
    )
    use_local_embeddings: bool = Field(
        False,
        description=(
            "Gdy True: embeddingi przez Ollama nomic-embed-text (darmowe, lokalne). "
            "Gdy False: embeddingi przez OpenAI (płatne, wyższa jakość)."
        ),
    )

    # ------------------------------------------------------------------
    # LLaMA API — cloud LLaMA (secondary, tani)
    # Groq: https://console.groq.com — do 200k TPM gratis (GROQ_API_KEY)
    # Together: https://api.together.xyz — alternatywa
    # ------------------------------------------------------------------
    groq_api_key: str = Field(
        "",
        description=(
            "Klucz do LLaMA API (Groq lub Together AI). "
            "Groq: https://console.groq.com (darmowe 200k TPM). "
            "Używany jako SECONDARY po Ollama, przed OpenAI."
        ),
    )
    groq_base_url: str = Field(
        "https://api.groq.com/openai/v1",
        description="Base URL dla LLaMA API. Groq: https://api.groq.com/openai/v1",
    )
    groq_model: str = Field(
        "llama-3.1-8b-instant",
        description=(
            "Model LLaMA API. "
            "Groq llama-3.1-8b-instant: 200k TPM free — zalecany. "
            "Groq llama-3.3-70b-versatile: lepsza jakość (12k TPM free). "
            "Together meta-llama/Llama-3.1-8B-Instruct-Turbo: alternatywa."
        ),
    )

    # ------------------------------------------------------------------
    # Fallback endpoint (OpenAI-compatible, np. vLLM)
    # ------------------------------------------------------------------
    vllm_base_url: str = Field(
        "https://api.openai.com/v1",
        description=(
            "OpenAI-compatible endpoint (fallback gdy Ollama i LLaMA API niedostępne).\n"
            "  OpenAI:   https://api.openai.com/v1\n"
            "  Together: https://api.together.xyz/v1\n"
            "  Local:    http://ollama:11434/v1"
        ),
    )
    vllm_model: str = Field("gpt-4o-mini")
    vllm_api_key: str = Field("not-needed", description="'not-needed' dla lokalnego; ustaw openai_api_key gdy przez OpenAI")
    vllm_temperature: float = Field(0.7)
    vllm_max_tokens: int = Field(2048,
        description="Max tokenów dla eksperta (zwiększone dla lepszego CoT)")

    # ------------------------------------------------------------------
    # OpenAI (Judge + Embeddings)
    # ------------------------------------------------------------------
    openai_api_key: str = Field(..., description="OpenAI secret key")
    openai_primary_model: str = Field("gpt-4o-mini")
    openai_fallback_model: str = Field(
        "gpt-4o-mini",
        description=(
            "Fallback sędziego gdy pewność < próg. "
            "gpt-4o-mini: ~15× taniej niż gpt-4o, wystarczający dla ewaluacji. "
            "Zmień na gpt-4o tylko jeśli potrzebujesz najwyższej jakości ocen."
        ),
    )
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
    # Rate-limit throttling
    # ------------------------------------------------------------------
    chunk_delay_seconds: float = Field(
        0.0,
        description=(
            "Opóźnienie między chunkami (sekundy). "
            "OpenAI gpt-4o-mini: 10M TPM — ustaw 0. "
            "Groq free tier: ustaw 1-5 w zależności od modelu."
        ),
    )

    # ------------------------------------------------------------------
    # Auto-calibration
    # ------------------------------------------------------------------
    calibration_samples: int = Field(
        50, ge=10, le=500,
        description="Number of chunks to analyse for auto-calibrating quality_threshold",
    )
    deepl_api_key: str = Field(
        "",
        description="DeepL API key for high-quality translation (optional; falls back to Groq)",
    )

    # ------------------------------------------------------------------
    # API / UI service
    # ------------------------------------------------------------------
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
