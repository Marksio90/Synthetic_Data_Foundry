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
    # SECONDARY provider — cloud fallback gdy Ollama niedostępny
    #
    # WYBÓR: Cerebras (rekomendowany — 1M tokenów/dzień FREE, ultraszybki)
    #   Klucz: https://cloud.cerebras.ai → Sign up → API Keys
    #   Model: llama3.1-8b (identyczny z Ollama lokalnym = spójne wyniki)
    #   Prędkość: 2000+ tokenów/sek
    #
    # Alternatywy (zmień tylko 3 zmienne w .env):
    #   OpenRouter free:  openrouter.ai/api/v1  | meta-llama/llama-3.1-8b-instruct:free
    #   Together AI paid: api.together.xyz/v1   | meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo ($0.18/1M)
    # ------------------------------------------------------------------
    secondary_api_key: str = Field(
        "",
        description="Klucz do secondary providera (SECONDARY_API_KEY). Cerebras: https://cloud.cerebras.ai",
    )
    secondary_base_url: str = Field(
        "https://api.cerebras.ai/v1",
        description="Base URL secondary providera. Default: Cerebras.",
    )
    secondary_model: str = Field(
        "llama3.1-8b",
        description=(
            "Model secondary providera. "
            "Cerebras: llama3.1-8b lub llama3.3-70b. "
            "OpenRouter free: meta-llama/llama-3.1-8b-instruct:free. "
            "Together: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo."
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
        "gpt-4o",
        description=(
            "Fallback sędziego gdy pewność < próg jakości (~5% przypadków). "
            "gpt-4o: 15× droższy od mini, ale wywołany rzadko — wart dla jakości datasetu. "
            "Ustaw gpt-4o-mini aby wyłączyć kaskadę i oszczędzać."
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
            "OpenAI/Ollama: ustaw 0. "
            "Cerebras/OpenRouter free tier: ustaw 1-3 w razie 429."
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
        description="DeepL API key for high-quality translation (optional; falls back to secondary/OpenAI LLM)",
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
