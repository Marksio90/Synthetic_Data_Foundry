"""
config/settings.py — Centralised, type-safe configuration via pydantic-settings.
All values come from environment variables (see .env.example).
"""

from __future__ import annotations

from typing import List

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
    database_url: str = Field(..., description="Sync SQLAlchemy URL (psycopg2)")
    async_database_url: str = Field(..., description="Async SQLAlchemy URL (asyncpg)")

    # ------------------------------------------------------------------
    # Ollama — LOCAL model (primary, zero cost)
    # ------------------------------------------------------------------
    ollama_model: str = Field(
        "",
        description="Model Ollama do generowania Q&A. Pusty = pomiń Ollama, użyj OpenAI fallback.",
    )
    ollama_embed_model: str = Field(
        "nomic-embed-text",
        description="Model Ollama do embeddingów (darmowy). Pusty = używaj OpenAI.",
    )
    use_local_embeddings: bool = Field(
        False,
        description="True = embeddingi Ollama (darmowe); False = OpenAI (wyższa jakość).",
    )

    # ------------------------------------------------------------------
    # Generation parameters (Ollama primary + OpenAI fallback)
    # ------------------------------------------------------------------
    generation_temperature: float = Field(0.7)
    generation_max_tokens: int = Field(2048, description="Max tokenów na odpowiedź eksperta")

    # ------------------------------------------------------------------
    # OpenAI (Judge + Embeddings)
    # ------------------------------------------------------------------
    openai_api_key: str = Field(..., description="OpenAI secret key")
    openai_primary_model: str = Field("gpt-4o-mini", description="Primary judge model")
    openai_fallback_model: str = Field(
        "gpt-4o",
        description="Fallback judge model (cascade gdy confidence < próg). ~5% wywołań.",
    )
    openai_embedding_model: str = Field("text-embedding-3-small")
    openai_embedding_dims: int = Field(1536)

    # ------------------------------------------------------------------
    # LlamaParse — najlepsza jakość dla złożonych PDF z tabelami
    # ------------------------------------------------------------------
    llama_parse_api_key: str = Field("", description="LlamaParse API key ($0.003/str)")

    # ------------------------------------------------------------------
    # Replicate — audio transcription via Whisper large-v3
    # ------------------------------------------------------------------
    replicate_api_key: str = Field(
        "",
        description="Replicate API key — transkrypcja audio (nagrania konferencji ESG → dane treningowe).",
    )

    # ------------------------------------------------------------------
    # HuggingFace Hub — auto-upload datasetu i modelu
    # ------------------------------------------------------------------
    hf_token: str = Field("", description="HuggingFace token (Write access)")
    hf_dataset_repo: str = Field(
        "",
        description="HF Hub repo do auto-uploadu datasetu (np. 'org/esg-foundry-dataset'). Pusty = pomiń.",
    )
    hf_model_repo: str = Field(
        "",
        description="HF Hub repo do uploadu wytrenowanego modelu. Pusty = pomiń.",
    )

    # ------------------------------------------------------------------
    # DeepL — tłumaczenie chunków
    # ------------------------------------------------------------------
    deepl_api_key: str = Field("", description="DeepL API key (opcjonalny)")
    translate_chunks: bool = Field(False)
    translate_source_lang: str = Field("en")

    # ------------------------------------------------------------------
    # Constitutional AI — self-critique + revision (darmowe pary DPO)
    # ------------------------------------------------------------------
    use_constitutional_ai: bool = Field(
        True,
        description=(
            "Włącza Constitutional AI: każda odpowiedź przechodzi przez "
            "self-critique + revision. Oryginalna odpowiedź staje się 'rejected' w DPO. "
            "Używa tego samego providera co generacja — zero dodatkowego kosztu."
        ),
    )
    constitutional_ai_threshold: float = Field(
        1.01,  # always revise by default (threshold > 1.0 = only below 1.0 scores)
        ge=0.0,
        le=1.01,
        description=(
            "Uruchom rewizję gdy quality_score < próg (0.0 = zawsze, 1.01 = zawsze). "
            "Domyślnie 1.01 = rewizja zawsze (maksymalna jakość)."
        ),
    )

    # ------------------------------------------------------------------
    # Pipeline perspectives — 8 ról eksperckich
    # ------------------------------------------------------------------
    perspectives: List[str] = Field(
        default=["cfo", "prawnik", "audytor", "analityk", "regulator", "akademik", "dziennikarz", "inwestor"],
        description=(
            "Lista perspektyw do generowania Q&A. Każda perspektywa = osobna rozmowa per chunk. "
            "8 perspektyw × N chunków = 8N rekordów. "
            "Skróć listę dla szybszych testów: PERSPECTIVES=[\"cfo\",\"prawnik\",\"audytor\"]"
        ),
    )

    # ------------------------------------------------------------------
    # Pipeline behaviour
    # ------------------------------------------------------------------
    adversarial_ratio: float = Field(0.10, ge=0.0, le=1.0)
    quality_threshold: float = Field(0.75, ge=0.0, le=1.0)
    max_retries_per_chunk: int = Field(3, ge=1)
    max_turns: int = Field(3, ge=1, le=5)
    max_refusal_ratio: float = Field(0.10, ge=0.0, le=1.0)
    chunk_overlap_chars: int = Field(150, ge=0)
    watermark_interval: int = Field(50, ge=1)
    batch_size: int = Field(10, ge=1)

    # ------------------------------------------------------------------
    # Output files — SFT + DPO + ORPO + KTO
    # ------------------------------------------------------------------
    output_file: str = Field("/app/output/dataset_esg_v1.jsonl", description="SFT ChatML")
    dpo_output_file: str = Field(
        "/app/output/dataset_esg_v1_dpo.jsonl",
        description="DPO preference pairs (TRL DPOTrainer)",
    )
    orpo_output_file: str = Field(
        "/app/output/dataset_esg_v1_orpo.jsonl",
        description="ORPO preference pairs (identyczny format co DPO, inny trainer)",
    )
    kto_output_file: str = Field(
        "/app/output/dataset_esg_v1_kto.jsonl",
        description="KTO pairs (Kahneman-Tversky Optimization — label: true/false)",
    )
    client_id: str = Field("dev_client", description="Unikalny ID klienta (B2B watermark)")

    # ------------------------------------------------------------------
    # Near-duplicate detection (MinHash LSH)
    # ------------------------------------------------------------------
    dedup_threshold: float = Field(0.85, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Cross-document synthesis
    # ------------------------------------------------------------------
    cross_doc_samples: int = Field(50, ge=0)

    # ------------------------------------------------------------------
    # Judge cascade
    # ------------------------------------------------------------------
    judge_confidence_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Próg pewności sędziego → eskalacja na model fallback. Różny od quality_threshold.",
    )

    # ------------------------------------------------------------------
    # Hybrid search weights
    # ------------------------------------------------------------------
    hybrid_vector_weight: float = Field(0.5, ge=0.0, le=1.0)
    hybrid_bm25_weight: float = Field(0.5, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Tenacity — API backoff
    # ------------------------------------------------------------------
    tenacity_max_attempts: int = Field(6)
    tenacity_initial_wait: float = Field(2.0)
    tenacity_max_wait: float = Field(64.0)

    # ------------------------------------------------------------------
    # Rate-limit throttling
    # ------------------------------------------------------------------
    chunk_delay_seconds: float = Field(0.0)

    # ------------------------------------------------------------------
    # Auto-calibration
    # ------------------------------------------------------------------
    calibration_samples: int = Field(50, ge=10, le=500)

    # ------------------------------------------------------------------
    # Gap Scout — expanded source crawler configuration
    # ------------------------------------------------------------------
    scout_sources_enabled: List[str] = Field(
        default_factory=list,
        description=(
            "Explicit allowlist of crawler IDs to activate. "
            "Empty list = all crawlers enabled. "
            "Example: [\"arxiv\",\"eurlex\",\"hackernews\",\"openalex\"]"
        ),
    )
    scout_min_gap_score: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Minimum KNOWLEDGE_GAP_SCORE for a topic to be surfaced in results.",
    )
    scout_model_targets: List[str] = Field(
        default_factory=lambda: ["gpt-4o", "claude-3.5-sonnet", "llama-3", "gemini-1.5", "mistral"],
        description="LLM identifiers used when computing cross-model divergence and cutoff targets.",
    )
    scout_languages: List[str] = Field(
        default_factory=lambda: ["en", "de", "fr", "ja", "zh", "ko", "ar", "ru", "pt", "es"],
        description="Active language codes for multilingual content scanning (BCP-47).",
    )
    scout_webhook_secret: str = Field(
        "",
        description="HMAC-SHA256 secret for verifying incoming WebSub/PubSubHubbub callbacks.",
    )
    scout_deepl_api_key: str = Field(
        "",
        description="DeepL API key for scout-specific translation (overrides deepl_api_key).",
    )
    scout_serpapi_key: str = Field(
        "",
        description="SerpAPI key used for niche_penetration scoring (Google Search results count).",
    )
    scout_max_concurrent_crawlers: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of source crawlers running concurrently via asyncio.",
    )
    # API keys for Layer A crawlers that require authentication
    ieee_api_key: str = Field(
        "",
        description="IEEE Xplore API key (free registration at developer.ieee.org).",
    )
    core_api_key: str = Field(
        "",
        description="CORE.ac.uk API key (free registration at core.ac.uk/services/api).",
    )

    # ------------------------------------------------------------------
    # API / UI service
    # ------------------------------------------------------------------
    data_dir: str = Field("/app/data")
    ollama_url: str = Field("http://localhost:11434")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field("INFO")

    # ------------------------------------------------------------------
    # Batch ID (overridable from CLI; also readable from env)
    # ------------------------------------------------------------------
    batch_id: str = Field("esg-production-v1")

    @field_validator(
        "adversarial_ratio", "quality_threshold", "max_refusal_ratio", "dedup_threshold",
        mode="before",
    )
    @classmethod
    def parse_float(cls, v: object) -> float:
        return float(v)

    @field_validator("perspectives", mode="before")
    @classmethod
    def parse_perspectives(cls, v: object) -> list:
        if isinstance(v, str):
            import json as _json
            v = v.strip()
            if v.startswith("["):
                return _json.loads(v)
            return [p.strip() for p in v.split(",") if p.strip()]
        return list(v)  # type: ignore[arg-type]


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
