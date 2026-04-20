"""
api/schemas.py — Pydantic request/response models for the Foundry API.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    uploaded_at: str
    in_db: bool = False
    chunk_count: int = 0
    sample_count: int = 0


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


# ---------------------------------------------------------------------------
# Analysis (DocAnalyzer result)
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    language: str
    translation_required: bool
    domain: str
    domain_label: str
    perspectives: list[str]
    domain_confidence: float
    total_chars: int
    auto_decisions: list[str]
    calibration: Optional[CalibrationInfo] = None


class CalibrationInfo(BaseModel):
    quality_threshold: float
    max_turns: int
    adversarial_ratio: float
    n_chunks: int
    avg_chunk_len: float
    vocab_richness: float
    info_density: float
    reasoning: list[str]


# ---------------------------------------------------------------------------
# Pipeline run
# ---------------------------------------------------------------------------

class PipelineRunRequest(BaseModel):
    filenames: list[str] = Field(..., description="Filenames in data/ directory to process")
    batch_id: Optional[str] = Field(None, description="Custom batch ID (auto-generated if empty)")
    chunk_limit: int = Field(0, ge=0, description="Max chunks to process (0 = unlimited)")
    # AutoPilot overrides — if None, auto-calibrated values are used
    quality_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_turns: Optional[int] = Field(None, ge=1, le=5)
    adversarial_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    perspectives: Optional[list[str]] = Field(None)


class PipelineRunResponse(BaseModel):
    run_id: str
    batch_id: str
    status: str
    message: str


class PipelineStatusResponse(BaseModel):
    run_id: str
    batch_id: str
    status: str              # starting | running | done | error
    progress_pct: int        # 0–100
    chunks_total: int
    chunks_done: int
    records_written: int
    dpo_pairs: int
    elapsed_seconds: float
    analysis: Optional[dict[str, Any]] = None
    calibration: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class LogLinesResponse(BaseModel):
    run_id: str
    lines: list[str]
    total_lines: int
