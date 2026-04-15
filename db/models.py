"""
db/models.py — SQLAlchemy ORM models mirroring the PostgreSQL schema.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    # Allow legacy Column()-style annotations (no Mapped[] wrapper required)
    __allow_unmapped__ = True


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: str = Column(Text, nullable=False, unique=True)
    file_hash: str = Column(String(64), nullable=False)
    directive_name: Optional[str] = Column(Text)
    directive_year: Optional[int] = Column(SmallInteger)
    valid_from_date: Optional[date] = Column(Date)
    ingested_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    raw_markdown: Optional[str] = Column(Text)

    chunks: list[DirectiveChunk] = relationship(
        "DirectiveChunk", back_populates="source_doc", cascade="all, delete-orphan"
    )


class DirectiveChunk(Base):
    __tablename__ = "directive_chunks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('new','in_progress','ready','unresolvable')",
            name="chk_chunk_status",
        ),
    )

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_doc_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: int = Column(Integer, nullable=False)
    content: str = Column(Text, nullable=False)
    content_md: Optional[str] = Column(Text)
    embedding: Optional[list[float]] = Column(Vector(1536))
    # Self-Check 3.0 — legal validity columns
    valid_from_date: Optional[date] = Column(Date)
    is_superseded: bool = Column(Boolean, nullable=False, default=False)
    superseded_by: Optional[uuid.UUID] = Column(
        UUID(as_uuid=True), ForeignKey("directive_chunks.id"), nullable=True
    )
    section_heading: Optional[str] = Column(Text)
    # Full-text search vector (populated by trg_chunk_fts trigger)
    fts_vector: Optional[str] = Column(TSVECTOR, nullable=True)
    # ACID status (Self-Check idempotency)
    status: str = Column(Text, nullable=False, default="new")
    retry_count: int = Column(SmallInteger, nullable=False, default=0)
    error_log: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), server_default=func.now())

    source_doc: SourceDocument = relationship("SourceDocument", back_populates="chunks")
    samples: list[GeneratedSample] = relationship(
        "GeneratedSample", back_populates="chunk", cascade="all, delete-orphan"
    )


class GeneratedSample(Base):
    __tablename__ = "generated_samples"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        ForeignKey("directive_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    question: str = Column(Text, nullable=False)
    answer: str = Column(Text, nullable=False)
    system_prompt: str = Column(Text, nullable=False)
    is_adversarial: bool = Column(Boolean, nullable=False, default=False)
    quality_score: Optional[float] = Column(Float)
    judge_model: Optional[str] = Column(Text)
    judge_reasoning: Optional[str] = Column(Text)
    # Dataset labelling and quality metadata
    perspective: Optional[str] = Column(Text)          # "cfo" | "prawnik" | "audytor" | "cross_doc"
    question_type: Optional[str] = Column(Text)        # factual | scope | process | compliance | comparative
    difficulty: Optional[str] = Column(Text)           # easy | medium | hard
    rejected_answer: Optional[str] = Column(Text)      # first-attempt answer for DPO pairing
    # Full multi-turn conversation (mirrors JSONL record for audit / analysis)
    conversation_json: Optional[list] = Column(JSONB)
    # Human review (Sprint 2 — Auto-Reviewer)
    human_reviewed: Optional[bool] = Column(Boolean, nullable=True, default=None)
    human_flag: Optional[str] = Column(Text, nullable=True)  # auto_approved|auto_rejected|human_approved|human_rejected

    # Watermark (Self-Check B2B)
    watermark_hash: Optional[str] = Column(String(64))
    batch_id: Optional[str] = Column(Text)
    record_index: Optional[int] = Column(Integer)
    written_to_file: bool = Column(Boolean, nullable=False, default=False)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())

    chunk: DirectiveChunk = relationship("DirectiveChunk", back_populates="samples")


class WatermarkRegistry(Base):
    __tablename__ = "watermark_registry"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id: str = Column(Text, nullable=False, unique=True)
    client_id: str = Column(Text, nullable=False)
    watermark_signature: str = Column(Text, nullable=False)
    watermark_hash: str = Column(String(64), nullable=False)
    record_indices: list[int] = Column(ARRAY(Integer))
    total_records: Optional[int] = Column(Integer)
    created_at: datetime = Column(DateTime(timezone=True), server_default=func.now())


class OpenAIBatchJob(Base):
    __tablename__ = "openai_batch_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('submitted','in_progress','completed','failed')",
            name="chk_batch_status",
        ),
    )

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_job_id: str = Column(Text, nullable=False, unique=True)
    status: str = Column(Text, nullable=False, default="submitted")
    input_file_id: Optional[str] = Column(Text)
    output_file_id: Optional[str] = Column(Text)
    sample_ids: list[uuid.UUID] = Column(ARRAY(UUID(as_uuid=True)))
    submitted_at: datetime = Column(DateTime(timezone=True), server_default=func.now())
    completed_at: Optional[datetime] = Column(DateTime(timezone=True))
    error_detail: Optional[str] = Column(Text)
