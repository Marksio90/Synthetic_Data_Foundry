"""
api/routers/samples.py — Generated Q&A sample browsing (Sprint 1 basic version).

Endpoints:
  GET  /api/samples          Paginated list of generated samples with filters
  GET  /api/samples/stats    Dataset statistics
  GET  /api/samples/{id}     Single sample detail
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.db import get_session
from db.models import DirectiveChunk, GeneratedSample, SourceDocument

router = APIRouter()


@router.get("")
def list_samples(
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = 50,
    perspective: Optional[str] = None,
    difficulty: Optional[str] = None,
    min_score: Optional[float] = None,
    batch_id: Optional[str] = None,
) -> dict:
    """Return paginated samples with optional filters."""
    q = select(GeneratedSample).order_by(GeneratedSample.created_at.desc())

    if perspective:
        q = q.where(GeneratedSample.perspective == perspective)
    if difficulty:
        q = q.where(GeneratedSample.difficulty == difficulty)
    if min_score is not None:
        q = q.where(GeneratedSample.quality_score >= min_score)
    if batch_id:
        q = q.where(GeneratedSample.batch_id == batch_id)

    total = session.scalar(select(func.count()).select_from(q.subquery())) or 0
    samples = session.scalars(q.offset(offset).limit(limit)).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "samples": [
            {
                "id": str(s.id),
                "question": s.question[:200],
                "answer": s.answer[:300],
                "perspective": s.perspective,
                "difficulty": s.difficulty,
                "question_type": s.question_type,
                "quality_score": s.quality_score,
                "is_adversarial": s.is_adversarial,
                "batch_id": s.batch_id,
                "has_dpo": bool(s.rejected_answer),
                "turn_count": len(s.conversation_json) // 2 if s.conversation_json else 1,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in samples
        ],
    }


@router.get("/stats")
def dataset_stats(session: Session = Depends(get_session)) -> dict:
    """Return aggregated dataset statistics."""
    total = session.scalar(select(func.count(GeneratedSample.id))) or 0
    if total == 0:
        return {"total": 0}

    avg_score = session.scalar(
        select(func.avg(GeneratedSample.quality_score))
    ) or 0.0

    dpo_count = session.scalar(
        select(func.count(GeneratedSample.id)).where(
            GeneratedSample.rejected_answer.isnot(None)
        )
    ) or 0

    # Per-perspective breakdown
    persp_rows = session.execute(
        select(GeneratedSample.perspective, func.count(GeneratedSample.id))
        .group_by(GeneratedSample.perspective)
    ).all()

    # Per-difficulty breakdown
    diff_rows = session.execute(
        select(GeneratedSample.difficulty, func.count(GeneratedSample.id))
        .group_by(GeneratedSample.difficulty)
    ).all()

    return {
        "total": total,
        "avg_quality_score": round(float(avg_score), 3),
        "dpo_pairs": dpo_count,
        "perspectives": {row[0] or "unknown": row[1] for row in persp_rows},
        "difficulties": {row[0] or "unknown": row[1] for row in diff_rows},
    }


@router.get("/{sample_id}")
def get_sample(sample_id: str, session: Session = Depends(get_session)) -> dict:
    """Return full detail for one sample (including full conversation JSON)."""
    import uuid as _uuid
    try:
        sid = _uuid.UUID(sample_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")

    sample = session.scalar(select(GeneratedSample).where(GeneratedSample.id == sid))
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return {
        "id": str(sample.id),
        "question": sample.question,
        "answer": sample.answer,
        "rejected_answer": sample.rejected_answer,
        "perspective": sample.perspective,
        "difficulty": sample.difficulty,
        "question_type": sample.question_type,
        "quality_score": sample.quality_score,
        "judge_model": sample.judge_model,
        "judge_reasoning": sample.judge_reasoning,
        "is_adversarial": sample.is_adversarial,
        "batch_id": sample.batch_id,
        "record_index": sample.record_index,
        "conversation_json": sample.conversation_json,
        "created_at": sample.created_at.isoformat() if sample.created_at else None,
    }
