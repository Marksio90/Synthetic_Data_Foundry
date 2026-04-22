"""
utils/cost_tracker.py — LLM cost recording to workflow_cost_ledger.

Usage:
    from utils.cost_tracker import record_cost

    record_cost(
        workflow_id=batch_id,
        agent_name="expert",
        model_name="gpt-4o-mini",
        model_tier="quality",
        prompt_tokens=500,
        completion_tokens=200,
        cost_usd=0.000123,
        chunk_id=chunk_id,
        perspective="cfo",
    )
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

logger = logging.getLogger("foundry.cost_tracker")

# In-memory accumulator for batch writes (flush every N calls)
_BATCH_SIZE = 20
_pending: list[dict] = []


def record_cost(
    workflow_id: str,
    agent_name: str,
    model_name: str,
    model_tier: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    chunk_id: Optional[str] = None,
    perspective: Optional[str] = None,
) -> None:
    """
    Record an LLM cost event. Batches writes to PostgreSQL.
    Thread-safe via append-only list (CPython GIL protects list.append).
    """
    _pending.append({
        "workflow_id": workflow_id,
        "agent_name": agent_name,
        "model_name": model_name,
        "model_tier": model_tier,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost_usd,
        "chunk_id": chunk_id,
        "perspective": perspective,
    })

    if len(_pending) >= _BATCH_SIZE:
        _flush_sync()


def _flush_sync() -> None:
    """Write accumulated cost records to PostgreSQL."""
    global _pending
    if not _pending:
        return
    batch = _pending[:]
    _pending = []
    try:
        from sqlalchemy import create_engine, text
        from config.settings import settings

        engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=1)
        with engine.begin() as conn:
            for row in batch:
                conn.execute(
                    text(
                        """
                        INSERT INTO workflow_cost_ledger
                          (workflow_id, agent_name, model_name, model_tier,
                           prompt_tokens, completion_tokens, cost_usd, chunk_id, perspective)
                        VALUES
                          (:workflow_id, :agent_name, :model_name, :model_tier,
                           :prompt_tokens, :completion_tokens, :cost_usd,
                           :chunk_id::uuid, :perspective)
                        """
                    ),
                    {
                        **row,
                        "chunk_id": row["chunk_id"] if row["chunk_id"] else None,
                    },
                )
        engine.dispose()
    except Exception as exc:
        logger.warning("Cost tracker flush failed: %s — %d records dropped.", exc, len(batch))


def flush() -> None:
    """Force-flush pending cost records (call at end of pipeline run)."""
    _flush_sync()


async def get_workflow_cost(workflow_id: str) -> dict:
    """Return cost summary for a workflow_id."""
    try:
        from sqlalchemy import create_engine, text
        from config.settings import settings

        engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=1)
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        SUM(cost_usd)                           AS total_usd,
                        SUM(prompt_tokens + completion_tokens)  AS total_tokens,
                        COUNT(*)                                AS call_count,
                        SUM(cost_usd) FILTER (WHERE model_tier = 'local')   AS local_usd,
                        SUM(cost_usd) FILTER (WHERE model_tier = 'quality') AS quality_usd,
                        SUM(cost_usd) FILTER (WHERE model_tier = 'judge')   AS judge_usd
                    FROM workflow_cost_ledger
                    WHERE workflow_id = :wf_id
                    """
                ),
                {"wf_id": workflow_id},
            ).fetchone()
        engine.dispose()
        if row is None:
            return {}
        return {
            "workflow_id": workflow_id,
            "total_cost_usd": float(row[0] or 0),
            "total_tokens": int(row[1] or 0),
            "call_count": int(row[2] or 0),
            "local_cost_usd": float(row[3] or 0),
            "quality_cost_usd": float(row[4] or 0),
            "judge_cost_usd": float(row[5] or 0),
        }
    except Exception as exc:
        logger.error("get_workflow_cost failed for %s: %s", workflow_id, exc)
        return {}
