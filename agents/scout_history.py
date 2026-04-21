"""
agents/scout_history.py — Persistent domain history for Gap Scout.

Tracks which domains have been scanned (to avoid repetition across restarts)
and which topics the user has ingested (permanently excluded from future scans).

All public functions are async via asyncio.to_thread() wrapping sync SQLAlchemy,
so they integrate cleanly with the async topic_scout pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from config.settings import settings

logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_size=2,
            max_overflow=3,
            pool_pre_ping=True,
        )
    return _engine


# ---------------------------------------------------------------------------
# Sync helpers (run inside asyncio.to_thread)
# ---------------------------------------------------------------------------

def _sync_get_excluded_domains() -> set[str]:
    try:
        with _get_engine().connect() as conn:
            rows = conn.execute(
                text("SELECT domain_text FROM scout_domain_exclusions")
            ).fetchall()
            return {r[0] for r in rows}
    except Exception as exc:
        logger.warning("scout_history: cannot load exclusions: %s", exc)
        return set()


def _sync_get_recent_domains(window: int) -> list[str]:
    try:
        with _get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT domain_text FROM scout_domain_history "
                    "ORDER BY selected_at DESC LIMIT :n"
                ),
                {"n": window},
            ).fetchall()
            return [r[0] for r in rows]
    except Exception as exc:
        logger.warning("scout_history: cannot load recent domains: %s", exc)
        return []


def _sync_record_domains(domains: list[str], run_id: str) -> None:
    try:
        with _get_engine().begin() as conn:
            for domain in domains:
                conn.execute(
                    text(
                        "INSERT INTO scout_domain_history (domain_text, run_id, selected_at) "
                        "VALUES (:d, :r, :t)"
                    ),
                    {"d": domain, "r": run_id, "t": datetime.now(tz=timezone.utc)},
                )
    except Exception as exc:
        logger.warning("scout_history: cannot record domains: %s", exc)


def _sync_exclude_domain(topic_id: str, domain_text: str, topic_title: str) -> None:
    try:
        with _get_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO scout_domain_exclusions "
                    "(topic_id, domain_text, topic_title, excluded_at) "
                    "VALUES (:tid, :d, :title, :t) "
                    "ON CONFLICT (topic_id) DO NOTHING"
                ),
                {
                    "tid": topic_id,
                    "d": domain_text,
                    "title": topic_title,
                    "t": datetime.now(tz=timezone.utc),
                },
            )
    except Exception as exc:
        logger.warning("scout_history: cannot exclude domain: %s", exc)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def get_excluded_domains() -> set[str]:
    """Domains permanently excluded because the user selected them for ingestion."""
    return await asyncio.to_thread(_sync_get_excluded_domains)


async def get_recent_domains(window: int = 40) -> list[str]:
    """Most recently scanned domain texts (newest first, up to `window` entries)."""
    return await asyncio.to_thread(_sync_get_recent_domains, window)


async def record_selected_domains(domains: list[str], run_id: str = "") -> None:
    """Persist domains chosen for this scout run so future runs avoid them."""
    await asyncio.to_thread(_sync_record_domains, domains, run_id)


async def exclude_domain_permanently(
    topic_id: str,
    domain_text: str,
    topic_title: str = "",
) -> None:
    """Permanently exclude a domain — user selected this topic for work."""
    await asyncio.to_thread(_sync_exclude_domain, topic_id, domain_text, topic_title)
