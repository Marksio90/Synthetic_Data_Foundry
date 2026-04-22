"""
api/state.py — Shared in-memory run state for the Foundry API.

One RunManager singleton tracks all pipeline runs in the current process.
Each run stores: status, log lines, progress counters, analysis/calibration results.

Thread-safe for use from asyncio background tasks.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field  # noqa: F401 — field used by ScoutTopic
from typing import Any, Optional

from config.settings import settings

_MAX_LOG_LINES = settings.state_max_log_lines
_MAX_RUNS = settings.state_max_runs
_MAX_SCOUT_RUNS = settings.state_max_scout_runs


@dataclass
class RunRecord:
    run_id: str
    batch_id: str
    status: str = "starting"          # starting | running | done | error
    progress_pct: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    records_written: int = 0
    dpo_pairs: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)
    analysis: Optional[dict[str, Any]] = None
    calibration: Optional[dict[str, Any]] = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.ended_at or time.time()
        return round(end - self.started_at, 1)


class RunManager:
    """
    Simple in-memory store for pipeline run records.
    Designed for single-process use (one Uvicorn worker).
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def create(self, run_id: str, batch_id: str) -> RunRecord:
        if len(self._runs) >= _MAX_RUNS:
            oldest_run_id = next(iter(self._runs))
            self._runs.pop(oldest_run_id, None)
        rec = RunRecord(run_id=run_id, batch_id=batch_id)
        self._runs[run_id] = rec
        return rec

    def get(self, run_id: str) -> Optional[RunRecord]:
        return self._runs.get(run_id)

    def update(self, run_id: str, **kwargs: Any) -> None:
        rec = self._runs.get(run_id)
        if rec is None:
            return
        for k, v in kwargs.items():
            if hasattr(rec, k):
                setattr(rec, k, v)

    def append_log(self, run_id: str, line: str) -> None:
        rec = self._runs.get(run_id)
        if rec is not None:
            rec.log_lines.append(line)
            if len(rec.log_lines) > _MAX_LOG_LINES:
                del rec.log_lines[: len(rec.log_lines) - _MAX_LOG_LINES]
            # Parse progress hints from log lines
            _parse_progress(rec, line)

    def list_runs(self) -> list[RunRecord]:
        return list(self._runs.values())


# ---------------------------------------------------------------------------
# Log-line parser — extracts structured progress from main.py stdout
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(
    r"Progress:\s*(\d+)\s*chunks\s*\|\s*(\d+)\s*ready\s*\|\s*(\d+)\s*unresolvable"
    r"\s*\|\s*(\d+)\s*records\s*\|\s*(\d+)\s*DPO"
)
_RECORD_RE = re.compile(r"✓ Record (\d+) written")
_FOUND_RE = re.compile(r"Found (\d+) pending chunks")
_DONE_RE = re.compile(r"ESG Data Foundry — Run Complete")
_ERROR_RE = re.compile(r"\[ERROR\]")


def _parse_progress(rec: RunRecord, line: str) -> None:
    m = _PROGRESS_RE.search(line)
    if m:
        rec.chunks_done = int(m.group(1))
        rec.records_written = int(m.group(4))
        rec.dpo_pairs = int(m.group(5))
        if rec.chunks_total > 0:
            rec.progress_pct = min(int(rec.chunks_done / rec.chunks_total * 100), 99)
        return

    m = _FOUND_RE.search(line)
    if m:
        rec.chunks_total = int(m.group(1))
        rec.status = "running"
        return

    m = _RECORD_RE.search(line)
    if m:
        rec.records_written = int(m.group(1)) + 1
        return

    if _DONE_RE.search(line):
        rec.status = "done"
        rec.progress_pct = 100
        rec.ended_at = time.time()
        return

    # _ERROR_RE matches are non-fatal — only the subprocess exit code matters


# Singleton imported by routers
runs = RunManager()


# ===========================================================================
# Gap Scout — in-memory state
# ===========================================================================


@dataclass
class ScoutTopic:
    topic_id: str
    title: str
    summary: str
    score: float
    recency_score: float
    llm_uncertainty: float
    source_count: int
    social_signal: float
    sources: list[dict]          # serialised ScoutSource dicts
    domains: list[str]
    discovered_at: str           # ISO-8601
    # --- new fields (backwards-compatible, all have defaults) ---
    knowledge_gap_score: float = 0.0
    cutoff_model_targets: list = field(default_factory=list)
    format_types: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    citation_velocity: float = 0.0
    source_tier: str = "C"
    estimated_tokens: int = 0
    ingest_ready: bool = False
    dataset_category: str = "general"
    dataset_purpose: str = "qa_reasoning"
    demand_score: float = 0.0
    uniqueness_score: float = 0.0
    quality_score: float = 0.0
    quality_gate_passed: bool = False
    quality_gate_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "summary": self.summary,
            "score": self.score,
            "recency_score": self.recency_score,
            "llm_uncertainty": self.llm_uncertainty,
            "source_count": self.source_count,
            "social_signal": self.social_signal,
            "sources": self.sources,
            "domains": self.domains,
            "discovered_at": self.discovered_at,
            # new fields
            "knowledge_gap_score": self.knowledge_gap_score,
            "cutoff_model_targets": self.cutoff_model_targets,
            "format_types": self.format_types,
            "languages": self.languages,
            "citation_velocity": self.citation_velocity,
            "source_tier": self.source_tier,
            "estimated_tokens": self.estimated_tokens,
            "ingest_ready": self.ingest_ready,
            "dataset_category": self.dataset_category,
            "dataset_purpose": self.dataset_purpose,
            "demand_score": self.demand_score,
            "uniqueness_score": self.uniqueness_score,
            "quality_score": self.quality_score,
            "quality_gate_passed": self.quality_gate_passed,
            "quality_gate_reasons": self.quality_gate_reasons,
        }


@dataclass
class FeedbackRecord:
    topic_id: str
    rating: int       # 1-5
    helpful: bool
    comment: str
    submitted_at: str


@dataclass
class ScoutRecord:
    scout_id: str
    status: str = "starting"     # starting | running | done | error
    topics_found: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)
    topics: list[ScoutTopic] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        end = self.ended_at or time.time()
        return round(end - self.started_at, 1)


class ScoutManager:
    """In-memory store for scout runs and discovered topics."""

    def __init__(self) -> None:
        self._runs: dict[str, ScoutRecord] = {}
        self._topics: dict[str, ScoutTopic] = {}   # topic_id → latest version
        self._sse_queues: set = set()
        self._feedback: list = []

    def create(self, scout_id: str) -> ScoutRecord:
        if len(self._runs) >= _MAX_SCOUT_RUNS:
            oldest_scout_id = next(iter(self._runs))
            self._runs.pop(oldest_scout_id, None)
        rec = ScoutRecord(scout_id=scout_id)
        self._runs[scout_id] = rec
        return rec

    def get(self, scout_id: str) -> Optional[ScoutRecord]:
        return self._runs.get(scout_id)

    def update(self, scout_id: str, **kwargs: Any) -> None:
        rec = self._runs.get(scout_id)
        if rec is None:
            return
        for k, v in kwargs.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
        if kwargs.get("status") in ("done", "error") and rec.ended_at is None:
            rec.ended_at = time.time()

    def append_log(self, scout_id: str, line: str) -> None:
        rec = self._runs.get(scout_id)
        if rec is not None:
            rec.log_lines.append(line)
            if len(rec.log_lines) > _MAX_LOG_LINES:
                del rec.log_lines[: len(rec.log_lines) - _MAX_LOG_LINES]

    def _make_scout_topic(self, t: Any) -> "ScoutTopic":
        return ScoutTopic(
            topic_id=t.topic_id,
            title=t.title,
            summary=t.summary,
            score=t.score,
            recency_score=t.recency_score,
            llm_uncertainty=t.llm_uncertainty,
            source_count=t.source_count,
            social_signal=t.social_signal,
            sources=[
                {
                    "url": s.url,
                    "title": s.title,
                    "published_at": s.published_at,
                    "source_type": s.source_type,
                    "verified": s.verified,
                    "source_tier": getattr(s, "source_tier", "C"),
                    "language": getattr(s, "language", ""),
                    "snippet": getattr(s, "snippet", ""),
                }
                for s in t.sources
            ],
            domains=t.domains,
            discovered_at=t.discovered_at,
            # new fields — getattr for backwards compat with older ScoutTopicData instances
            knowledge_gap_score=getattr(t, "knowledge_gap_score", t.score),
            cutoff_model_targets=getattr(t, "cutoff_model_targets", []),
            format_types=getattr(t, "format_types", []),
            languages=getattr(t, "languages", []),
            citation_velocity=getattr(t, "citation_velocity", 0.0),
            source_tier=getattr(t, "source_tier", "C"),
            estimated_tokens=getattr(t, "estimated_tokens", 0),
            ingest_ready=getattr(t, "ingest_ready", False),
            dataset_category=getattr(t, "dataset_category", "general"),
            dataset_purpose=getattr(t, "dataset_purpose", "qa_reasoning"),
            demand_score=getattr(t, "demand_score", 0.0),
            uniqueness_score=getattr(t, "uniqueness_score", 0.0),
            quality_score=getattr(t, "quality_score", 0.0),
            quality_gate_passed=getattr(t, "quality_gate_passed", False),
            quality_gate_reasons=getattr(t, "quality_gate_reasons", []),
        )

    def add_single_topic(self, scout_id: str, topic_data: Any) -> None:
        """Add one topic immediately when discovered (progressive streaming)."""
        rec = self._runs.get(scout_id)
        if rec is None:
            return
        if topic_data.topic_id in self._topics:
            return  # already added
        topic = self._make_scout_topic(topic_data)
        rec.topics.append(topic)
        self._topics[topic.topic_id] = topic
        rec.topics_found = len(rec.topics)
        self.broadcast_event({"event": "topic", "data": topic.to_dict()})
        try:
            from api.monitoring import topics_discovered_total, scout_topics_per_source
            topics_discovered_total.inc()
            for src_dict in topic.sources[:1]:
                scout_topics_per_source.labels(source=src_dict.get("source_type", "unknown")).inc()
        except Exception:
            pass

    def add_topics(self, scout_id: str, topic_data_list: list) -> None:
        """Accept list[ScoutTopicData] from agents/topic_scout.py and persist.
        Idempotent: skips topics already added via add_single_topic."""
        rec = self._runs.get(scout_id)
        if rec is None:
            return
        self._evict_old_topics()
        existing_ids = {t.topic_id for t in rec.topics}
        for t in topic_data_list:
            if t.topic_id in existing_ids:
                continue
            topic = self._make_scout_topic(t)
            rec.topics.append(topic)
            self._topics[topic.topic_id] = topic
        rec.topics_found = len(rec.topics)

    def _evict_old_topics(self, ttl_hours: int = 72) -> None:
        """Remove topics older than ttl_hours to prevent unbounded memory growth."""
        import datetime as _dt
        cutoff = _dt.datetime.now(tz=_dt.timezone.utc)
        to_remove = []
        for tid, topic in self._topics.items():
            try:
                age_h = (cutoff - _dt.datetime.fromisoformat(topic.discovered_at)).total_seconds() / 3600
                if age_h > ttl_hours:
                    to_remove.append(tid)
            except Exception:
                pass
        for tid in to_remove:
            self._topics.pop(tid, None)

    def get_topic(self, topic_id: str) -> Optional[ScoutTopic]:
        return self._topics.get(topic_id)

    def latest_topics(self, limit: int = 50) -> list[ScoutTopic]:
        topics = sorted(
            self._topics.values(),
            key=lambda t: (
                0.45 * t.quality_score
                + 0.25 * t.uniqueness_score
                + 0.20 * t.knowledge_gap_score
                + 0.10 * t.demand_score
            ),
            reverse=True,
        )
        return topics[:limit]

    def list_runs(self) -> list[ScoutRecord]:
        return list(self._runs.values())

    def latest_run(self) -> Optional[ScoutRecord]:
        """Return the most-recently started scout run, or None if no runs exist."""
        if not self._runs:
            return None
        return max(self._runs.values(), key=lambda r: r.started_at)

    # ------------------------------------------------------------------
    # SSE broadcast — real-time source / topic stream
    # ------------------------------------------------------------------

    def register_sse_subscriber(self, queue: asyncio.Queue) -> None:
        self._sse_queues.add(queue)

    def unregister_sse_subscriber(self, queue: asyncio.Queue) -> None:
        self._sse_queues.discard(queue)

    def broadcast_event(self, event: dict) -> None:
        """Push event to all active SSE subscriber queues (non-blocking, drop on overflow)."""
        dead = []
        for q in list(self._sse_queues):
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            self._sse_queues.discard(q)

    # ------------------------------------------------------------------
    # Human feedback store
    # ------------------------------------------------------------------

    def add_feedback(
        self,
        topic_id: str,
        rating: int,
        helpful: bool,
        comment: str = "",
    ) -> None:
        from datetime import datetime, timezone
        fb = FeedbackRecord(
            topic_id=topic_id,
            rating=max(1, min(5, rating)),
            helpful=helpful,
            comment=comment[:500],
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        self._feedback.append(fb)
        if len(self._feedback) > 1000:
            self._feedback = self._feedback[-1000:]
        try:
            from api.monitoring import scout_feedback_total
            scout_feedback_total.labels(helpful="yes" if helpful else "no").inc()
        except Exception:
            pass

    def list_feedback(self, topic_id: Optional[str] = None) -> list:
        if topic_id:
            return [f for f in self._feedback if f.topic_id == topic_id]
        return list(self._feedback)

    def sse_subscriber_count(self) -> int:
        return len(self._sse_queues)


# Singleton imported by scout router
scouts = ScoutManager()
