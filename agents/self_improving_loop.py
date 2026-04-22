"""
agents/self_improving_loop.py — Autonomous self-improving calibration loop.

Reads BatchWeaknessReport from CriticAgent after each pipeline run and
automatically adjusts pipeline parameters:
  - quality_threshold   — raised when hallucination or factual errors are found
  - adversarial_ratio   — raised when critique coverage is low
  - max_turns           — raised when depth/completeness is weak
  - perspective_weights — shifts focus toward under-performing perspectives

All adjustments are bounded, smoothed (EMA), and persisted to PostgreSQL so
the loop converges rather than oscillates. A hard-reset escape hatch restores
safe defaults if metrics regress for 3+ consecutive cycles.

Usage:
    loop = SelfImprovingLoop(db_url=settings.database_url)
    new_calibration = await loop.run_cycle(
        workflow_id="wf-abc123",
        weakness_report=critic_agent.batch_weakness_report,
        current_calibration=calibration_result,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("foundry.self_improving_loop")

# ── Safety bounds ─────────────────────────────────────────────────────────────
_QUALITY_MIN = 0.60
_QUALITY_MAX = 0.95
_ADVERSARIAL_MIN = 0.05
_ADVERSARIAL_MAX = 0.35
_MAX_TURNS_MIN = 1
_MAX_TURNS_MAX = 8

# EMA smoothing factor (0 = no update, 1 = full step)
_EMA_ALPHA = 0.3

# Consecutive regression cycles before hard reset
_REGRESSION_LIMIT = 3


@dataclass
class AdaptedCalibration:
    """Extended calibration result produced by the self-improving loop."""
    quality_threshold: float
    adversarial_ratio: float
    max_turns: int
    perspective_weights: Dict[str, float] = field(default_factory=dict)
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reasoning: List[str] = field(default_factory=list)
    was_reset: bool = False

    def as_env_overrides(self) -> Dict[str, str]:
        return {
            "QUALITY_THRESHOLD": str(round(self.quality_threshold, 3)),
            "ADVERSARIAL_RATIO": str(round(self.adversarial_ratio, 3)),
            "MAX_TURNS": str(self.max_turns),
        }

    def summary(self) -> str:
        lines = [
            f"[cycle={self.cycle_id}] quality_threshold={self.quality_threshold:.3f} "
            f"adversarial_ratio={self.adversarial_ratio:.3f} max_turns={self.max_turns}",
        ]
        for r in self.reasoning:
            lines.append(f"  → {r}")
        return "\n".join(lines)


@dataclass
class ImprovementRecord:
    """Persisted record of one calibration adjustment cycle."""
    record_id: str
    workflow_id: str
    cycle_id: str
    timestamp: str
    old_quality_threshold: float
    new_quality_threshold: float
    old_adversarial_ratio: float
    new_adversarial_ratio: float
    old_max_turns: int
    new_max_turns: int
    hallucination_rate: float
    avg_quality_score: float
    n_critiqued: int
    top_weaknesses: List[str]
    was_reset: bool
    reasoning: List[str]


class SelfImprovingLoop:
    """
    Autonomous calibration optimizer.

    After each pipeline run, collects weakness signals from the CriticAgent
    and nudges calibration parameters in the direction that should reduce
    weaknesses while keeping metrics stable.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self._db_url = db_url
        self._history: List[ImprovementRecord] = []
        self._regression_streak = 0
        self._prev_avg_quality: Optional[float] = None
        self._db_available = False
        self._pool = None

    async def _ensure_db(self) -> None:
        if self._db_available or not self._db_url:
            return
        try:
            import asyncpg  # type: ignore
            self._pool = await asyncpg.create_pool(self._db_url, min_size=1, max_size=3)
            await self._pool.execute("""
                CREATE TABLE IF NOT EXISTS calibration_history (
                    record_id      TEXT PRIMARY KEY,
                    workflow_id    TEXT NOT NULL,
                    cycle_id       TEXT NOT NULL,
                    recorded_at    TIMESTAMPTZ DEFAULT NOW(),
                    params_before  JSONB NOT NULL,
                    params_after   JSONB NOT NULL,
                    metrics        JSONB NOT NULL,
                    reasoning      JSONB NOT NULL,
                    was_reset      BOOLEAN DEFAULT FALSE
                );
                CREATE INDEX IF NOT EXISTS idx_cal_history_workflow
                    ON calibration_history (workflow_id, recorded_at DESC);
            """)
            self._db_available = True
            logger.info("SelfImprovingLoop: DB connection established")
        except Exception as exc:
            logger.warning("SelfImprovingLoop: DB unavailable (%s) — history stored in-memory only", exc)

    async def _load_recent_history(self, workflow_id: str, limit: int = 10) -> List[dict]:
        """Fetch recent calibration records for this workflow from DB."""
        if not self._db_available or not self._pool:
            return []
        try:
            rows = await self._pool.fetch(
                """SELECT params_before, params_after, metrics, was_reset, recorded_at
                   FROM calibration_history
                   WHERE workflow_id = $1
                   ORDER BY recorded_at DESC LIMIT $2""",
                workflow_id, limit,
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("SelfImprovingLoop._load_recent_history failed: %s", exc)
            return []

    async def _persist_record(self, rec: ImprovementRecord) -> None:
        if not self._db_available or not self._pool:
            self._history.append(rec)
            return
        try:
            await self._pool.execute(
                """INSERT INTO calibration_history
                   (record_id, workflow_id, cycle_id, params_before, params_after, metrics, reasoning, was_reset)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (record_id) DO NOTHING""",
                rec.record_id,
                rec.workflow_id,
                rec.cycle_id,
                json.dumps({
                    "quality_threshold": rec.old_quality_threshold,
                    "adversarial_ratio": rec.old_adversarial_ratio,
                    "max_turns": rec.old_max_turns,
                }),
                json.dumps({
                    "quality_threshold": rec.new_quality_threshold,
                    "adversarial_ratio": rec.new_adversarial_ratio,
                    "max_turns": rec.new_max_turns,
                }),
                json.dumps({
                    "hallucination_rate": rec.hallucination_rate,
                    "avg_quality_score": rec.avg_quality_score,
                    "n_critiqued": rec.n_critiqued,
                    "top_weaknesses": rec.top_weaknesses,
                }),
                json.dumps(rec.reasoning),
                rec.was_reset,
            )
            logger.debug("SelfImprovingLoop: persisted cycle %s", rec.cycle_id)
        except Exception as exc:
            logger.warning("SelfImprovingLoop._persist_record failed: %s", exc)
            self._history.append(rec)

    # ── Core adjustment logic ──────────────────────────────────────────────────

    def _ema(self, old: float, new_target: float) -> float:
        return old + _EMA_ALPHA * (new_target - old)

    def _detect_regression(self, avg_quality: float) -> bool:
        """Returns True if quality has been declining for _REGRESSION_LIMIT cycles."""
        if self._prev_avg_quality is not None and avg_quality < self._prev_avg_quality - 0.02:
            self._regression_streak += 1
        else:
            self._regression_streak = 0
        self._prev_avg_quality = avg_quality
        return self._regression_streak >= _REGRESSION_LIMIT

    def _safe_defaults(self) -> AdaptedCalibration:
        self._regression_streak = 0
        return AdaptedCalibration(
            quality_threshold=0.82,
            adversarial_ratio=0.10,
            max_turns=3,
            reasoning=["Hard reset — consecutive quality regression detected; restoring safe defaults"],
            was_reset=True,
        )

    def _compute_adjustments(
        self,
        report: dict,
        quality_threshold: float,
        adversarial_ratio: float,
        max_turns: int,
    ) -> AdaptedCalibration:
        """
        Compute new calibration parameters from a BatchWeaknessReport dict.

        Expected report keys (all optional, gracefully defaulted):
            hallucination_rate, avg_quality_score, n_critiqued,
            top_weakness_categories (list of str)
        """
        reasoning: List[str] = []

        hall_rate   = float(report.get("hallucination_rate", 0.0))
        avg_quality = float(report.get("avg_quality_score", quality_threshold))
        n_critiqued = int(report.get("n_critiqued", 0))
        weaknesses: List[str] = [w.lower() for w in report.get("top_weakness_categories", [])]

        new_qt = quality_threshold
        new_ar = adversarial_ratio
        new_mt = max_turns

        # ── Rule 1: high hallucination → tighten quality threshold ────────────
        if hall_rate > 0.10:
            target_qt = min(quality_threshold + 0.04, _QUALITY_MAX)
            new_qt = self._ema(quality_threshold, target_qt)
            reasoning.append(
                f"Hallucination rate {hall_rate:.1%} > 10% → quality_threshold {quality_threshold:.3f}→{new_qt:.3f}"
            )
        elif hall_rate > 0.05:
            target_qt = min(quality_threshold + 0.02, _QUALITY_MAX)
            new_qt = self._ema(quality_threshold, target_qt)
            reasoning.append(
                f"Hallucination rate {hall_rate:.1%} > 5% → quality_threshold {quality_threshold:.3f}→{new_qt:.3f}"
            )
        elif hall_rate < 0.01 and avg_quality > 0.88:
            # Quality very high — can relax threshold slightly to increase throughput
            target_qt = max(quality_threshold - 0.01, _QUALITY_MIN)
            new_qt = self._ema(quality_threshold, target_qt)
            reasoning.append(
                f"Low hallucination ({hall_rate:.1%}) + high quality → relax threshold to {new_qt:.3f}"
            )

        # ── Rule 2: low critique coverage → generate more adversarial pairs ───
        if n_critiqued == 0:
            reasoning.append("No critiques generated — adversarial_ratio unchanged")
        elif avg_quality < 0.70:
            target_ar = min(adversarial_ratio + 0.05, _ADVERSARIAL_MAX)
            new_ar = self._ema(adversarial_ratio, target_ar)
            reasoning.append(
                f"Low avg_quality {avg_quality:.2f} → adversarial_ratio {adversarial_ratio:.3f}→{new_ar:.3f}"
            )
        elif avg_quality > 0.90 and adversarial_ratio > 0.12:
            target_ar = max(adversarial_ratio - 0.02, _ADVERSARIAL_MIN)
            new_ar = self._ema(adversarial_ratio, target_ar)
            reasoning.append(
                f"High avg_quality {avg_quality:.2f} → reduce adversarial_ratio to {new_ar:.3f}"
            )

        # ── Rule 3: weakness-specific adjustments ─────────────────────────────
        if any(w in ("depth", "completeness", "comprehensiveness") for w in weaknesses):
            new_mt = min(max_turns + 1, _MAX_TURNS_MAX)
            reasoning.append(f"Weakness 'depth' detected → max_turns {max_turns}→{new_mt}")

        if any(w in ("accuracy", "factual", "factual_accuracy", "citations") for w in weaknesses):
            target_qt = min(new_qt + 0.02, _QUALITY_MAX)
            new_qt = self._ema(new_qt, target_qt)
            reasoning.append(f"Weakness 'factual_accuracy' → quality_threshold further tightened to {new_qt:.3f}")

        if any(w in ("brevity", "verbosity", "conciseness") for w in weaknesses):
            new_mt = max(max_turns - 1, _MAX_TURNS_MIN)
            reasoning.append(f"Weakness 'verbosity' → max_turns {max_turns}→{new_mt}")

        # ── Rule 4: perspective weighting from weakness categories ────────────
        _PERSPECTIVE_KEYS = [
            "regulatory", "financial", "risk", "esg_data",
            "cross_reference", "practical", "critical", "forward_looking",
        ]
        perspective_weights: Dict[str, float] = {k: 1.0 for k in _PERSPECTIVE_KEYS}
        for w in weaknesses:
            for p in _PERSPECTIVE_KEYS:
                if p in w or w in p:
                    # Boost this perspective proportionally
                    perspective_weights[p] = min(perspective_weights[p] + 0.3, 2.0)
                    reasoning.append(f"Boosting perspective '{p}' weight due to weakness '{w}'")

        if not reasoning:
            reasoning.append("No significant weaknesses detected — parameters unchanged")

        return AdaptedCalibration(
            quality_threshold=round(max(_QUALITY_MIN, min(new_qt, _QUALITY_MAX)), 4),
            adversarial_ratio=round(max(_ADVERSARIAL_MIN, min(new_ar, _ADVERSARIAL_MAX)), 4),
            max_turns=max(_MAX_TURNS_MIN, min(new_mt, _MAX_TURNS_MAX)),
            perspective_weights=perspective_weights,
            reasoning=reasoning,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run_cycle(
        self,
        workflow_id: str,
        weakness_report: Any,
        current_quality_threshold: float = 0.82,
        current_adversarial_ratio: float = 0.10,
        current_max_turns: int = 3,
    ) -> AdaptedCalibration:
        """
        Main entry point. Call after each pipeline run.

        Args:
            workflow_id:               Identifier of the completed pipeline run.
            weakness_report:           BatchWeaknessReport instance or plain dict.
            current_quality_threshold: Current quality threshold value.
            current_adversarial_ratio: Current adversarial ratio.
            current_max_turns:         Current max turns per chunk.

        Returns:
            AdaptedCalibration with updated parameters + reasoning.
        """
        await self._ensure_db()

        # Accept both dataclass and plain dict
        if hasattr(weakness_report, "__dict__"):
            report_dict: dict = {
                k: v for k, v in vars(weakness_report).items()
                if not k.startswith("_")
            }
        elif isinstance(weakness_report, dict):
            report_dict = weakness_report
        else:
            report_dict = {}

        avg_quality = float(report_dict.get("avg_quality_score", current_quality_threshold))

        # Check for sustained regression → hard reset
        if self._detect_regression(avg_quality):
            logger.warning(
                "SelfImprovingLoop: %d consecutive quality regressions — issuing hard reset",
                _REGRESSION_LIMIT,
            )
            adapted = self._safe_defaults()
        else:
            adapted = self._compute_adjustments(
                report_dict,
                current_quality_threshold,
                current_adversarial_ratio,
                current_max_turns,
            )

        logger.info("SelfImprovingLoop cycle %s for workflow=%s:\n%s", adapted.cycle_id, workflow_id, adapted.summary())

        # Persist record
        record = ImprovementRecord(
            record_id=hashlib.sha1(f"{workflow_id}-{adapted.cycle_id}".encode()).hexdigest()[:16],
            workflow_id=workflow_id,
            cycle_id=adapted.cycle_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            old_quality_threshold=current_quality_threshold,
            new_quality_threshold=adapted.quality_threshold,
            old_adversarial_ratio=current_adversarial_ratio,
            new_adversarial_ratio=adapted.adversarial_ratio,
            old_max_turns=current_max_turns,
            new_max_turns=adapted.max_turns,
            hallucination_rate=float(report_dict.get("hallucination_rate", 0.0)),
            avg_quality_score=avg_quality,
            n_critiqued=int(report_dict.get("n_critiqued", 0)),
            top_weaknesses=list(report_dict.get("top_weakness_categories", [])),
            was_reset=adapted.was_reset,
            reasoning=adapted.reasoning,
        )
        await self._persist_record(record)

        return adapted

    async def get_history(self, workflow_id: str, limit: int = 20) -> List[dict]:
        """Return recent improvement history for this workflow."""
        db_rows = await self._load_recent_history(workflow_id, limit)
        if db_rows:
            return db_rows
        # Fall back to in-memory history
        return [
            asdict(r) for r in self._history
            if r.workflow_id == workflow_id
        ][-limit:]

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()


# ── Convenience singleton ──────────────────────────────────────────────────────
_loop_instance: Optional[SelfImprovingLoop] = None


def get_self_improving_loop(db_url: Optional[str] = None) -> SelfImprovingLoop:
    global _loop_instance
    if _loop_instance is None:
        _loop_instance = SelfImprovingLoop(db_url=db_url)
    return _loop_instance
