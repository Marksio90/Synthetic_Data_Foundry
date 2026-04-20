"""
agents/auto_reviewer.py — Trójstrefowa automatyczna weryfikacja jakości Q&A.

Strefy:
  ZIELONA  score >= APPROVE_THRESHOLD  → auto-approve (zapisz human_reviewed=True)
  SZARA    REVIEW_THRESHOLD <= score < APPROVE_THRESHOLD → kolejka do opcjonalnego przeglądu
  CZERWONA score < REVIEW_THRESHOLD    → auto-reject (oznacz jako odrzucony)

Progi domyślne (nadpisywalne):
  APPROVE_THRESHOLD = 0.88  (wysoka pewność — nie wymaga człowieka)
  REVIEW_THRESHOLD  = 0.70  (strefa szara — człowiek może ocenić opcjonalnie)

Pipeline NIE czeka na decyzję człowieka — próbki w strefie szarej są
włączane do datasetu z flagą human_reviewed=False.
Człowiek może je przejrzeć w UI i ewentualnie odrzucić lub edytować.

Wymagane migracje DB (dodane do run_migrations w main.py):
  ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS human_reviewed BOOLEAN DEFAULT NULL;
  ALTER TABLE generated_samples ADD COLUMN IF NOT EXISTS human_flag TEXT;
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text, update
from sqlalchemy.orm import Session

from db.models import GeneratedSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

APPROVE_THRESHOLD: float = 0.88  # ≥ → auto-approve
REVIEW_THRESHOLD: float = 0.70   # in [0.70, 0.88) → szara strefa (opcjonalny przegląd)
# < 0.70 → auto-reject


@dataclass
class ReviewDecision:
    sample_id: str
    score: float
    zone: str              # "green" | "grey" | "red"
    action: str            # "approved" | "queued" | "rejected"
    reason: str


@dataclass
class ReviewSummary:
    total: int
    approved: int
    queued: int
    rejected: int
    approve_threshold: float
    review_threshold: float
    priority_preview: list[dict[str, object]]

    @property
    def approval_rate(self) -> float:
        return self.approved / max(self.total, 1)

    def log(self) -> None:
        logger.info(
            "AutoReview: %d total → %d approved (%.0f%%) | %d queued | %d rejected",
            self.total, self.approved, self.approval_rate * 100,
            self.queued, self.rejected,
        )


# ---------------------------------------------------------------------------
# Core review logic
# ---------------------------------------------------------------------------

def review_sample(
    score: Optional[float],
    approve_threshold: float = APPROVE_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> ReviewDecision:
    """Classify a single sample score into a zone and decide action."""
    s = float(score) if score is not None else 0.0

    if s >= approve_threshold:
        return ReviewDecision(
            sample_id="",
            score=s,
            zone="green",
            action="approved",
            reason=f"Score {s:.2f} ≥ {approve_threshold} → auto-approved",
        )
    elif s >= review_threshold:
        return ReviewDecision(
            sample_id="",
            score=s,
            zone="grey",
            action="queued",
            reason=f"Score {s:.2f} in [{review_threshold}, {approve_threshold}) → optional review",
        )
    else:
        return ReviewDecision(
            sample_id="",
            score=s,
            zone="red",
            action="rejected",
            reason=f"Score {s:.2f} < {review_threshold} → auto-rejected",
        )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run_auto_review(
    session: Session,
    batch_id: Optional[str] = None,
    approve_threshold: float = APPROVE_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
    priority_limit: int = 20,
    dry_run: bool = False,
) -> ReviewSummary:
    """
    Process all unreviewed samples (human_reviewed IS NULL) for the given batch.
    Updates DB in bulk — auto-approved samples get human_reviewed=True,
    auto-rejected get human_reviewed=False + human_flag='auto_rejected'.
    Grey-zone samples stay as human_reviewed=NULL (pending optional review).

    Args:
        session:           SQLAlchemy session.
        batch_id:          If set, only process samples from this batch.
        approve_threshold: Score above which samples are auto-approved.
        review_threshold:  Score below which samples are auto-rejected.
        priority_limit:    Limit rekordów zwracanych w podglądzie kolejki review.
        dry_run:           If True, compute decisions without writing to DB.

    Returns:
        ReviewSummary with counts.
    """
    from sqlalchemy import select, and_

    query = select(GeneratedSample).where(
        GeneratedSample.human_reviewed.is_(None)
    )
    if batch_id:
        query = query.where(GeneratedSample.batch_id == batch_id)

    samples = session.scalars(query).all()

    approved_ids: list = []
    rejected_ids: list = []
    queued_ids: list = []
    queued_scores: dict = {}
    queued = 0

    for s in samples:
        decision = review_sample(s.quality_score, approve_threshold, review_threshold)
        decision.sample_id = str(s.id)

        if decision.action == "approved":
            approved_ids.append(s.id)
        elif decision.action == "rejected":
            rejected_ids.append(s.id)
        else:
            queued += 1
            queued_ids.append(s.id)
            queued_scores[s.id] = float(s.quality_score or 0.0)

        logger.debug(
            "AutoReview [%s]: chunk=%s score=%.2f → %s",
            decision.action, str(s.id)[:8], decision.score, decision.reason,
        )

    if not dry_run:
        if approved_ids:
            session.execute(
                update(GeneratedSample)
                .where(GeneratedSample.id.in_(approved_ids))
                .values(human_reviewed=True, human_flag="auto_approved")
            )
        if rejected_ids:
            session.execute(
                update(GeneratedSample)
                .where(GeneratedSample.id.in_(rejected_ids))
                .values(human_reviewed=False, human_flag="auto_rejected")
            )
        session.commit()

    summary = ReviewSummary(
        total=len(samples),
        approved=len(approved_ids),
        queued=queued,
        rejected=len(rejected_ids),
        approve_threshold=approve_threshold,
        review_threshold=review_threshold,
        priority_preview=[
            {
                "sample_id": str(sid),
                "score": round(queued_scores.get(sid, 0.0), 4),
                "distance_to_midpoint": round(
                    abs(queued_scores.get(sid, 0.0) - ((approve_threshold + review_threshold) / 2.0)),
                    4,
                ),
            }
            for sid in sorted(
                queued_ids,
                key=lambda sid: abs(
                    queued_scores.get(sid, 0.0)
                    - ((approve_threshold + review_threshold) / 2.0)
                ),
            )[: max(1, priority_limit)]
        ],
    )
    summary.log()
    return summary
