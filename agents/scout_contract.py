"""
agents/scout_contract.py — shared Scout topic contract and ranking helpers.

This module centralizes cross-layer constants used by:
  - agents/topic_scout.py (discovery ranking)
  - api/state.py (persisted topic listing ranking)

Keeping one source of truth prevents drift and reduces merge conflicts.
"""

from __future__ import annotations

from typing import Any

# Shared ranking weights for topic prioritization
TOPIC_PRIORITY_WEIGHTS: dict[str, float] = {
    "quality_gate_passed": 0.20,
    "quality_score": 0.45,
    "uniqueness_score": 0.25,
    "knowledge_gap_score": 0.20,
    "demand_score": 0.10,
}


def topic_priority_score(topic: Any) -> float:
    """Compute normalized priority score for Scout topic-like objects."""
    gate = 1.0 if bool(getattr(topic, "quality_gate_passed", False)) else 0.0
    quality = float(getattr(topic, "quality_score", 0.0))
    uniq = float(getattr(topic, "uniqueness_score", 0.0))
    gap = float(getattr(topic, "knowledge_gap_score", 0.0))
    demand = float(getattr(topic, "demand_score", 0.0))
    w = TOPIC_PRIORITY_WEIGHTS
    return (
        w["quality_gate_passed"] * gate
        + w["quality_score"] * quality
        + w["uniqueness_score"] * uniq
        + w["knowledge_gap_score"] * gap
        + w["demand_score"] * demand
    )
