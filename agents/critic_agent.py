"""
agents/critic_agent.py — Critic Agent: self-reflection loop + quality improvement.

Responsibilities:
  1. Selective critique (20% sampling, not every sample)
  2. Generate DPO rejected pairs via targeted degradation
  3. Detect systematic weaknesses across batches
  4. Recommend parameter recalibration to Orchestrator

Integration points:
  - Called after judge_answer when quality < 0.85 (selective)
  - Writes DPO pairs to dpo_writer
  - Publishes recalibration events to NATS bus

Why selective (20%)?
  - Constitutional AI on every sample = 2× LLM cost with diminishing returns
  - Selective critique on low/medium-confidence samples = 70% cost savings
  - Full batch critique only on < quality_threshold samples
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("foundry.agents.critic")

# Critique only this fraction of samples that score above threshold
# (below-threshold samples are always critiqued)
_SELECTIVE_RATE = float(__import__("os").getenv("CRITIC_SELECTIVE_RATE", "0.20"))

# Minimum score to trigger selective (vs always) critique
_SELECTIVE_MIN_SCORE = 0.75


@dataclass
class CritiqueResult:
    sample_id: str
    original_answer: str
    critique: str
    improved_answer: str
    degraded_answer: str    # intentionally worse version for DPO rejected pair
    issues: list[str]
    improvement_score_delta: float
    cost_cents: float


@dataclass
class BatchWeaknessReport:
    """Aggregated weaknesses detected across a batch of samples."""
    batch_id: str
    total_critiqued: int
    avg_score: float
    top_weaknesses: list[str]
    recommended_changes: dict[str, Any]
    perspective_scores: dict[str, float]
    generated_at: float = field(default_factory=time.time)


_CRITIQUE_SYSTEM = """Jesteś krytycznym audytorem syntetycznych danych treningowych ESG.
Twoim zadaniem jest:
1. Zidentyfikować konkretne słabości w odpowiedzi asystenta (max 3 punkty)
2. Napisać ulepszoną wersję odpowiedzi
3. Napisać celowo pogorszoną wersję (do par DPO) — usuń cytowania artykułów, dodaj ogólniki

FORMAT:
<issues>
- [opis słabości 1]
- [opis słabości 2]
</issues>

<improved>
[ulepszona odpowiedź — precyzyjna, z cytatami artykułów]
</improved>

<degraded>
[pogorszona odpowiedź — bez cytowań, ogólnikowa, dla DPO jako "rejected"]
</degraded>
"""

_WEAKNESS_ANALYSIS_SYSTEM = """Jesteś analitykiem jakości danych treningowych.
Przeanalizuj listę zidentyfikowanych słabości i zwróć:
1. Top-3 systematyczne problemy (z frekwencją)
2. Rekomendacje zmian konfiguracji (adversarial_ratio, quality_threshold, perspective weights)

Zwróć JSON:
{
  "top_weaknesses": ["opis1", "opis2", "opis3"],
  "recommended_config": {
    "adversarial_ratio": <float>,
    "quality_threshold": <float>,
    "perspective_focus": ["lista perspektyw do wzmocnienia"]
  }
}
"""


class CriticAgent:
    """
    Selective critique agent. Operates on completed pipeline outputs.
    The base selective rate adapts dynamically based on observed rejection rate —
    when quality drops, more samples are critiqued to surface weaknesses faster.
    """

    def __init__(self) -> None:
        self._total_seen: int = 0
        self._total_below_threshold: int = 0

    @property
    def effective_selective_rate(self) -> float:
        """Current critique rate for above-threshold samples, scaled by rejection rate."""
        if self._total_seen < 10:
            return _SELECTIVE_RATE
        rejection_rate = self._total_below_threshold / self._total_seen
        # Scale linearly: rejection_rate=0 → 0.5× base, rejection_rate=0.3 → 2× base
        return min(_SELECTIVE_RATE * (1.0 + 2.0 * rejection_rate), 0.80)

    def should_critique(self, quality_score: float) -> bool:
        """
        Decision rule: always critique low-quality, adaptively sample high-quality.
        Rate increases automatically as rejection rate rises.
        """
        self._total_seen += 1
        if quality_score < _SELECTIVE_MIN_SCORE:
            self._total_below_threshold += 1
            return True  # always critique borderline/poor quality
        return random.random() < self.effective_selective_rate

    def critique_sample(
        self,
        question: str,
        answer: str,
        context: str,
        perspective: str,
        sample_id: str = "",
        workflow_id: str = "",
    ) -> Optional[CritiqueResult]:
        """
        Run LLM critique on a single sample.
        Returns CritiqueResult or None on failure.
        """
        from utils.llm_router import LLMTier
        from utils.cost_tracker import record_cost
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self._async_critique(
                    question, answer, context, perspective, sample_id, workflow_id
                )
            )
        finally:
            loop.close()
        return result

    async def _async_critique(
        self,
        question: str,
        answer: str,
        context: str,
        perspective: str,
        sample_id: str,
        workflow_id: str,
    ) -> Optional[CritiqueResult]:
        from utils.llm_router import get_completion, LLMTier
        from utils.cost_tracker import record_cost
        import re

        user_msg = (
            f"KONTEKST DYREKTYWY:\n{context[:3000]}\n\n"
            f"PERSPEKTYWA: {perspective}\n"
            f"PYTANIE: {question}\n\n"
            f"ODPOWIEDŹ DO OCENY:\n{answer}"
        )

        try:
            raw, cost = await get_completion(
                messages=[
                    {"role": "system", "content": _CRITIQUE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                tier=LLMTier.MID,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning("Critic LLM call failed: %s", exc)
            return None

        # Parse structured response
        issues = re.findall(r"- (.+)", _extract_tag(raw, "issues"))
        improved = _extract_tag(raw, "improved").strip()
        degraded = _extract_tag(raw, "degraded").strip()

        if not improved or not degraded:
            return None

        if workflow_id:
            record_cost(
                workflow_id=workflow_id,
                agent_name="critic",
                model_name="claude-haiku-4-5-20251001",
                model_tier="mid",
                prompt_tokens=len(user_msg) // 4,
                completion_tokens=len(raw) // 4,
                cost_usd=cost / 100,
            )

        return CritiqueResult(
            sample_id=sample_id,
            original_answer=answer,
            critique=_extract_tag(raw, "issues"),
            improved_answer=improved,
            degraded_answer=degraded,
            issues=issues,
            improvement_score_delta=0.0,  # filled by caller after re-judging
            cost_cents=cost,
        )

    async def analyze_batch_weaknesses(
        self,
        all_issues: List[str],
        perspective_scores: Dict[str, List[float]],
        batch_id: str,
        current_config: Dict[str, Any],
    ) -> BatchWeaknessReport:
        """
        Aggregate weaknesses across a batch and recommend config changes.
        """
        from utils.llm_router import get_completion, LLMTier
        import json

        avg_perspective_scores = {
            p: sum(scores) / len(scores)
            for p, scores in perspective_scores.items()
            if scores
        }

        issues_text = "\n".join(f"- {issue}" for issue in all_issues[:100])
        user_msg = (
            f"ZIDENTYFIKOWANE SŁABOŚCI (próbka {len(all_issues)}):\n{issues_text}\n\n"
            f"WYNIKI PERSPEKTYW: {json.dumps(avg_perspective_scores, ensure_ascii=False)}\n"
            f"AKTUALNA KONFIGURACJA: {json.dumps(current_config, ensure_ascii=False)}"
        )

        try:
            raw, _ = await get_completion(
                messages=[
                    {"role": "system", "content": _WEAKNESS_ANALYSIS_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                tier=LLMTier.QUALITY,
                temperature=0.1,
                max_tokens=512,
            )
            parsed = json.loads(raw)
        except Exception as exc:
            logger.warning("Batch weakness analysis failed: %s", exc)
            parsed = {"top_weaknesses": [], "recommended_config": {}}

        return BatchWeaknessReport(
            batch_id=batch_id,
            total_critiqued=len(all_issues),
            avg_score=sum(avg_perspective_scores.values()) / max(len(avg_perspective_scores), 1),
            top_weaknesses=parsed.get("top_weaknesses", []),
            recommended_changes=parsed.get("recommended_config", {}),
            perspective_scores=avg_perspective_scores,
        )


def _extract_tag(text: str, tag: str) -> str:
    import re
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# Module-level singleton
_critic: Optional[CriticAgent] = None


def get_critic() -> CriticAgent:
    global _critic
    if _critic is None:
        _critic = CriticAgent()
    return _critic
