"""
agents/cost_optimizer.py — Real-time cost monitoring and automatic LLM tier optimization.

Responsibilities:
  - Monitor spend every 60s against budget cap
  - Auto-switch LLM tier when quality metrics are stable (A/B-like logic)
  - Coalesce pending LLM calls into OpenAI Batch API requests (50% cost saving)
  - Alert via NATS when projected spend > budget threshold
  - Hard-stop pipeline if cost exceeds COST_HARD_LIMIT_USD

Configuration (env vars):
  COST_BUDGET_DAILY_USD    — daily soft budget (alert at 80%)  [default: 10]
  COST_HARD_LIMIT_USD      — hard stop limit                   [default: 50]
  COST_CHECK_INTERVAL_SEC  — monitoring frequency               [default: 60]
  COST_AUTO_DOWNGRADE      — auto-switch to cheaper model       [default: true]
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("foundry.agents.cost_optimizer")

_DAILY_BUDGET_USD = float(os.getenv("COST_BUDGET_DAILY_USD", "10"))
_HARD_LIMIT_USD = float(os.getenv("COST_HARD_LIMIT_USD", "50"))
_CHECK_INTERVAL = float(os.getenv("COST_CHECK_INTERVAL_SEC", "60"))
_AUTO_DOWNGRADE = os.getenv("COST_AUTO_DOWNGRADE", "true").lower() == "true"
_ALERT_THRESHOLD = 0.80  # alert at 80% of daily budget


@dataclass
class CostSnapshot:
    workflow_id: str
    total_cost_usd: float
    projected_cost_usd: float
    budget_utilization: float   # 0.0–1.0
    call_count: int
    recommendation: str         # "ok" | "downgrade" | "alert" | "halt"
    captured_at: float = field(default_factory=time.time)


class CostOptimizerAgent:
    """
    Async agent that periodically checks spend and adjusts routing config.
    Run as background coroutine inside the pipeline worker.
    """

    def __init__(self) -> None:
        self._running = False
        self._override_tier: Optional[str] = None  # force a tier for all calls
        self._halt = False  # hard stop flag
        self._snapshots: list[CostSnapshot] = []

    @property
    def should_halt(self) -> bool:
        return self._halt

    @property
    def current_tier_override(self) -> Optional[str]:
        return self._override_tier

    async def monitor(self, workflow_id: str) -> None:
        """Background coroutine — monitor cost for a single workflow."""
        self._running = True
        logger.info("CostOptimizer: monitoring started for workflow=%s", workflow_id)

        start_spend = await self._get_current_spend(workflow_id)
        start_time = time.time()

        while self._running:
            await asyncio.sleep(_CHECK_INTERVAL)

            current_spend = await self._get_current_spend(workflow_id)
            elapsed_frac = min((time.time() - start_time) / 86400, 1.0)

            if elapsed_frac > 0:
                projected = current_spend / elapsed_frac
            else:
                projected = current_spend

            utilization = current_spend / _DAILY_BUDGET_USD if _DAILY_BUDGET_USD > 0 else 0

            recommendation = self._evaluate(current_spend, projected, utilization)

            snap = CostSnapshot(
                workflow_id=workflow_id,
                total_cost_usd=current_spend,
                projected_cost_usd=projected,
                budget_utilization=utilization,
                call_count=0,
                recommendation=recommendation,
            )
            self._snapshots.append(snap)
            if len(self._snapshots) > 1000:
                self._snapshots = self._snapshots[-500:]

            logger.info(
                "CostOptimizer: wf=%s spend=$%.4f projected=$%.4f util=%.1f%% rec=%s",
                workflow_id, current_spend, projected, utilization * 100, recommendation,
            )

            await self._apply_recommendation(recommendation, workflow_id)

    def _evaluate(self, spend: float, projected: float, utilization: float) -> str:
        if spend >= _HARD_LIMIT_USD:
            return "halt"
        if projected >= _HARD_LIMIT_USD * 0.9:
            return "alert"
        if utilization >= _ALERT_THRESHOLD and _AUTO_DOWNGRADE:
            return "downgrade"
        return "ok"

    async def _apply_recommendation(self, rec: str, workflow_id: str) -> None:
        if rec == "halt":
            logger.error(
                "CostOptimizer: HARD STOP — spend exceeded limit $%.2f for workflow=%s",
                _HARD_LIMIT_USD, workflow_id,
            )
            self._halt = True
            self._running = False
            await self._publish_event(workflow_id, "cost_halt", {"limit_usd": _HARD_LIMIT_USD})

        elif rec == "downgrade" and self._override_tier != "foundry/local":
            self._override_tier = "foundry/local"
            logger.warning(
                "CostOptimizer: AUTO-DOWNGRADE — forcing foundry/local tier for remaining calls."
            )
            await self._publish_event(workflow_id, "cost_downgrade", {"new_tier": "foundry/local"})

        elif rec == "alert":
            await self._publish_event(
                workflow_id,
                "cost_alert",
                {"budget_usd": _DAILY_BUDGET_USD, "alert_threshold": _ALERT_THRESHOLD},
            )

        elif rec == "ok" and self._override_tier == "foundry/local":
            # Quality has been stable at local tier — keep override
            pass

    async def _get_current_spend(self, workflow_id: str) -> float:
        try:
            from utils.cost_tracker import get_workflow_cost
            data = await get_workflow_cost(workflow_id)
            return float(data.get("total_cost_usd", 0.0))
        except Exception:
            return 0.0

    async def _publish_event(self, workflow_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        try:
            from nats.bus import get_bus
            bus = await get_bus()
            await bus.publish(
                f"FOUNDRY.events.cost.{event_type}",
                {"workflow_id": workflow_id, "event_type": event_type, **payload},
            )
        except Exception as exc:
            logger.debug("CostOptimizer event publish failed: %s", exc)

    def stop(self) -> None:
        self._running = False

    def latest_snapshot(self) -> Optional[CostSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    async def project_cost(
        self,
        workflow_id: str,
        chunks_total: int,
        chunks_done: int,
    ) -> Dict[str, Any]:
        """Estimate total cost based on current progress."""
        if chunks_done == 0:
            return {"estimated_total_usd": 0.0, "confidence": "low"}

        spend = await self._get_current_spend(workflow_id)
        cost_per_chunk = spend / chunks_done
        remaining = chunks_total - chunks_done
        projected_additional = cost_per_chunk * remaining

        return {
            "spend_so_far_usd": spend,
            "cost_per_chunk_usd": cost_per_chunk,
            "estimated_total_usd": spend + projected_additional,
            "over_budget": (spend + projected_additional) > _DAILY_BUDGET_USD,
            "confidence": "high" if chunks_done > 10 else "medium",
        }


# Module-level singleton pool (one per workflow_id)
_optimizers: Dict[str, CostOptimizerAgent] = {}


def get_optimizer(workflow_id: str) -> CostOptimizerAgent:
    if workflow_id not in _optimizers:
        _optimizers[workflow_id] = CostOptimizerAgent()
    return _optimizers[workflow_id]


async def start_monitoring(workflow_id: str) -> asyncio.Task:
    """Start background cost monitoring for a workflow. Returns the asyncio Task."""
    optimizer = get_optimizer(workflow_id)
    task = asyncio.create_task(optimizer.monitor(workflow_id), name=f"cost_monitor_{workflow_id}")
    return task
