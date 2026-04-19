"""
api/monitoring.py — Optional Prometheus metrics for Foundry Studio.

Install: pip install prometheus-client
Exposes: GET /metrics  (scraped by Prometheus or Grafana Agent)

Counters & gauges:
  foundry_pipeline_runs_total       — total pipeline runs started
  foundry_pipeline_runs_active      — currently running pipeline runs
  foundry_records_written_total     — SFT records written to JSONL
  foundry_dpo_pairs_total           — DPO/ORPO preference pairs written
  foundry_scout_runs_total          — Gap Scout runs started
  foundry_topics_discovered_total   — knowledge-gap topics discovered
  foundry_judge_score               — histogram of judge quality scores
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _AVAILABLE = True
    _registry = CollectorRegistry(auto_describe=True)

    pipeline_runs_total = Counter(
        "foundry_pipeline_runs_total",
        "Total pipeline runs started",
        registry=_registry,
    )
    pipeline_runs_active = Gauge(
        "foundry_pipeline_runs_active",
        "Currently active pipeline runs",
        registry=_registry,
    )
    records_written_total = Counter(
        "foundry_records_written_total",
        "Total SFT records written to JSONL",
        registry=_registry,
    )
    dpo_pairs_total = Counter(
        "foundry_dpo_pairs_total",
        "Total DPO/ORPO preference pairs written",
        registry=_registry,
    )
    scout_runs_total = Counter(
        "foundry_scout_runs_total",
        "Total Gap Scout runs started",
        registry=_registry,
    )
    topics_discovered_total = Counter(
        "foundry_topics_discovered_total",
        "Total knowledge-gap topics discovered",
        registry=_registry,
    )
    judge_score_histogram = Histogram(
        "foundry_judge_score",
        "Distribution of judge quality scores",
        buckets=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
        registry=_registry,
    )

except ImportError:
    _AVAILABLE = False
    _registry = None  # type: ignore
    logger.info("prometheus-client not installed — /metrics endpoint disabled")


def is_available() -> bool:
    return _AVAILABLE


def get_metrics_payload() -> tuple[bytes, str]:
    """Return (body_bytes, content_type). Raises RuntimeError if unavailable."""
    if not _AVAILABLE:
        raise RuntimeError("prometheus-client not installed: pip install prometheus-client")
    return generate_latest(_registry), CONTENT_TYPE_LATEST  # type: ignore[return-value]
