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
  foundry_scout_sources_active      — active crawlers per source (gauge)
  foundry_scout_topics_per_source   — topics found per source (counter)
  foundry_scout_fetch_duration_seconds — fetch latency per source (histogram)
  foundry_scout_verification_failures  — verification failures per reason/source
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

    # ------------------------------------------------------------------
    # Gap Scout crawler metrics (added in Step 2)
    # ------------------------------------------------------------------
    scout_sources_active = Gauge(
        "foundry_scout_sources_active",
        "Number of active (non-paused) crawlers",
        labelnames=["source"],
        registry=_registry,
    )
    scout_topics_per_source = Counter(
        "foundry_scout_topics_per_source",
        "Total topics discovered per source crawler",
        labelnames=["source"],
        registry=_registry,
    )
    scout_fetch_duration = Histogram(
        "foundry_scout_fetch_duration_seconds",
        "HTTP fetch duration per source crawler (seconds)",
        labelnames=["source"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0],
        registry=_registry,
    )
    scout_verification_failures = Counter(
        "foundry_scout_verification_failures",
        "Verification failures per reason and source crawler",
        labelnames=["reason", "source"],
        registry=_registry,
    )

    # ------------------------------------------------------------------
    # KROK 11 — additional Gap Scout observability
    # ------------------------------------------------------------------
    scout_websub_deliveries_total = Counter(
        "foundry_scout_websub_deliveries_total",
        "Total WebSub content deliveries received (verified + injected)",
        labelnames=["source_type"],
        registry=_registry,
    )
    scout_sse_subscribers = Gauge(
        "foundry_scout_sse_subscribers",
        "Currently active SSE /live stream subscribers",
        registry=_registry,
    )
    scout_feedback_total = Counter(
        "foundry_scout_feedback_total",
        "Total human feedback submissions on discovered topics",
        labelnames=["helpful"],
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
