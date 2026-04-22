"""
config/logging_config.py — Structured JSON logging via structlog.

Replaces bare logging.basicConfig() with a pipeline that emits JSON to stdout,
making logs parseable by Loki/Grafana, Datadog, and CloudWatch out of the box.

Usage:
    from config.logging_config import configure_logging
    configure_logging(level="INFO")

    import structlog
    logger = structlog.get_logger("foundry.agents.expert")
    logger.info("answer_generated", chunk_id=chunk_id, perspective=perspective, cost_cents=0.042)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    """
    Configure structlog + stdlib logging with JSON output.

    In development (LOG_FORMAT=text), falls back to coloured console output.
    In production (LOG_FORMAT=json, default), emits newline-delimited JSON.
    """
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    try:
        import structlog  # type: ignore

        shared_processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]

        if log_format == "text":
            renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
        else:
            renderer = structlog.processors.JSONRenderer()

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(numeric_level)

        # Silence noisy third-party loggers
        for noisy in ("httpx", "httpcore", "uvicorn.access", "apscheduler"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    except ImportError:
        # structlog not installed — fall back to stdlib JSON-ish format
        logging.basicConfig(
            level=numeric_level,
            format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)r}',
            stream=sys.stdout,
        )
        logging.getLogger(__name__).warning(
            "structlog not installed — using stdlib fallback. Run: pip install structlog"
        )
