"""
telemetry/tracing.py — OpenTelemetry distributed tracing setup.

Instruments FastAPI, SQLAlchemy, httpx, and custom pipeline spans.
Exports to Tempo (OTLP) in production; stdout in development.

Usage:
    from telemetry.tracing import setup_tracing, get_tracer

    # In api/main.py:
    setup_tracing(service_name="foundry-api")

    # In agents:
    tracer = get_tracer("foundry.agents.expert")
    with tracer.start_as_current_span("generate_answer") as span:
        span.set_attribute("chunk_id", chunk_id)
        span.set_attribute("perspective", perspective)
        answer = _call_vllm(system, user)
        span.set_attribute("answer_len", len(answer))
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("foundry.telemetry")

_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "foundry")
_ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
_tracer_provider = None


def setup_tracing(service_name: str = _SERVICE_NAME) -> None:
    """
    Configure OpenTelemetry with OTLP exporter → Grafana Tempo.
    Falls back to stdout (ConsoleSpanExporter) if opentelemetry packages missing.
    """
    global _tracer_provider

    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    except ImportError:
        logger.warning(
            "opentelemetry-sdk not installed — tracing disabled. "
            "Run: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return

    resource = Resource.create({
        "service.name": service_name,
        "deployment.environment": _ENVIRONMENT,
        "service.version": "2.0.0",
    })

    provider = TracerProvider(resource=resource)

    # Try OTLP exporter (Grafana Tempo)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore
        otlp_exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("OTel OTLP exporter configured → %s", _OTLP_ENDPOINT)
    except Exception as exc:
        logger.warning("OTLP exporter unavailable (%s) — using stdout exporter.", exc)
        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # type: ignore
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter  # type: ignore
            if _ENVIRONMENT == "development":
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        except ImportError:
            pass

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    # Auto-instrument FastAPI
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
        FastAPIInstrumentor().instrument()
        logger.info("FastAPI auto-instrumented with OTel.")
    except ImportError:
        pass

    # Auto-instrument SQLAlchemy
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor  # type: ignore
        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy auto-instrumented with OTel.")
    except ImportError:
        pass

    # Auto-instrument httpx
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # type: ignore
        HTTPXClientInstrumentor().instrument()
        logger.info("httpx auto-instrumented with OTel.")
    except ImportError:
        pass

    logger.info("OpenTelemetry tracing configured (service=%s, env=%s).", service_name, _ENVIRONMENT)


def get_tracer(name: str = "foundry"):
    """Return a tracer. No-op stub if OTel not configured."""
    try:
        from opentelemetry import trace  # type: ignore
        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


class _NoopSpan:
    def set_attribute(self, *_: object, **__: object) -> None:
        pass
    def record_exception(self, *_: object, **__: object) -> None:
        pass
    def set_status(self, *_: object, **__: object) -> None:
        pass
    def __enter__(self):
        return self
    def __exit__(self, *_: object) -> None:
        pass


class _NoopTracer:
    def start_as_current_span(self, name: str, **_: object):
        return _NoopSpan()
    def start_span(self, name: str, **_: object):
        return _NoopSpan()


def trace_agent(agent_name: str):
    """Decorator: wraps a sync agent function with an OTel span."""
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer(f"foundry.agents.{agent_name}")
            with tracer.start_as_current_span(f"{agent_name}.run") as span:
                # Extract chunk_id from state if present
                if args and isinstance(args[0], dict):
                    chunk_id = args[0].get("chunk", {}).get("id", "")
                    if chunk_id:
                        span.set_attribute("chunk_id", chunk_id[:8])
                    perspective = args[0].get("perspective", "")
                    if perspective:
                        span.set_attribute("perspective", perspective)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    raise
        return wrapper
    return decorator


def inject_trace_id(headers: dict) -> dict:
    """Inject W3C traceparent header for cross-service trace propagation."""
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.propagate import inject  # type: ignore
        inject(headers)
    except Exception:
        pass
    return headers
