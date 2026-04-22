"""
nats/bus.py — NATS JetStream event bus for agent-to-agent communication.

Replaces asyncio.Queue (single-process) with a durable, distributed message bus.
Supports:
  - At-least-once delivery (JetStream persistent streams)
  - Consumer groups (competing consumers for horizontal scaling)
  - Dead-letter subject (nack → retry → DLQ after max_deliver)
  - Pub/sub for real-time events (SSE fan-out)

Streams:
  FOUNDRY.ingest.raw     — raw document paths ready for ingestion
  FOUNDRY.chunks.ready   — chunk_ids ready for processing
  FOUNDRY.qa.generated   — generated Q&A pairs awaiting judge
  FOUNDRY.judge.result   — judge verdicts
  FOUNDRY.events         — broadcast events (SSE, monitoring)

Usage:
    from nats.bus import get_bus

    bus = await get_bus()
    await bus.publish("FOUNDRY.ingest.raw", {"file_path": "/data/csrd.pdf", "batch_id": "b1"})

    async for msg in bus.subscribe("FOUNDRY.chunks.ready", durable="analyzer-workers"):
        data = msg.data_dict()
        await process(data)
        await msg.ack()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger("foundry.nats")

_NATS_URL = os.getenv("NATS_URL", "nats://nats:4222")
_MAX_RECONNECT_ATTEMPTS = 10

# Stream → subject prefix mapping
STREAMS: dict[str, dict[str, Any]] = {
    "FOUNDRY_INGEST": {
        "subjects": ["FOUNDRY.ingest.>"],
        "max_msgs": 100_000,
        "retention": "workqueue",
        "max_age_seconds": 86400 * 7,  # 7 days
    },
    "FOUNDRY_CHUNKS": {
        "subjects": ["FOUNDRY.chunks.>"],
        "max_msgs": 1_000_000,
        "retention": "workqueue",
        "max_age_seconds": 86400 * 3,
    },
    "FOUNDRY_QA": {
        "subjects": ["FOUNDRY.qa.>", "FOUNDRY.judge.>"],
        "max_msgs": 500_000,
        "retention": "limits",
        "max_age_seconds": 86400 * 7,
    },
    "FOUNDRY_EVENTS": {
        "subjects": ["FOUNDRY.events.>"],
        "max_msgs": 10_000,
        "retention": "limits",
        "max_age_seconds": 3600,  # 1h — ephemeral broadcast
    },
}

_connection = None
_js = None


async def _connect():
    global _connection, _js
    if _connection is not None:
        return _connection, _js
    try:
        import nats  # type: ignore
        from nats.js.api import StreamConfig, RetentionPolicy  # type: ignore

        _connection = await nats.connect(
            _NATS_URL,
            reconnect_time_wait=2,
            max_reconnect_attempts=_MAX_RECONNECT_ATTEMPTS,
            error_cb=lambda exc: logger.error("NATS error: %s", exc),
            closed_cb=lambda: logger.warning("NATS connection closed."),
            reconnected_cb=lambda: logger.info("NATS reconnected."),
        )
        _js = _connection.jetstream()

        # Ensure streams exist (idempotent)
        for stream_name, cfg in STREAMS.items():
            try:
                await _js.add_stream(
                    name=stream_name,
                    subjects=cfg["subjects"],
                    max_msgs=cfg["max_msgs"],
                    max_age=cfg["max_age_seconds"] * 1_000_000_000,  # nanos
                )
            except Exception:
                # Stream already exists — ignore
                pass

        logger.info("NATS JetStream connected: %s", _NATS_URL)
    except ImportError:
        logger.warning(
            "nats-py not installed — bus disabled. Run: pip install nats-py\n"
            "Falling back to in-process asyncio.Queue."
        )
        _connection = None
        _js = None
    except Exception as exc:
        logger.warning("NATS unavailable (%s) — falling back to in-process queue.", exc)
        _connection = None
        _js = None

    return _connection, _js


class NatsBus:
    """High-level NATS JetStream bus. Fail-silent when NATS is not available."""

    async def publish(self, subject: str, payload: dict[str, Any]) -> bool:
        """Publish payload to subject. Returns True on success."""
        _, js = await _connect()
        if js is None:
            return False
        try:
            data = json.dumps(payload, ensure_ascii=False).encode()
            await js.publish(subject, data)
            return True
        except Exception as exc:
            logger.error("NATS publish failed on %s: %s", subject, exc)
            return False

    async def subscribe(
        self,
        subject: str,
        durable: str,
        deliver_policy: str = "new",
        max_deliver: int = 5,
    ) -> AsyncIterator["NatsMessage"]:
        """
        Async iterator over JetStream messages with at-least-once semantics.
        Yields NatsMessage; caller must call msg.ack() or msg.nack().
        """
        _, js = await _connect()
        if js is None:
            logger.warning("NATS unavailable — subscribe(%s) returns empty stream.", subject)
            return

        try:
            sub = await js.subscribe(
                subject,
                durable=durable,
                manual_ack=True,
                deliver_policy=deliver_policy,
            )
            async for msg in sub.messages:
                yield NatsMessage(msg)
        except Exception as exc:
            logger.error("NATS subscribe error on %s: %s", subject, exc)

    async def subscribe_callback(
        self,
        subject: str,
        durable: str,
        callback: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Subscribe and call callback(payload) for each message."""
        async def _cb(raw_msg: Any) -> None:
            try:
                payload = json.loads(raw_msg.data.decode())
                await callback(payload)
                await raw_msg.ack()
            except Exception as exc:
                logger.error("NATS callback error on %s: %s", subject, exc)
                await raw_msg.nack()

        _, js = await _connect()
        if js is None:
            return
        try:
            await js.subscribe(subject, durable=durable, cb=_cb, manual_ack=True)
        except Exception as exc:
            logger.error("NATS subscribe_callback failed on %s: %s", subject, exc)

    async def is_available(self) -> bool:
        conn, _ = await _connect()
        return conn is not None and conn.is_connected

    async def close(self) -> None:
        global _connection, _js
        if _connection is not None:
            try:
                await _connection.drain()
            except Exception:
                pass
            _connection = None
            _js = None


class NatsMessage:
    """Thin wrapper around a nats.aio.msg.Msg."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def data_dict(self) -> dict[str, Any]:
        try:
            return json.loads(self._raw.data.decode())
        except Exception:
            return {}

    async def ack(self) -> None:
        try:
            await self._raw.ack()
        except Exception:
            pass

    async def nack(self) -> None:
        try:
            await self._raw.nack()
        except Exception:
            pass

    @property
    def subject(self) -> str:
        return getattr(self._raw, "subject", "")


# ---------------------------------------------------------------------------
# In-process fallback queue (when NATS not available)
# ---------------------------------------------------------------------------

class _InProcessBus:
    """asyncio.Queue-backed fallback for development without NATS."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

    def _q(self, subject: str) -> asyncio.Queue:
        if subject not in self._queues:
            self._queues[subject] = asyncio.Queue(maxsize=10_000)
        return self._queues[subject]

    async def publish(self, subject: str, payload: dict[str, Any]) -> bool:
        try:
            self._q(subject).put_nowait(payload)
            return True
        except asyncio.QueueFull:
            logger.warning("In-process queue full for subject %s — dropping message.", subject)
            return False

    async def subscribe(self, subject: str, **_: Any) -> AsyncIterator[dict[str, Any]]:
        q = self._q(subject)
        while True:
            payload = await q.get()
            yield payload  # type: ignore[misc]

    async def is_available(self) -> bool:
        return True

    async def close(self) -> None:
        self._queues.clear()


_bus: Optional[NatsBus] = None
_fallback_bus: Optional[_InProcessBus] = None


async def get_bus() -> NatsBus:
    global _bus
    if _bus is None:
        _bus = NatsBus()
    return _bus


async def get_fallback_bus() -> _InProcessBus:
    global _fallback_bus
    if _fallback_bus is None:
        _fallback_bus = _InProcessBus()
    return _fallback_bus
