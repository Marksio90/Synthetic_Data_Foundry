"""
api/redis_state.py — Redis-backed run state cache for cross-replica consistency.

Architecture:
  - Redis hash per run_id (primary, fast reads <1ms, TTL-based eviction)
  - PostgreSQL (secondary durability, survives Redis restart)
  - asyncio.Queue per SSE subscriber (local fan-out via Redis pub/sub)

Usage:
  from api.redis_state import get_redis_state
  rs = await get_redis_state()
  await rs.set_run(run_id, payload)
  payload = await rs.get_run(run_id)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("foundry.redis_state")

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
_RUN_TTL_SECONDS = 86_400       # 24h — runs older than this are evicted
_SCOUT_TTL_SECONDS = 259_200    # 72h
_KEY_PREFIX_RUN = "foundry:run:"
_KEY_PREFIX_SCOUT = "foundry:scout:"
_KEY_PREFIX_TOPIC = "foundry:topic:"
_PUBSUB_CHANNEL = "foundry:events"

_redis_client = None


async def _get_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as aioredis  # type: ignore
        _redis_client = aioredis.from_url(
            _REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await _redis_client.ping()
        logger.info("Redis state layer connected: %s", _REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — state layer disabled, falling back to PostgreSQL.", exc)
        _redis_client = None
    return _redis_client


class RedisStateLayer:
    """Async Redis operations for run + scout state. Fail-silent — never raises."""

    async def set_run(self, run_id: str, payload: dict[str, Any]) -> None:
        client = await _get_client()
        if client is None:
            return
        try:
            key = f"{_KEY_PREFIX_RUN}{run_id}"
            await client.set(key, json.dumps(payload, ensure_ascii=False), ex=_RUN_TTL_SECONDS)
        except Exception as exc:
            logger.debug("Redis set_run failed for %s: %s", run_id, exc)

    async def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        client = await _get_client()
        if client is None:
            return None
        try:
            raw = await client.get(f"{_KEY_PREFIX_RUN}{run_id}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("Redis get_run failed for %s: %s", run_id, exc)
            return None

    async def delete_run(self, run_id: str) -> None:
        client = await _get_client()
        if client is None:
            return
        try:
            await client.delete(f"{_KEY_PREFIX_RUN}{run_id}")
        except Exception:
            pass

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        client = await _get_client()
        if client is None:
            return []
        try:
            keys = await client.keys(f"{_KEY_PREFIX_RUN}*")
            if not keys:
                return []
            raw_values = await client.mget(*keys[:limit])
            result = []
            for raw in raw_values:
                if raw:
                    try:
                        result.append(json.loads(raw))
                    except Exception:
                        pass
            return sorted(result, key=lambda r: r.get("started_at", 0), reverse=True)
        except Exception as exc:
            logger.debug("Redis list_runs failed: %s", exc)
            return []

    async def set_scout(self, scout_id: str, payload: dict[str, Any]) -> None:
        client = await _get_client()
        if client is None:
            return
        try:
            key = f"{_KEY_PREFIX_SCOUT}{scout_id}"
            await client.set(key, json.dumps(payload, ensure_ascii=False), ex=_SCOUT_TTL_SECONDS)
        except Exception as exc:
            logger.debug("Redis set_scout failed for %s: %s", scout_id, exc)

    async def get_scout(self, scout_id: str) -> Optional[dict[str, Any]]:
        client = await _get_client()
        if client is None:
            return None
        try:
            raw = await client.get(f"{_KEY_PREFIX_SCOUT}{scout_id}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("Redis get_scout failed for %s: %s", scout_id, exc)
            return None

    async def set_topic(self, topic_id: str, payload: dict[str, Any]) -> None:
        client = await _get_client()
        if client is None:
            return
        try:
            key = f"{_KEY_PREFIX_TOPIC}{topic_id}"
            await client.set(key, json.dumps(payload, ensure_ascii=False), ex=_SCOUT_TTL_SECONDS)
        except Exception:
            pass

    async def get_topic(self, topic_id: str) -> Optional[dict[str, Any]]:
        client = await _get_client()
        if client is None:
            return None
        try:
            raw = await client.get(f"{_KEY_PREFIX_TOPIC}{topic_id}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def list_topics(self, limit: int = 200) -> list[dict[str, Any]]:
        client = await _get_client()
        if client is None:
            return []
        try:
            keys = await client.keys(f"{_KEY_PREFIX_TOPIC}*")
            if not keys:
                return []
            raw_values = await client.mget(*keys[:limit])
            result = []
            for raw in raw_values:
                if raw:
                    try:
                        result.append(json.loads(raw))
                    except Exception:
                        pass
            return result
        except Exception:
            return []

    async def publish_event(self, event: dict[str, Any]) -> None:
        """Publish to Redis pub/sub for cross-replica SSE fan-out."""
        client = await _get_client()
        if client is None:
            return
        try:
            await client.publish(_PUBSUB_CHANNEL, json.dumps(event, ensure_ascii=False))
        except Exception:
            pass

    async def is_available(self) -> bool:
        client = await _get_client()
        if client is None:
            return False
        try:
            await client.ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        global _redis_client
        if _redis_client is not None:
            try:
                await _redis_client.aclose()
            except Exception:
                pass
            _redis_client = None


_layer: Optional[RedisStateLayer] = None


async def get_redis_state() -> RedisStateLayer:
    global _layer
    if _layer is None:
        _layer = RedisStateLayer()
    return _layer
