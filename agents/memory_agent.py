"""
agents/memory_agent.py — Memory Agent for short-term (Redis) and long-term (Qdrant) knowledge.

Architecture:
  Short-term: Redis (session context, 24h TTL, sub-millisecond reads)
  Long-term:  Qdrant (semantic vector search, cross-session knowledge)
  Structured: PostgreSQL (entity facts, contradiction detection)

Capabilities:
  - store_fact(fact, source, embedding)
  - retrieve_relevant(query, k, recency_weight)
  - detect_contradiction(new_fact, existing_facts)
  - summarize_session(session_id) → compressed context
  - build_knowledge_graph connections (stored as Qdrant payload)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("foundry.agents.memory")

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "foundry_knowledge")
_EMBEDDING_DIM = int(os.getenv("QDRANT_EMBEDDING_DIM", "1536"))
_SHORT_TERM_TTL = 86_400      # 24h
_LONG_TERM_SCORE_THRESHOLD = 0.72


@dataclass
class MemoryRecord:
    fact_id: str
    content: str
    source: str
    embedding: List[float]
    session_id: str
    domain: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    relevance_score: float = 0.0


# ---------------------------------------------------------------------------
# Short-term memory: Redis
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """Redis-backed session context store."""

    def __init__(self) -> None:
        self._client = None

    async def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as aioredis  # type: ignore
            self._client = aioredis.from_url(
                _REDIS_URL, encoding="utf-8", decode_responses=True, socket_timeout=2
            )
            await self._client.ping()
        except Exception as exc:
            logger.warning("ShortTermMemory: Redis unavailable (%s).", exc)
            self._client = None
        return self._client

    async def store(self, session_id: str, key: str, value: Any, ttl: int = _SHORT_TERM_TTL) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            redis_key = f"foundry:mem:{session_id}:{key}"
            await client.set(redis_key, json.dumps(value, ensure_ascii=False), ex=ttl)
        except Exception as exc:
            logger.debug("ShortTermMemory.store failed: %s", exc)

    async def retrieve(self, session_id: str, key: str) -> Optional[Any]:
        client = await self._get_client()
        if client is None:
            return None
        try:
            redis_key = f"foundry:mem:{session_id}:{key}"
            raw = await client.get(redis_key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def get_session_context(self, session_id: str) -> dict[str, Any]:
        client = await self._get_client()
        if client is None:
            return {}
        try:
            keys = await client.keys(f"foundry:mem:{session_id}:*")
            if not keys:
                return {}
            values = await client.mget(*keys)
            result = {}
            for k, v in zip(keys, values):
                short_key = k.split(":")[-1]
                if v:
                    try:
                        result[short_key] = json.loads(v)
                    except Exception:
                        result[short_key] = v
            return result
        except Exception:
            return {}

    async def clear_session(self, session_id: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            keys = await client.keys(f"foundry:mem:{session_id}:*")
            if keys:
                await client.delete(*keys)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Long-term memory: Qdrant
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Qdrant-backed semantic knowledge store."""

    def __init__(self) -> None:
        self._client = None
        self._collection_ready = False

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            self._client = QdrantClient(url=_QDRANT_URL, timeout=10)

            # Ensure collection exists
            collections = [c.name for c in self._client.get_collections().collections]
            if _COLLECTION_NAME not in collections:
                self._client.create_collection(
                    collection_name=_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=_EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Qdrant collection '%s' created.", _COLLECTION_NAME)
            self._collection_ready = True
        except ImportError:
            logger.warning("qdrant-client not installed — long-term memory disabled. pip install qdrant-client")
            self._client = None
        except Exception as exc:
            logger.warning("Qdrant unavailable (%s) — long-term memory disabled.", exc)
            self._client = None
        return self._client

    def store(self, record: MemoryRecord) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            from qdrant_client.models import PointStruct  # type: ignore
            point = PointStruct(
                id=int(hashlib.md5(record.fact_id.encode()).hexdigest()[:8], 16),
                vector=record.embedding,
                payload={
                    "fact_id": record.fact_id,
                    "content": record.content,
                    "source": record.source,
                    "session_id": record.session_id,
                    "domain": record.domain,
                    "tags": record.tags,
                    "created_at": record.created_at,
                },
            )
            client.upsert(collection_name=_COLLECTION_NAME, points=[point])
            return True
        except Exception as exc:
            logger.debug("LongTermMemory.store failed: %s", exc)
            return False

    def search(
        self,
        query_vector: List[float],
        k: int = 8,
        domain_filter: Optional[str] = None,
        score_threshold: float = _LONG_TERM_SCORE_THRESHOLD,
    ) -> List[MemoryRecord]:
        client = self._get_client()
        if client is None:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

            query_filter = None
            if domain_filter:
                query_filter = Filter(
                    must=[FieldCondition(key="domain", match=MatchValue(value=domain_filter))]
                )

            results = client.search(
                collection_name=_COLLECTION_NAME,
                query_vector=query_vector,
                limit=k,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )
            return [
                MemoryRecord(
                    fact_id=r.payload.get("fact_id", ""),
                    content=r.payload.get("content", ""),
                    source=r.payload.get("source", ""),
                    embedding=[],
                    session_id=r.payload.get("session_id", ""),
                    domain=r.payload.get("domain", ""),
                    tags=r.payload.get("tags", []),
                    created_at=r.payload.get("created_at", 0.0),
                    relevance_score=r.score,
                )
                for r in results
            ]
        except Exception as exc:
            logger.debug("LongTermMemory.search failed: %s", exc)
            return []

    def detect_contradiction(
        self, new_content: str, new_vector: List[float], threshold: float = 0.90
    ) -> Tuple[bool, List[str]]:
        """
        Check if new fact contradicts existing knowledge (high similarity + negation indicators).
        Returns (has_contradiction, list_of_conflicting_fact_ids).
        """
        similar = self.search(new_vector, k=5, score_threshold=threshold)
        contradicting = []
        negation_words = {"nie", "brak", "zakaz", "wyłączony", "niezgodny", "not", "no", "without"}
        new_has_neg = any(w in new_content.lower().split() for w in negation_words)
        for rec in similar:
            rec_has_neg = any(w in rec.content.lower().split() for w in negation_words)
            if new_has_neg != rec_has_neg:
                contradicting.append(rec.fact_id)
        return bool(contradicting), contradicting


# ---------------------------------------------------------------------------
# Memory Agent (unified interface)
# ---------------------------------------------------------------------------

class MemoryAgent:
    """
    Unified interface for both short-term (Redis) and long-term (Qdrant) memory.
    Used by Analyzer, Synthesizer, and Critic agents to maintain context.
    """

    def __init__(self) -> None:
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    async def store_fact(
        self,
        content: str,
        source: str,
        embedding: List[float],
        session_id: str,
        domain: str = "esg",
        tags: Optional[list[str]] = None,
    ) -> str:
        fact_id = hashlib.sha256(f"{content}{source}".encode()).hexdigest()[:16]

        # Check for contradictions before storing
        has_conflict, conflicts = self.long_term.detect_contradiction(content, embedding)
        if has_conflict:
            logger.warning(
                "MemoryAgent: contradiction detected for fact_id=%s — conflicts with %s",
                fact_id,
                conflicts,
            )

        record = MemoryRecord(
            fact_id=fact_id,
            content=content,
            source=source,
            embedding=embedding,
            session_id=session_id,
            domain=domain,
            tags=tags or [],
        )

        # Store in both tiers
        self.long_term.store(record)
        await self.short_term.store(session_id, f"fact:{fact_id}", record.content)

        return fact_id

    async def retrieve_relevant(
        self,
        query_vector: List[float],
        session_id: str,
        k: int = 8,
        domain: Optional[str] = None,
        recency_weight: float = 0.2,
    ) -> List[MemoryRecord]:
        """
        Hybrid retrieval: long-term semantic search + recency boost from short-term.
        recency_weight: how much to boost recently stored facts (0 = pure similarity).
        """
        results = self.long_term.search(query_vector, k=k * 2, domain_filter=domain)

        if recency_weight > 0 and results:
            now = time.time()
            max_age = max((now - r.created_at) for r in results) or 1.0
            for r in results:
                age_factor = 1.0 - ((now - r.created_at) / max_age)
                r.relevance_score = (1 - recency_weight) * r.relevance_score + recency_weight * age_factor

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:k]

    async def summarize_session(self, session_id: str, max_chars: int = 2000) -> str:
        """Compress session context into a brief summary for multi-turn conversations."""
        ctx = await self.short_term.get_session_context(session_id)
        if not ctx:
            return ""

        facts = [v for k, v in ctx.items() if k.startswith("fact:") and isinstance(v, str)]
        if not facts:
            return ""

        summary = " | ".join(facts)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."
        return summary

    async def clear_session(self, session_id: str) -> None:
        await self.short_term.clear_session(session_id)


# Module-level singleton
_memory_agent: Optional[MemoryAgent] = None


def get_memory_agent() -> MemoryAgent:
    global _memory_agent
    if _memory_agent is None:
        _memory_agent = MemoryAgent()
    return _memory_agent
