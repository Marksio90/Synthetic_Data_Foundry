"""
agents/crawlers/dedup.py — 3-Stage Cross-Layer Deduplication Pipeline

Eliminates duplicates that escape per-layer URL dedup:
  Stage 1  URL normalization + SHA-256 hash  — O(1) exact match
  Stage 2  SimHash (64-bit) of title text    — Hamming ≤ 3 → near-duplicate
  Stage 3  OpenAI embedding cosine ≥ 0.92   — semantic duplicate (optional)

Stage 3 only runs when:
  - settings.openai_api_key is set
  - at least 2 candidates survived stage 1+2
  - enable_semantic=True (default)

All stages are additive — a source must FAIL a stage to be dropped.
Sources that fail only on URL (exact dup) are always dropped.
Sources with missing titles skip stage 2 (pass through).
Stage 3 skips gracefully on any API error.

Usage:
    pipeline = DedupPipeline()
    unique = await pipeline.filter(sources)

    # With custom thresholds:
    pipeline = DedupPipeline(
        simhash_threshold=4,
        semantic_threshold=0.90,
        enable_semantic=False,
    )
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    from agents.topic_scout import ScoutSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------

_NOISE_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "referrer", "source", "fbclid", "gclid", "msclkid",
    "mc_cid", "mc_eid", "_ga", "session_id",
})


def _normalise_url(url: str) -> str:
    """
    Normalise a URL for dedup purposes:
      - lowercase scheme + host
      - strip tracking query params
      - remove trailing slash from path
      - drop fragment
    """
    try:
        p = urlparse(url)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower().lstrip("www.")
        path = p.path.rstrip("/") or "/"
        # Filter query params
        if p.query:
            pairs = [
                kv for kv in p.query.split("&")
                if kv.split("=")[0] not in _NOISE_PARAMS
            ]
            query = "&".join(sorted(pairs))
        else:
            query = ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.lower().strip()


def _url_fingerprint(url: str) -> int:
    """SHA-256 of normalised URL → first 8 bytes as int."""
    norm = _normalise_url(url)
    digest = hashlib.sha256(norm.encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Stage 2 — SimHash (64-bit)
# ---------------------------------------------------------------------------

def _simhash(text: str, bits: int = 64) -> int:
    """
    Compute a Charikar SimHash fingerprint.

    Features: word unigrams + word bigrams (lowercased, stopwords kept).
    No char n-grams — they add too much noise for short academic titles.
    Near-duplicate titles (same paper, minor wording change) share Hamming ≤ 4.
    """
    if not text:
        return 0

    words = text.lower().split()
    features: list[str] = list(words)
    # Word bigrams give order-sensitive duplicate detection
    features.extend(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1))

    if not features:
        return 0

    v = [0] * bits
    for feat in features:
        h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def _hamming(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Stage 3 — Semantic cosine similarity
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity ∈ [-1, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _embed_texts(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Batch-embed texts via OpenAI text-embedding-3-small.
    Returns None for any text that fails. Never raises.
    """
    try:
        import openai
        from config.settings import settings
        if not settings.openai_api_key:
            return [None] * len(texts)
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        # Truncate to 512 chars — enough for title-level semantic comparison
        truncated = [t[:512] for t in texts]
        resp = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=truncated,
        )
        return [item.embedding for item in resp.data]
    except Exception as exc:
        logger.debug("[dedup] embedding API error: %s", exc)
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# DedupPipeline
# ---------------------------------------------------------------------------


class DedupPipeline:
    """
    3-stage deduplication pipeline for ScoutSource lists.

    Thread-safety: NOT thread-safe — use one instance per async task.
    """

    def __init__(
        self,
        simhash_threshold: int = 4,
        semantic_threshold: float = 0.92,
        enable_semantic: bool = True,
    ) -> None:
        self._simhash_threshold = simhash_threshold
        self._semantic_threshold = semantic_threshold
        self._enable_semantic = enable_semantic

        # Stage 1 index — set of URL fingerprints
        self._url_fps: set[int] = set()

        # Stage 2 index — list of (fingerprint, source_title) for Hamming scan
        self._sh_index: list[tuple[int, str]] = []

        # Stage 3 index — list of (title, embedding)
        self._embeddings: list[tuple[str, list[float]]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def filter(self, sources: "list[ScoutSource]") -> "list[ScoutSource]":
        """
        Return a deduplicated sublist.
        Sources are processed in order; earlier entries win.
        """
        unique: list[ScoutSource] = []
        pending_semantic: list[ScoutSource] = []

        for src in sources:
            # Stage 1 — URL exact dedup
            if self._is_url_dup(src.url):
                logger.debug("[dedup] URL dup: %s", src.url[:80])
                continue

            # Stage 2 — SimHash title dedup
            title = (src.title or "").strip()
            if title and self._is_simhash_dup(title):
                logger.debug("[dedup] SimHash dup title=%s", title[:60])
                continue

            unique.append(src)
            pending_semantic.append(src)

            # Register in stages 1 + 2 immediately (order matters)
            self._add_url(src.url)
            if title:
                self._add_simhash(title)

        # Stage 3 — Semantic dedup (batch)
        if self._enable_semantic and len(pending_semantic) >= 2:
            unique = await self._semantic_filter(unique)

        logger.debug(
            "[dedup] %d → %d sources (simhash_thresh=%d, semantic=%s)",
            len(sources), len(unique),
            self._simhash_threshold, self._enable_semantic,
        )
        return unique

    def reset(self) -> None:
        """Clear all indices (useful for tests)."""
        self._url_fps.clear()
        self._sh_index.clear()
        self._embeddings.clear()

    def stats(self) -> dict:
        return {
            "url_index_size":       len(self._url_fps),
            "simhash_index_size":   len(self._sh_index),
            "embedding_index_size": len(self._embeddings),
        }

    # ------------------------------------------------------------------
    # Stage 1 internals
    # ------------------------------------------------------------------

    def _is_url_dup(self, url: str) -> bool:
        return _url_fingerprint(url) in self._url_fps

    def _add_url(self, url: str) -> None:
        self._url_fps.add(_url_fingerprint(url))

    # ------------------------------------------------------------------
    # Stage 2 internals
    # ------------------------------------------------------------------

    def _is_simhash_dup(self, title: str) -> bool:
        fp = _simhash(title)
        return any(
            _hamming(fp, existing_fp) <= self._simhash_threshold
            for existing_fp, _ in self._sh_index
        )

    def _add_simhash(self, title: str) -> None:
        self._sh_index.append((_simhash(title), title))

    # ------------------------------------------------------------------
    # Stage 3 internals
    # ------------------------------------------------------------------

    async def _semantic_filter(
        self, sources: "list[ScoutSource]"
    ) -> "list[ScoutSource]":
        """
        Batch-embed all titles; drop any source whose embedding is too close
        to an already-accepted source.
        """
        titles = [(src.title or "").strip() for src in sources]
        embeddings = await _embed_texts(titles)

        accepted: list[ScoutSource] = []
        accepted_embeddings: list[list[float]] = []

        for src, emb in zip(sources, embeddings):
            if emb is None:
                # API failed for this one — keep it (conservative)
                accepted.append(src)
                continue

            is_dup = False
            for acc_emb in self._embeddings:
                sim = _cosine(emb, acc_emb[1])
                if sim >= self._semantic_threshold:
                    logger.debug(
                        "[dedup] semantic dup sim=%.3f: %s",
                        sim, (src.title or "")[:60],
                    )
                    is_dup = True
                    break

            if not is_dup:
                accepted.append(src)
                self._embeddings.append(((src.title or ""), emb))
                accepted_embeddings.append(emb)

        return accepted
