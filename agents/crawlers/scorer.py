"""
agents/crawlers/scorer.py — 8-Component KNOWLEDGE_GAP_SCORE Calculator

KNOWLEDGE_GAP_SCORE = Σ(wi × componenti)  ∈ [0, 1]

  w1=0.25  temporal_gap          — post-cutoff recency across target LLMs
  w2=0.20  llm_uncertainty       — hedging rate from GPT-4o-mini + Ollama
  w3=0.15  cross_model_divergence— semantic distance between 3 model answers
  w4=0.15  citation_velocity     — Semantic Scholar weekly/quarterly growth
  w5=0.10  niche_penetration     — inverse of total paper saturation
  w6=0.08  source_authority      — mean trust score of verified sources
  w7=0.04  multilingual_gap      — Wikipedia language coverage inverse
  w8=0.03  format_diversity      — unique format types present

All components clamp to [0.0, 1.0]. Any component that fails (API down,
rate-limit, timeout) returns a neutral 0.5 and never crashes the pipeline.

Usage:
    ctx = ScorerContext(domain=..., sources=..., ...)
    result = await score_topic(ctx)
    # result.knowledge_gap_score  — final weighted score
    # result.components           — per-component breakdown for debugging
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights (from spec — adaptive calibration hooks in future)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "temporal_gap":           0.25,
    "llm_uncertainty":        0.20,
    "cross_model_divergence": 0.15,
    "citation_velocity":      0.15,
    "niche_penetration":      0.10,
    "source_authority":       0.08,
    "multilingual_gap":       0.04,
    "format_diversity":       0.03,
}

_TIER_SCORES = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.3}

# Total number of target LLMs tracked for temporal gap
_N_TARGET_MODELS = 8


# ---------------------------------------------------------------------------
# Input / Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScorerContext:
    """All data needed to compute the 8 knowledge-gap components."""
    domain: str                           # topic / query string
    sources: list                         # list[ScoutSource] — verified sources
    cutoff_model_targets: list[str]       # models for which sources are post-cutoff
    languages: list[str]                  # detected languages in sources
    format_types: list[str]               # format types present (pdf, html, video…)
    recency_score: float                  # pre-computed from publication dates
    base_uncertainty: float               # pre-computed by _probe_llm_uncertainty (GPT-only)
    http_client: httpx.AsyncClient


@dataclass
class ScoringResult:
    knowledge_gap_score: float
    llm_uncertainty: float
    citation_velocity: float
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _strip_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw).strip()


def _days_ago_str(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# w1 — temporal_gap
# Fraction of tracked LLMs for which sources are demonstrably post-cutoff,
# blended with recency score from publication dates.
# ---------------------------------------------------------------------------


def _w1_temporal_gap(ctx: ScorerContext) -> float:
    model_fraction = len(ctx.cutoff_model_targets) / _N_TARGET_MODELS
    # 60% from model coverage, 40% from raw recency
    return _clamp(0.60 * model_fraction + 0.40 * ctx.recency_score)


# ---------------------------------------------------------------------------
# w2 — llm_uncertainty
# Extends the GPT-4o-mini probe with an Ollama query when available.
# ---------------------------------------------------------------------------


async def _w2_llm_uncertainty(ctx: ScorerContext) -> float:
    base = ctx.base_uncertainty   # already computed, avoids duplicate GPT call

    # Extend with Ollama if configured
    ollama_score = await _query_ollama_uncertainty(ctx.domain, ctx.http_client)

    if ollama_score is not None:
        # Average both models for a more reliable estimate
        return _clamp((base + ollama_score) / 2.0)
    return _clamp(base)


async def _query_ollama_uncertainty(topic: str, client: httpx.AsyncClient) -> Optional[float]:
    """Query Ollama for confidence on topic. Returns uncertainty [0,1] or None."""
    try:
        from config.settings import settings
        if not settings.ollama_url:
            return None
        model = settings.ollama_model or "qwen2.5:14b"
        url = f"{settings.ollama_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rate your knowledge of the topic 0-10 "
                        "(10=comprehensive, 0=unknown). "
                        'Respond ONLY with valid JSON: {"confidence": <0-10>}'
                    ),
                },
                {"role": "user", "content": f"Topic: {topic}"},
            ],
            "stream": False,
        }
        resp = await client.post(url, json=payload, timeout=15.0)
        if resp.status_code != 200:
            return None
        content = resp.json().get("message", {}).get("content", "")
        data = json.loads(_strip_fences(content))
        confidence = float(data.get("confidence", 5)) / 10.0
        return _clamp(1.0 - confidence)
    except Exception as exc:
        logger.debug("[scorer/w2] Ollama unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# w3 — cross_model_divergence
# Ask 3 models a factual question about the topic, embed all 3 answers,
# compute mean pairwise cosine distance.
# High distance = models disagree = knowledge gap.
# ---------------------------------------------------------------------------


async def _w3_cross_model_divergence(ctx: ScorerContext) -> float:
    try:
        answers = await _gather_model_answers(ctx.domain, ctx.http_client)
        if len(answers) < 2:
            return 0.5   # not enough answers to compare

        embeddings = await _embed_texts(answers)
        if len(embeddings) < 2:
            return 0.5

        distances = _mean_pairwise_cosine_distance(embeddings)
        # Typical cosine distances for similar answers: 0.05–0.15
        # High divergence (models disagree): 0.25+
        # Normalize to [0, 1]: distance / 0.35 (cap)
        return _clamp(distances / 0.35)

    except Exception as exc:
        logger.debug("[scorer/w3] cross_model_divergence failed: %s", exc)
        return 0.5


async def _gather_model_answers(topic: str, client: httpx.AsyncClient) -> list[str]:
    """Get factual answers from GPT-4o-mini, Ollama (if available), fallback GPT-3.5."""
    from config.settings import settings
    import openai

    question = (
        f"In one sentence, state the most important recent development "
        f"regarding: {topic}"
    )

    async def _ask_openai(model: str) -> Optional[str]:
        try:
            oai = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            resp = await oai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.0,
                max_tokens=80,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    async def _ask_ollama() -> Optional[str]:
        try:
            model = settings.ollama_model or "qwen2.5:14b"
            url = f"{settings.ollama_url}/api/chat"
            resp = await client.post(
                url,
                json={"model": model,
                      "messages": [{"role": "user", "content": question}],
                      "stream": False},
                timeout=15.0,
            )
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return None

    # Run all in parallel
    results = await asyncio.gather(
        _ask_openai(settings.openai_primary_model),   # gpt-4o-mini
        _ask_openai(settings.openai_fallback_model),  # gpt-4o
        _ask_ollama(),
        return_exceptions=True,
    )
    return [r for r in results if isinstance(r, str) and r]


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI. Returns list of vectors."""
    try:
        from config.settings import settings
        import openai
        oai = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await oai.embeddings.create(
            input=texts[:3],
            model=settings.openai_embedding_model,
        )
        return [e.embedding for e in resp.data]
    except Exception as exc:
        logger.debug("[scorer/w3] embedding failed: %s", exc)
        return []


def _mean_pairwise_cosine_distance(vectors: list[list[float]]) -> float:
    """Compute mean pairwise cosine distance between all vector pairs."""
    try:
        import numpy as np
        arrs = [np.array(v) for v in vectors]
        distances: list[float] = []
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                a, b = arrs[i], arrs[j]
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                if denom > 1e-8:
                    cos_sim = float(np.dot(a, b) / denom)
                    distances.append(1.0 - cos_sim)
        return float(sum(distances) / len(distances)) if distances else 0.0
    except ImportError:
        # numpy not available — rough estimate from string comparison
        if len(vectors) < 2:
            return 0.0
        return 0.25   # neutral fallback


# ---------------------------------------------------------------------------
# w4 — citation_velocity
# Semantic Scholar: papers_this_week / (papers_last_quarter / 13)
# Ratio > 1.0 = accelerating; normalize to [0, 1].
# ---------------------------------------------------------------------------


async def _w4_citation_velocity(ctx: ScorerContext) -> float:
    try:
        q = quote_plus(ctx.domain[:100])
        today = _today_str()
        week_ago = _days_ago_str(7)
        quarter_ago = _days_ago_str(90)

        week_url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={q}&limit=1&fields=paperId"
            f"&publicationDateOrYear={week_ago}:{today}"
        )
        quarter_url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={q}&limit=1&fields=paperId"
            f"&publicationDateOrYear={quarter_ago}:{today}"
        )

        w_resp, q_resp = await asyncio.gather(
            ctx.http_client.get(week_url, timeout=8.0),
            ctx.http_client.get(quarter_url, timeout=8.0),
        )

        # SS rate-limited → don't penalise gap_score for external outage
        if w_resp.status_code == 429 or q_resp.status_code == 429:
            logger.debug("[scorer/w4] Semantic Scholar rate-limited (429) — returning neutral 0.5")
            return 0.5

        weekly = w_resp.json().get("total", 0) if w_resp.status_code == 200 else 0
        quarterly = q_resp.json().get("total", 0) if q_resp.status_code == 200 else 0

        if quarterly == 0:
            return 0.0 if weekly == 0 else 0.8   # newly trending

        # Expected weekly = quarterly / 13
        ratio = (weekly * 13) / quarterly
        # ratio 1.0 = stable, >3.0 = strongly trending
        velocity_score = _clamp((ratio - 1.0) / 4.0)
        return velocity_score

    except Exception as exc:
        logger.debug("[scorer/w4] citation_velocity failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# w5 — niche_penetration
# Total paper count on Semantic Scholar for query.
# Fewer papers = more niche = higher score.
# ---------------------------------------------------------------------------


async def _w5_niche_penetration(ctx: ScorerContext) -> float:
    try:
        q = quote_plus(ctx.domain[:100])
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={q}&limit=1&fields=paperId"
        )
        resp = await ctx.http_client.get(url, timeout=8.0)
        if resp.status_code != 200:
            return 0.5

        total = resp.json().get("total", 0)
        # log10 scale: 0 papers→1.0, 10→0.8, 100→0.6, 1000→0.4, 10000→0.2, 100k→0.0
        score = max(0.0, 1.0 - math.log10(max(1, total)) / 5.0)
        return _clamp(score)

    except Exception as exc:
        logger.debug("[scorer/w5] niche_penetration failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# w6 — source_authority
# PageRank-like: mean trust score of verified sources based on tier.
# ---------------------------------------------------------------------------


def _w6_source_authority(ctx: ScorerContext) -> float:
    if not ctx.sources:
        return 0.0
    tier_vals = [_TIER_SCORES.get(getattr(s, "source_tier", "C"), 0.3)
                 for s in ctx.sources]
    return _clamp(sum(tier_vals) / len(tier_vals))


# ---------------------------------------------------------------------------
# w7 — multilingual_gap
# Wikipedia language link count: fewer languages = larger gap.
# ---------------------------------------------------------------------------


async def _w7_multilingual_gap(ctx: ScorerContext) -> float:
    # Fast path: use detected languages from sources
    if ctx.languages:
        lang_count = len(set(ctx.languages))
        # <3 languages → high gap, 15+ → well covered
        return _clamp(1.0 - lang_count / 15.0)

    # Wikipedia API fallback: count language editions
    try:
        topic_slug = quote_plus(ctx.domain[:60])
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&titles={topic_slug}&prop=langlinks"
            f"&lllimit=max&format=json&redirects=1"
        )
        resp = await ctx.http_client.get(url, timeout=8.0)
        if resp.status_code == 200:
            pages = resp.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            # langlinks + 1 for English itself
            lang_count = len(page.get("langlinks", [])) + 1
            return _clamp(1.0 - lang_count / 15.0)
    except Exception as exc:
        logger.debug("[scorer/w7] multilingual_gap Wikipedia failed: %s", exc)

    return 0.5   # unknown coverage → neutral


# ---------------------------------------------------------------------------
# w8 — format_diversity
# More unique format types = richer knowledge base = higher score.
# (PDF + video + dataset > single HTML page)
# ---------------------------------------------------------------------------


def _w8_format_diversity(ctx: ScorerContext) -> float:
    unique = len(set(ctx.format_types)) if ctx.format_types else 1
    # 1 format→0.2, 3 formats→0.6, 5+ formats→1.0
    return _clamp(unique / 5.0)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


async def score_topic(ctx: ScorerContext) -> ScoringResult:
    """
    Compute the full 8-component KNOWLEDGE_GAP_SCORE for a topic.
    All components run in parallel; any failure returns 0.5 (neutral).
    Never raises — errors are logged at DEBUG level.
    """
    # Fast (sync) components run immediately; async ones run in parallel
    w1 = _w1_temporal_gap(ctx)
    w6 = _w6_source_authority(ctx)
    w8 = _w8_format_diversity(ctx)

    # Async components in parallel
    results = await asyncio.gather(
        _w2_llm_uncertainty(ctx),
        _w3_cross_model_divergence(ctx),
        _w4_citation_velocity(ctx),
        _w5_niche_penetration(ctx),
        _w7_multilingual_gap(ctx),
        return_exceptions=True,
    )

    def _safe(v, fallback: float = 0.5) -> float:
        return _clamp(v) if not isinstance(v, Exception) else fallback

    w2 = _safe(results[0])
    w3 = _safe(results[1])
    w4 = _safe(results[2])
    w5 = _safe(results[3])
    w7 = _safe(results[4])

    components = {
        "temporal_gap":           w1,
        "llm_uncertainty":        w2,
        "cross_model_divergence": w3,
        "citation_velocity":      w4,
        "niche_penetration":      w5,
        "source_authority":       w6,
        "multilingual_gap":       w7,
        "format_diversity":       w8,
    }

    gap_score = round(
        sum(WEIGHTS[k] * v for k, v in components.items()),
        4,
    )

    logger.info(
        "[scorer] domain=%r gap_score=%.3f components=%s",
        ctx.domain[:50],
        gap_score,
        {k: round(v, 3) for k, v in components.items()},
        extra={"domain": ctx.domain, "knowledge_gap_score": gap_score},
    )

    return ScoringResult(
        knowledge_gap_score=_clamp(gap_score),
        llm_uncertainty=w2,
        citation_velocity=w4,
        components=components,
    )
