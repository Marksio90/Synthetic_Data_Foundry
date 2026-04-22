"""
utils/llm_router.py — Unified LLM routing layer via LiteLLM Proxy.

Replaces direct OpenAI/Ollama clients with cost-aware routing:
  - Tier LOCAL  (foundry/local)   → Ollama qwen2.5:14b   ~80% calls
  - Tier MID    (foundry/mid)     → Claude Haiku 4.5      ~15% calls
  - Tier QUALITY(foundry/quality) → GPT-4o-mini            ~4% calls
  - Tier JUDGE  (foundry/judge)   → Claude Sonnet 4.6       ~1% calls

Falls back to direct OpenAI SDK when LiteLLM Proxy is not running.

Usage:
    from utils.llm_router import get_completion, get_embedding, LLMTier

    answer, cost = await get_completion(
        messages=[{"role": "user", "content": "..."}],
        tier=LLMTier.LOCAL,
    )
    vectors, cost = await get_embedding(texts=["doc1", "doc2"])
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from enum import Enum
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("foundry.llm_router")

_LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://litellm:4000")
_LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "foundry-dev-key")

# Costs per 1M tokens in USD (fallback when proxy not available)
_FALLBACK_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "text-embedding-3-small": (0.02, 0.0),
}


class LLMTier(str, Enum):
    LOCAL = "foundry/local"
    MID = "foundry/mid"
    QUALITY = "foundry/quality"
    JUDGE = "foundry/judge"
    VISION = "foundry/vision"
    AUTO = "foundry/auto"


# ---------------------------------------------------------------------------
# Proxy-aware async completion
# ---------------------------------------------------------------------------

_proxy_available: Optional[bool] = None
_proxy_check_time: float = 0.0
_PROXY_CHECK_INTERVAL = 60.0  # re-check every 60s


async def _check_proxy() -> bool:
    global _proxy_available, _proxy_check_time
    now = time.monotonic()
    if _proxy_available is not None and (now - _proxy_check_time) < _PROXY_CHECK_INTERVAL:
        return _proxy_available
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{_LITELLM_PROXY_URL}/health")
            _proxy_available = r.status_code == 200
    except Exception:
        _proxy_available = False
    _proxy_check_time = now
    if not _proxy_available:
        logger.debug("LiteLLM Proxy not reachable at %s — using direct SDK fallback.", _LITELLM_PROXY_URL)
    return _proxy_available


async def get_completion(
    messages: list[dict[str, str]],
    tier: LLMTier = LLMTier.LOCAL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs: Any,
) -> Tuple[str, float]:
    """
    Call LLM through proxy (preferred) or direct SDK (fallback).
    Returns (text_response, cost_cents).
    """
    if await _check_proxy():
        return await _completion_via_proxy(messages, tier.value, temperature, max_tokens)
    return await _completion_direct(messages, tier, temperature, max_tokens)


async def _completion_via_proxy(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, float]:
    import httpx

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {_LITELLM_MASTER_KEY}"}

    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(
            f"{_LITELLM_PROXY_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    # Proxy returns x-litellm-response-cost header
    cost_cents = float(data.get("_response_ms", 0)) * 0.0  # placeholder
    # More precise: use usage tokens × known cost
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    # Cost from proxy metadata if available
    cost_usd = data.get("usage", {}).get("_cost", 0.0) or 0.0
    cost_cents = cost_usd * 100

    return text, cost_cents


async def _completion_direct(
    messages: list[dict[str, str]],
    tier: LLMTier,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, float]:
    """Direct SDK fallback — maps tier → model string."""
    from config.settings import settings
    import openai

    _TIER_TO_MODEL = {
        LLMTier.LOCAL: settings.ollama_model or settings.openai_primary_model,
        LLMTier.MID: "claude-haiku-4-5-20251001",
        LLMTier.QUALITY: settings.openai_primary_model,
        LLMTier.JUDGE: settings.openai_fallback_model,
        LLMTier.VISION: "gpt-4o",
        LLMTier.AUTO: settings.ollama_model or settings.openai_primary_model,
    }

    model = _TIER_TO_MODEL[tier]
    cost_cents = 0.0

    # Try Ollama for LOCAL tier
    if tier == LLMTier.LOCAL and settings.ollama_model:
        try:
            client = openai.AsyncOpenAI(
                api_key="ollama",
                base_url=f"{settings.ollama_url.rstrip('/')}/v1",
                timeout=180.0,
            )
            resp = await client.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip(), 0.0
        except Exception as exc:
            logger.warning("Ollama direct call failed (%s) → escalating to OpenAI.", exc)
            model = settings.openai_primary_model

    # OpenAI / Anthropic fallback
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content.strip()

    if resp.usage and model in _FALLBACK_COSTS:
        c_in, c_out = _FALLBACK_COSTS[model]
        cost_cents = (
            (resp.usage.prompt_tokens / 1_000_000) * c_in * 100
            + (resp.usage.completion_tokens / 1_000_000) * c_out * 100
        )

    return text, cost_cents


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------

async def get_embedding(
    texts: List[str],
    model: str = "foundry/local-embed",
) -> Tuple[List[List[float]], float]:
    """
    Get embeddings for a list of texts.
    Returns (list_of_vectors, total_cost_cents).
    Uses proxy if available, falls back to direct OpenAI SDK.
    """
    if not texts:
        return [], 0.0

    if await _check_proxy():
        return await _embed_via_proxy(texts, model)

    loop = asyncio.get_event_loop()
    from agents.expert import embed_batch
    return await loop.run_in_executor(None, embed_batch, texts)


async def _embed_via_proxy(texts: List[str], model: str) -> Tuple[List[List[float]], float]:
    import httpx

    headers = {"Authorization": f"Bearer {_LITELLM_MASTER_KEY}"}
    all_vectors: List[List[float]] = []
    total_cost = 0.0
    batch_size = 2048

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = await client.post(
                f"{_LITELLM_PROXY_URL}/v1/embeddings",
                json={"model": model, "input": batch},
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_vectors.extend(item["embedding"] for item in sorted_data)
            cost_usd = data.get("usage", {}).get("_cost", 0.0) or 0.0
            total_cost += cost_usd * 100

    return all_vectors, total_cost
