"""
agents/crawlers/verifier.py — 5-Point Anti-Hallucination & Verification Firewall

Every source passes through checks tiered by domain trust level:

  Check 1  Domain Trust Score   — whitelist tier S/A/B/C, always runs, never rejects
  Check 2  Temporal Validity    — rejects impossible future dates; tags post-cutoff models
  Check 3  HTTP Reachability    — HEAD + SSL + status < 400  (skipped for Tier S)
  Check 4  Cross-Source         — DOI resolve / Semantic Scholar / Wayback  (Tier C only)
  Check 5  Content Integrity    — error-pattern detection + text density   (Tier B/C)

Tier schedule:
  Tier S: checks 1+2 only  (arxiv.org, eur-lex, pubmed, sec.gov, …)
  Tier A: checks 1+2+3
  Tier B: checks 1+2+3+5
  Tier C: checks 1+2+3+4+5  (≥1 cross-source confirmation required)

Usage:
    result = await verify_source(source, http_client)
    batch  = await verify_batch(sources, http_client)
    # batch.verified  — list[ScoutSource] that passed
    # batch.post_cutoff_models  — union of post-cutoff LLMs across verified sources
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

import httpx

from agents.topic_scout import ScoutSource, _get_source_tier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM training cutoff dates
# ---------------------------------------------------------------------------

MODEL_CUTOFFS: dict[str, datetime] = {
    "gpt-4o":            datetime(2024, 4,  1, tzinfo=timezone.utc),
    "claude-3.5-sonnet": datetime(2024, 4,  1, tzinfo=timezone.utc),
    "llama-3":           datetime(2024, 3,  1, tzinfo=timezone.utc),
    "gemini-1.5":        datetime(2024, 2,  1, tzinfo=timezone.utc),
    "mistral":           datetime(2024, 3,  1, tzinfo=timezone.utc),
    "qwen2.5":           datetime(2024, 7,  1, tzinfo=timezone.utc),
    "phi-3":             datetime(2024, 3,  1, tzinfo=timezone.utc),
    "gemma-2":           datetime(2024, 6,  1, tzinfo=timezone.utc),
}

TIER_TRUST_SCORES: dict[str, float] = {
    "S": 1.0,
    "A": 0.8,
    "B": 0.6,
    "C": 0.3,
}

# ---------------------------------------------------------------------------
# Content-integrity error patterns (Check 5)
# ---------------------------------------------------------------------------

_ERROR_RE = re.compile(
    r"(?i)("
    r"404\s*not\s*found|page\s*not\s*found|"
    r"access\s*denied|403\s*forbidden|"
    r"captcha|verify\s*you\s*are\s*human|i\s*am\s*not\s*a\s*robot|"
    r"cloudflare\s*ray\s*id|ddos\s*protection|"
    r"this\s*page\s*(is\s*)?not\s*available|content\s*unavailable|"
    r"requires\s*subscription|purchase\s*access|pay\s*per\s*view|"
    r"javascript\s*(must\s*be\s*)?is\s*disabled|enable\s*javascript"
    r")"
)

_BINARY_EXTS = frozenset({".pdf", ".docx", ".xlsx", ".pptx", ".mp3", ".mp4"})

_DATE_FMTS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%Y",
)


def _parse_dt(date_str: str) -> Optional[datetime]:
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(date_str[:32].strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    passed: bool
    trust_score: float
    tier: str
    post_cutoff_models: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class BatchVerificationResult:
    verified: list[ScoutSource]
    post_cutoff_models: list[str]  # union across all verified sources


# ---------------------------------------------------------------------------
# Check 1 — Domain Trust Score
# ---------------------------------------------------------------------------


def _check_1_domain(source: ScoutSource) -> tuple[float, str]:
    """Derives authoritative tier from URL. Returns (trust_score, tier)."""
    tier = _get_source_tier(source.url)
    return TIER_TRUST_SCORES.get(tier, 0.3), tier


# ---------------------------------------------------------------------------
# Check 2 — Temporal Validity
# ---------------------------------------------------------------------------


def _check_2_temporal(
    source: ScoutSource,
) -> tuple[bool, list[str], list[str]]:
    """
    Returns (passed, failure_reasons, post_cutoff_models).
    Hard-fails only if the date is provably in the impossible future.
    """
    if not source.published_at:
        return True, [], []                # unknown date — allow

    dt = _parse_dt(source.published_at)
    if dt is None:
        return True, [], []                # unparseable — conservative allow

    max_future = datetime.now(timezone.utc) + timedelta(days=7)
    if dt > max_future:
        return False, [f"future_date:{dt.date().isoformat()}"], []

    post_cutoff = [
        model for model, cutoff in MODEL_CUTOFFS.items() if dt > cutoff
    ]
    return True, [], post_cutoff


# ---------------------------------------------------------------------------
# Check 3 — HTTP Reachability
# ---------------------------------------------------------------------------


async def _check_3_http(
    url: str, client: httpx.AsyncClient
) -> tuple[bool, str]:
    """
    HEAD → follow redirects → status < 400 → pass.
    Falls back to GET if HEAD returns 405.
    SSL errors counted as failure.
    """
    try:
        resp = await client.head(url, timeout=8.0, follow_redirects=True)
        if resp.status_code == 405:
            resp = await client.get(
                url, timeout=8.0, follow_redirects=True,
                headers={"Range": "bytes=0-1023"},
            )
        if resp.status_code >= 400:
            return False, f"http_{resp.status_code}"
        return True, ""
    except httpx.ConnectError as exc:
        err = str(exc).lower()
        label = "ssl_error" if ("ssl" in err or "certificate" in err) else "connect_error"
        return False, label
    except httpx.TimeoutException:
        return False, "http_timeout"
    except Exception as exc:
        return False, f"http_error:{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Check 4 — Cross-Source Verification  (Tier C only)
# ---------------------------------------------------------------------------


async def _check_4_cross_source(
    source: ScoutSource, client: httpx.AsyncClient
) -> tuple[bool, str]:
    """
    Three escalating strategies for Tier C sources:
      1. DOI resolution via doi.org
      2. Semantic Scholar title search
      3. Wayback Machine CDX archivability
    Returns True as soon as any strategy confirms existence.
    """
    url = source.url

    # Strategy 1: DOI URL resolves successfully
    if "doi.org/" in url or "/doi/" in url:
        try:
            resp = await client.head(url, timeout=8.0, follow_redirects=True)
            if resp.status_code < 400:
                return True, ""
        except Exception:
            pass

    # Strategy 2: Semantic Scholar title match
    title = (source.title or "").strip()
    if len(title) > 15:
        try:
            ss_url = (
                f"https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={quote_plus(title[:120])}&limit=3&fields=title"
            )
            resp = await client.get(ss_url, timeout=8.0)
            if resp.status_code == 200 and resp.json().get("data"):
                return True, ""
        except Exception:
            pass

    # Strategy 3a: Wayback Machine availability (requires recent snapshot)
    try:
        wb = f"https://archive.org/wayback/available?url={quote_plus(url)}"
        resp = await client.get(wb, timeout=8.0)
        if resp.status_code == 200:
            snap = resp.json().get("archived_snapshots", {}).get("closest", {})
            if snap.get("available"):
                return True, ""
    except Exception:
        pass

    # Strategy 3b: Wayback CDX — any historical crawl suffices (lower bar, catches new regulatory docs)
    try:
        cdx = f"https://web.archive.org/cdx/search/cdx?url={quote_plus(url)}&limit=1&output=json&fl=timestamp"
        resp = await client.get(cdx, timeout=8.0)
        if resp.status_code == 200:
            rows = resp.json()
            if rows and len(rows) > 1:  # row[0] is the header ["timestamp"]
                return True, ""
    except Exception:
        pass

    return False, "cross_verification_failed"


# ---------------------------------------------------------------------------
# Check 5 — Content Integrity  (Tier B and C)
# ---------------------------------------------------------------------------


async def _check_5_content(
    url: str, client: httpx.AsyncClient
) -> tuple[bool, str]:
    """
    Streams first 8 KB of HTML response and checks:
      - No error / CAPTCHA / cookie-wall patterns
      - Text-to-HTML ratio > 0.10
      - Extracted text > 100 chars
    Binary formats (PDF, DOCX…) are skipped — trust the HTTP check.
    """
    # Skip binary formats
    url_lower = url.lower().split("?")[0]
    if any(url_lower.endswith(ext) for ext in _BINARY_EXTS):
        return True, ""

    try:
        async with client.stream(
            "GET", url, timeout=12.0, follow_redirects=True
        ) as resp:
            if resp.status_code >= 400:
                return False, f"content_http_{resp.status_code}"

            ct = resp.headers.get("content-type", "")
            if any(t in ct for t in ("pdf", "octet-stream", "zip", "msword")):
                return True, ""

            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes(2048):
                chunks.append(chunk)
                total += len(chunk)
                if total >= 8192:
                    break

        html = b"".join(chunks).decode("utf-8", errors="replace")

        # Error-pattern check
        m = _ERROR_RE.search(html)
        if m:
            return False, f"error_pattern:{m.group(0)[:40].strip()}"

        # Text density
        stripped = re.sub(r"<[^>]+>", " ", html)
        clean = " ".join(stripped.split())

        if len(html) > 500:
            density = len(clean) / max(len(html), 1)
            if density < 0.10:
                return False, f"low_density:{density:.2f}"

        if len(clean) < 100:
            return False, "too_short"

        return True, ""

    except httpx.TimeoutException:
        return False, "content_timeout"
    except Exception as exc:
        logger.debug("Content check error %s: %s", url[:80], exc)
        return True, ""   # conservative: don't block on unexpected errors


# ---------------------------------------------------------------------------
# Main single-source verification
# ---------------------------------------------------------------------------


async def verify_source(
    source: ScoutSource,
    http_client: httpx.AsyncClient,
) -> VerificationResult:
    """
    Run the 5-point firewall for one ScoutSource.
    Tier determines which checks are applied (see module docstring).
    """
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    # Check 1 — always runs
    trust_score, tier = _check_1_domain(source)
    checks["domain_trust"] = True

    # Check 2 — temporal (hard fail on impossible future date)
    temp_ok, temp_reasons, post_cutoff = _check_2_temporal(source)
    checks["temporal"] = temp_ok
    reasons.extend(temp_reasons)
    if not temp_ok:
        _emit_metric(reasons, source.source_type)
        return VerificationResult(
            passed=False, trust_score=trust_score, tier=tier,
            checks=checks, failure_reasons=reasons,
        )

    # Check 3 — HTTP (Tier S trusts domain whitelist; skip)
    if tier == "S":
        checks["http"] = True
    else:
        ok, reason = await _check_3_http(source.url, http_client)
        checks["http"] = ok
        if not ok:
            reasons.append(reason)

    # Check 4 — cross-source (Tier C only; run even if HTTP failed)
    if tier == "C":
        ok, reason = await _check_4_cross_source(source, http_client)
        checks["cross_source"] = ok
        if not ok:
            reasons.append(reason)
    else:
        checks["cross_source"] = True

    # Check 5 — content integrity (Tier B/C, only when HTTP passed)
    if tier in ("B", "C") and checks.get("http", False):
        ok, reason = await _check_5_content(source.url, http_client)
        checks["content_integrity"] = ok
        if not ok:
            reasons.append(reason)
    else:
        checks["content_integrity"] = True

    passed = all(checks.values())
    if not passed:
        _emit_metric(reasons, source.source_type)

    return VerificationResult(
        passed=passed,
        trust_score=trust_score,
        tier=tier,
        post_cutoff_models=post_cutoff,
        checks=checks,
        failure_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------


async def verify_batch(
    sources: list[ScoutSource],
    http_client: httpx.AsyncClient,
    *,
    max_concurrent: int = 8,
) -> BatchVerificationResult:
    """
    Verify a batch of sources with bounded concurrency.
    Updates source.verified and source.source_tier in-place.
    Returns BatchVerificationResult with verified sources and union of post-cutoff models.
    """
    if not sources:
        return BatchVerificationResult(verified=[], post_cutoff_models=[])

    sem = asyncio.Semaphore(max_concurrent)

    async def _guarded(src: ScoutSource) -> tuple[ScoutSource, VerificationResult]:
        async with sem:
            result = await verify_source(src, http_client)
            return src, result

    pairs = await asyncio.gather(
        *[_guarded(s) for s in sources],
        return_exceptions=True,
    )

    verified: list[ScoutSource] = []
    all_post_cutoff: set[str] = set()

    for item in pairs:
        if isinstance(item, Exception):
            logger.debug("verify_batch exception: %s", item)
            continue
        src, result = item
        src.verified = result.passed
        src.source_tier = result.tier          # authoritative tier from URL
        if result.passed:
            verified.append(src)
            all_post_cutoff.update(result.post_cutoff_models)
        else:
            logger.debug(
                "[verifier] REJECTED %s tier=%s: %s",
                src.url[:80], result.tier, " | ".join(result.failure_reasons),
            )

    logger.info(
        "[verifier] %d/%d sources passed firewall",
        len(verified), len(sources),
    )
    return BatchVerificationResult(
        verified=verified,
        post_cutoff_models=sorted(all_post_cutoff),
    )


# ---------------------------------------------------------------------------
# Prometheus shim
# ---------------------------------------------------------------------------


def _emit_metric(reasons: list[str], source_type: str) -> None:
    try:
        from api.monitoring import _AVAILABLE, scout_verification_failures
        if not _AVAILABLE:
            return
        for reason in reasons:
            prefix = reason.split(":")[0]
            scout_verification_failures.labels(reason=prefix, source=source_type).inc()
    except Exception:
        pass
