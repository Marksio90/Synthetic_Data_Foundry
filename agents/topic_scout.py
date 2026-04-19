"""
agents/topic_scout.py — Automated Knowledge Gap Discovery Agent

3 parallel discovery methods:
  1. Source crawling — EUR-Lex, arXiv, OpenAlex, HackerNews
  2. LLM uncertainty probing — measures GPT hedging rate per topic
  3. Cutoff gap analysis — favours content published after LLM training cutoffs

Anti-hallucination guarantees:
  - Every URL verified via HTTP HEAD against a trusted-domain whitelist
  - Publication dates taken directly from API metadata (never inferred)
  - Topics with zero verified sources are silently dropped
  - No LLM is asked to generate source URLs

Scoring:
  score = 0.40×recency + 0.30×llm_uncertainty + 0.20×source_density + 0.10×social_signal

Public function:
    run_scout(domains, max_topics, progress_callback) → list[ScoutTopicData]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx
import openai

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: feedparser (RSS/Atom parsing for EUR-Lex + arXiv)
# ---------------------------------------------------------------------------

try:
    import feedparser as _feedparser  # type: ignore
    _HAS_FEEDPARSER = True
except ImportError:
    _feedparser = None  # type: ignore
    _HAS_FEEDPARSER = False
    logger.warning("feedparser not installed — RSS crawlers (arXiv, EUR-Lex) disabled")

# ---------------------------------------------------------------------------
# Trusted-domain whitelist  (anti-hallucination firewall)
# ---------------------------------------------------------------------------

# Domain trust tiers for verification firewall (Part 5, Step 4)
_TIER_S_DOMAINS = frozenset({
    "arxiv.org", "export.arxiv.org",
    "eur-lex.europa.eu", "ec.europa.eu", "publications.europa.eu",
    "pubmed.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov",
    "sec.gov", "wipo.int", "who.int", "imf.org",
    "federalregister.gov", "biorxiv.org", "medrxiv.org",
    "semanticscholar.org", "api.semanticscholar.org",
})

_TIER_A_DOMAINS = frozenset({
    "openalex.org", "api.openalex.org",
    "europepmc.org", "core.ac.uk",
    "ieee.org", "ieeexplore.ieee.org",
    "ssrn.com", "philpapers.org",
    "worldbank.org", "data.worldbank.org",
    "oecd.org", "oecd-ilibrary.org",
})

_TIER_B_DOMAINS = frozenset({
    "reuters.com", "ft.com", "bloomberg.com",
    "nature.com", "science.org",
    "bbc.com", "bbc.co.uk",
    "news.ycombinator.com", "hn.algolia.com",
    "github.com",
    "paperswithcode.com",
})

_TRUSTED_DOMAINS = frozenset({
    "eur-lex.europa.eu",
    "ec.europa.eu",
    "publications.europa.eu",
    "arxiv.org",
    "export.arxiv.org",
    "openalex.org",
    "api.openalex.org",
    "semanticscholar.org",
    "news.ycombinator.com",
    "hn.algolia.com",
    "youtube.com",
    "youtu.be",
    "pubmed.ncbi.nlm.nih.gov",
    "ssrn.com",
    "bbc.com",
    "reuters.com",
    "ft.com",
    "bloomberg.com",
})

# ---------------------------------------------------------------------------
# Domain tier resolver (used in verification firewall)
# ---------------------------------------------------------------------------


def _get_source_tier(url: str) -> str:
    """Return S/A/B/C trust tier for a URL based on its domain."""
    try:
        host = urlparse(url).netloc.lstrip("www.")
    except Exception:
        return "C"
    if any(host == d or host.endswith("." + d) for d in _TIER_S_DOMAINS):
        return "S"
    if any(host == d or host.endswith("." + d) for d in _TIER_A_DOMAINS):
        return "A"
    # .edu / .gov / .ac.uk automatically get tier A
    if host.endswith(".edu") or host.endswith(".gov") or host.endswith(".ac.uk"):
        return "A"
    if any(host == d or host.endswith("." + d) for d in _TIER_B_DOMAINS):
        return "B"
    return "C"


_TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3}


def _best_tier(tiers: list[str]) -> str:
    """Return the highest-quality tier from a list ('S' beats 'A' beats 'B' beats 'C')."""
    if not tiers:
        return "C"
    return min(tiers, key=lambda t: _TIER_ORDER.get(t, 3))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScoutSource:
    url: str
    title: str
    published_at: str        # ISO-8601 from source metadata, "" if unknown
    source_type: str         # eurlex | arxiv | openalex | hackernews | ...
    verified: bool = False
    # --- new fields (backwards-compatible, all have defaults) ---
    source_tier: str = "C"   # S | A | B | C  (domain trust level)
    language: str = ""       # BCP-47 code detected from content, "" = unknown
    snippet: str = ""        # up to 500-char content preview (no full text stored)


@dataclass
class ScoutTopicData:
    topic_id: str
    title: str
    summary: str
    score: float
    recency_score: float
    llm_uncertainty: float
    source_count: int
    social_signal: float
    sources: list[ScoutSource]
    domains: list[str]
    discovered_at: str       # ISO-8601
    # --- new fields (backwards-compatible, all have defaults) ---
    knowledge_gap_score: float = 0.0          # 8-component unified score (Part 2)
    cutoff_model_targets: list[str] = field(default_factory=list)  # ["gpt-4o", "claude-3.5", ...]
    format_types: list[str] = field(default_factory=list)          # ["pdf", "html", "video", ...]
    languages: list[str] = field(default_factory=list)             # detected languages in sources
    citation_velocity: float = 0.0            # Semantic Scholar weekly/quarterly ratio
    source_tier: str = "C"                    # best tier across all verified sources
    estimated_tokens: int = 0                 # rough token budget for ingest
    ingest_ready: bool = False                # True when ≥1 verified source present


# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------

_HTTP = httpx.AsyncClient(
    timeout=20.0,
    follow_redirects=True,
    headers={"User-Agent": "FoundryScout/1.0 (research-tool; non-commercial)"},
)

_OPENAI = openai.AsyncOpenAI(api_key=settings.openai_api_key)


def _strip_json_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?|```", "", raw).strip()


# ---------------------------------------------------------------------------
# URL verification
# ---------------------------------------------------------------------------


async def _verify_url(url: str) -> bool:
    """Return True only when URL is reachable AND its domain is whitelisted."""
    if not url:
        return False
    try:
        host = urlparse(url).netloc.lstrip("www.")
        if not any(host == td or host.endswith("." + td) for td in _TRUSTED_DOMAINS):
            return False
        resp = await _HTTP.head(url, timeout=8.0)
        return resp.status_code < 400
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Source crawlers
# ---------------------------------------------------------------------------


async def _fetch_arxiv(query: str) -> list[ScoutSource]:
    """arXiv Atom feed — no API key needed."""
    sources: list[ScoutSource] = []
    if not _HAS_FEEDPARSER:
        return sources
    try:
        url = (
            "https://export.arxiv.org/api/query"
            f"?search_query=all:{query.replace(' ', '+')}"
            "&sortBy=submittedDate&sortOrder=descending&max_results=8"
        )
        resp = await _HTTP.get(url)
        if resp.status_code != 200:
            return sources
        feed = _feedparser.parse(resp.text)
        for entry in feed.entries[:5]:
            link = entry.get("link", "")
            if not link:
                for lnk in entry.get("links", []):
                    if lnk.get("type") == "text/html":
                        link = lnk.get("href", "")
                        break
            if not link or "arxiv.org" not in link:
                continue
            # arxiv.org domain verified above, URL format checked — mark verified
            sources.append(ScoutSource(
                url=link,
                title=entry.get("title", "").replace("\n", " ").strip(),
                published_at=entry.get("published", ""),
                source_type="arxiv",
                verified=True,
            ))
    except Exception as exc:
        logger.warning("arXiv crawler error: %s", exc)
    return sources


async def _fetch_openalex(query: str) -> list[ScoutSource]:
    """OpenAlex REST API — free, no key. Sources marked unverified for later HTTP check."""
    sources: list[ScoutSource] = []
    try:
        url = (
            "https://api.openalex.org/works"
            f"?search={query.replace(' ', '%20')}"
            "&sort=publication_date:desc"
            "&filter=publication_year:>2023"
            "&per-page=8"
            "&select=id,title,doi,publication_date,open_access"
        )
        resp = await _HTTP.get(url)
        if resp.status_code != 200:
            return sources
        data = resp.json()
        for work in data.get("results", [])[:5]:
            oa_url = (work.get("open_access") or {}).get("oa_url") or work.get("doi") or ""
            if not oa_url:
                continue
            # oa_url may be from any external domain — mark unverified for HTTP HEAD check
            sources.append(ScoutSource(
                url=oa_url,
                title=(work.get("title") or "").strip(),
                published_at=work.get("publication_date") or "",
                source_type="openalex",
                verified=False,
            ))
    except Exception as exc:
        logger.warning("OpenAlex crawler error: %s", exc)
    return sources


async def _fetch_hackernews(query: str) -> list[ScoutSource]:
    """HackerNews via Algolia API — free, no key, real-time."""
    sources: list[ScoutSource] = []
    try:
        url = (
            "https://hn.algolia.com/api/v1/search"
            f"?query={query.replace(' ', '%20')}"
            "&tags=story"
            "&numericFilters=created_at_i%3E1704067200"
            "&hitsPerPage=8"
        )
        resp = await _HTTP.get(url)
        if resp.status_code != 200:
            return sources
        data = resp.json()
        for hit in data.get("hits", [])[:5]:
            hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            pub_ts = hit.get("created_at_i", 0)
            pub = (
                datetime.fromtimestamp(pub_ts, tz=timezone.utc).isoformat()
                if pub_ts else ""
            )
            # URL constructed from Algolia API — always valid news.ycombinator.com links
            sources.append(ScoutSource(
                url=hn_url,
                title=(hit.get("title") or "").strip(),
                published_at=pub,
                source_type="hackernews",
                verified=True,
            ))
    except Exception as exc:
        logger.warning("HackerNews crawler error: %s", exc)
    return sources


async def _fetch_eurlex(query: str) -> list[ScoutSource]:
    """EUR-Lex RSS search — EU legislative documents."""
    sources: list[ScoutSource] = []
    if not _HAS_FEEDPARSER:
        return sources
    try:
        url = (
            "https://eur-lex.europa.eu/search.html"
            f"?type=quick&lang=EN&text={query.replace(' ', '+')}"
            "&scope=EURLEX&FM_CODED=REG%2CDIR"
            "&SORT=VALIDITY_DATE%3Adesc&HITS_PER_PAGE=10&RSS=true"
        )
        resp = await _HTTP.get(url, timeout=15.0)
        if resp.status_code != 200:
            return sources

        feed = _feedparser.parse(resp.text)
        entries = feed.entries[:5]
        verify_results = await asyncio.gather(
            *[_verify_url(e.get("link", "")) for e in entries],
            return_exceptions=True,
        )
        for entry, ok in zip(entries, verify_results):
            if ok is not True:
                continue
            sources.append(ScoutSource(
                url=entry.get("link", ""),
                title=(entry.get("title") or "").strip(),
                published_at=entry.get("published", entry.get("updated", "")),
                source_type="eurlex",
                verified=True,
            ))
    except Exception as exc:
        logger.warning("EUR-Lex crawler error: %s", exc)
    return sources


# ---------------------------------------------------------------------------
# LLM uncertainty probing
# ---------------------------------------------------------------------------


async def _probe_llm_uncertainty(topic: str) -> float:
    """
    Ask gpt-4o-mini how confident it is about this topic.
    Returns uncertainty 0.0 (model knows well) → 1.0 (model doesn't know).
    """
    try:
        resp = await _OPENAI.chat.completions.create(
            model=settings.openai_primary_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rate your knowledge of the following topic on a scale 0-10 where "
                        "10 = comprehensive up-to-date knowledge, 0 = unknown / outside training data. "
                        'Respond ONLY with valid JSON: {"confidence": <0-10>}'
                    ),
                },
                {"role": "user", "content": f"Topic: {topic}"},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        raw = _strip_json_fences(resp.choices[0].message.content.strip())
        confidence = float(json.loads(raw).get("confidence", 5)) / 10.0
        return round(max(0.0, min(1.0, 1.0 - confidence)), 3)
    except Exception as exc:
        logger.debug("LLM uncertainty probe failed for '%s': %s", topic, exc)
        return 0.5


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
)


def _parse_date(date_str: str) -> Optional[datetime]:
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str[:30], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _compute_recency(sources: list[ScoutSource]) -> float:
    if not sources:
        return 0.0
    now = datetime.now(tz=timezone.utc)
    scores = []
    for src in sources:
        if not src.published_at:
            continue
        dt = _parse_date(src.published_at)
        if dt is None:
            continue
        age_days = max(0, (now - dt).days)
        scores.append(max(0.0, 1.0 - age_days / 365.0))
    return round(max(scores), 3) if scores else 0.3


# ---------------------------------------------------------------------------
# Domain auto-selection
# ---------------------------------------------------------------------------

_CANDIDATE_DOMAINS = [
    "CSRD corporate sustainability reporting directive",
    "EU AI Act compliance obligations",
    "SFDR sustainable finance disclosure regulation",
    "carbon border adjustment mechanism CBAM",
    "DORA digital operational resilience act financial sector",
    "EU taxonomy green finance technical screening criteria",
    "corporate sustainability due diligence CSDDD",
    "climate risk financial reporting TCFD ISSB",
    "ESG rating providers regulation EU",
    "net zero transition planning financial institutions",
    "TNFD biodiversity nature-related financial disclosure",
    "ISSB IFRS S1 S2 sustainability accounting standards",
    "SEC climate disclosure rules scope 3 emissions",
    "supply chain ESG due diligence regulation",
    "EU hydrogen strategy renewable energy directive",
    "greenwashing enforcement ESG claims regulation",
]


async def _select_domains(n: int = 6) -> list[str]:
    """Use the LLM to pick the domains with the most likely knowledge gaps."""
    try:
        resp = await _OPENAI.chat.completions.create(
            model=settings.openai_primary_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ESG and regulatory compliance expert. "
                        f"Select the {n} domains most likely to have significant "
                        "recent developments (post-2023) that AI models would not know about. "
                        "Return ONLY a JSON array of strings, no other text. "
                        "Domains to choose from:\n"
                        + "\n".join(f"- {d}" for d in _CANDIDATE_DOMAINS)
                    ),
                },
                {"role": "user", "content": f"Select {n} domains with the highest knowledge gaps."},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = _strip_json_fences(resp.choices[0].message.content.strip())
        selected = json.loads(raw)
        if isinstance(selected, list) and selected:
            return [str(d) for d in selected[:n]]
    except Exception as exc:
        logger.warning("Domain auto-selection failed: %s — using defaults", exc)
    return _CANDIDATE_DOMAINS[:n]


# ---------------------------------------------------------------------------
# Topic ID + callback type
# ---------------------------------------------------------------------------


def _topic_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


ProgressCallback = Callable[[str], Awaitable[None]]
TopicCallback = Callable[["ScoutTopicData"], Awaitable[None]]


# ---------------------------------------------------------------------------
# Format inference from URL/content-type (extended in Part 4)
# ---------------------------------------------------------------------------

_FORMAT_PATTERNS: list[tuple[str, str]] = [
    (".pdf", "pdf"),
    (".docx", "docx"),
    (".xlsx", "xlsx"),
    (".pptx", "pptx"),
    ("/pdf/", "pdf"),
    ("youtube.com/watch", "video"),
    ("youtu.be/", "video"),
    (".mp3", "audio"),
    (".mp4", "video"),
    (".srt", "srt"),
    (".csv", "csv"),
    (".json", "json"),
]


def _infer_format(url: str) -> str:
    url_lower = url.lower()
    for pattern, fmt in _FORMAT_PATTERNS:
        if pattern in url_lower:
            return fmt
    return "html"


# ---------------------------------------------------------------------------
# Per-domain processing (runs in parallel)
# ---------------------------------------------------------------------------


async def _process_domain(
    domain: str,
    progress_callback: Optional[ProgressCallback] = None,
    topic_callback: Optional[TopicCallback] = None,
) -> Optional[ScoutTopicData]:
    """Crawl all sources for one domain, verify URLs, score, and return a topic."""
    async def _log(msg: str) -> None:
        logger.info("[Scout] %s", msg)
        if progress_callback:
            try:
                await progress_callback(msg)
            except Exception:
                pass

    await _log(f"Scanning: {domain[:60]}")

    # Lazy imports avoid circular dependency (layers import ScoutSource from this module)
    from agents.crawlers.layer_a import run_layer_a
    from agents.crawlers.layer_b import run_layer_b
    from agents.crawlers.layer_c import run_layer_c

    gathered = await asyncio.gather(
        run_layer_a(domain),           # Layer A: 14 science/research crawlers
        run_layer_b(domain),           # Layer B: 10 legislation/regulatory crawlers
        run_layer_c(domain),           # Layer C: 10 finance/economic data crawlers
        _fetch_openalex(domain),       # cross-layer aggregator (kept for coverage)
        _fetch_hackernews(domain),     # Layer D social signal (full integration in Step 9)
        _probe_llm_uncertainty(domain),
        return_exceptions=True,
    )

    layer_a_srcs = gathered[0] if not isinstance(gathered[0], Exception) else []
    layer_b_srcs = gathered[1] if not isinstance(gathered[1], Exception) else []
    layer_c_srcs = gathered[2] if not isinstance(gathered[2], Exception) else []
    oa_srcs      = gathered[3] if not isinstance(gathered[3], Exception) else []
    hn_srcs      = gathered[4] if not isinstance(gathered[4], Exception) else []
    uncertainty  = gathered[5] if not isinstance(gathered[5], Exception) else 0.5

    # Merge: Layer A (science) → B (legislation) → C (finance) → cross-layer → social
    all_sources: list[ScoutSource] = [
        *layer_a_srcs, *layer_b_srcs, *layer_c_srcs, *oa_srcs, *hn_srcs,
    ]

    # 5-point Anti-Hallucination & Verification Firewall
    # Lazy import avoids circular dependency with verifier → topic_scout
    from agents.crawlers.verifier import verify_batch
    batch = await verify_batch(all_sources, _HTTP, max_concurrent=8)
    verified = batch.verified
    post_cutoff_models = batch.post_cutoff_models

    # 3-Stage Deduplication: URL hash → SimHash → semantic cosine
    # Eliminates cross-layer near-duplicates (same paper at different URLs/sources)
    if verified:
        from agents.crawlers.dedup import DedupPipeline
        dedup = DedupPipeline(enable_semantic=False)   # semantic dedup in scorer (w3)
        verified = await dedup.filter(verified)

    if not verified:
        await _log(f"  {domain[:40]}: no sources passed verification firewall — skipped")
        return None

    recency      = _compute_recency(verified)
    density      = min(1.0, len(verified) / 10.0)
    social_count = sum(1 for s in verified if s.source_type == "hackernews")
    social       = min(1.0, social_count / 5.0)

    best = max(
        (s for s in verified if s.published_at),
        key=lambda s: s.published_at,
        default=verified[0],
    )

    # Derive metadata fields
    best_tier  = _best_tier([s.source_tier for s in verified])
    fmt_types  = list({_infer_format(s.url) for s in verified})
    langs      = list({s.language for s in verified if s.language})
    est_tokens = len(verified) * 800

    # 8-component KNOWLEDGE_GAP_SCORE (lazy import avoids circular dependency)
    from agents.crawlers.scorer import ScorerContext, score_topic
    ctx = ScorerContext(
        domain=domain,
        sources=verified,
        cutoff_model_targets=post_cutoff_models,
        languages=langs,
        format_types=fmt_types,
        recency_score=recency,
        base_uncertainty=float(uncertainty),
        http_client=_HTTP,
    )
    scoring = await score_topic(ctx)

    # Legacy 4-component score kept for backwards compat (used in sort + old API consumers)
    score = round(
        0.40 * recency
        + 0.30 * float(uncertainty)
        + 0.20 * density
        + 0.10 * social,
        3,
    )

    await _log(
        f"  {domain[:40]}: gap={scoring.knowledge_gap_score:.3f} score={score:.2f} "
        f"tier={best_tier} sources={len(verified)} "
        f"uncertainty={scoring.llm_uncertainty:.2f} cutoff_models={len(post_cutoff_models)}"
    )

    topic = ScoutTopicData(
        topic_id=_topic_id(domain),
        title=domain,
        summary=best.title[:250],
        score=score,
        recency_score=recency,
        llm_uncertainty=scoring.llm_uncertainty,
        source_count=len(verified),
        social_signal=round(social, 3),
        sources=verified[:10],
        domains=[domain],
        discovered_at=datetime.now(tz=timezone.utc).isoformat(),
        # new fields
        knowledge_gap_score=scoring.knowledge_gap_score,
        cutoff_model_targets=post_cutoff_models,
        format_types=fmt_types,
        languages=langs,
        citation_velocity=scoring.citation_velocity,
        source_tier=best_tier,
        estimated_tokens=est_tokens,
        ingest_ready=True,
    )

    if topic_callback:
        try:
            await topic_callback(topic)
        except Exception:
            pass

    return topic


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


async def run_scout(
    domains: Optional[list[str]] = None,
    max_topics: int = 50,
    progress_callback: Optional[ProgressCallback] = None,
    topic_callback: Optional[TopicCallback] = None,
) -> list[ScoutTopicData]:
    """
    Discover knowledge-gap topics using all available methods.
    All domains are processed in parallel for minimum wall-clock time.
    Returns topics sorted by score (highest first), only with verified sources.
    """
    async def _log(msg: str) -> None:
        logger.info("[Scout] %s", msg)
        if progress_callback:
            try:
                await progress_callback(msg)
            except Exception:
                pass

    await _log("Initialising Gap Scout...")

    if domains is None:
        await _log("AI selecting optimal scan domains...")
        domains = await _select_domains(n=6)
        await _log(f"Domains selected: {', '.join(d.split()[0] for d in domains[:4])}...")

    # Process all domains concurrently (each internally already fans out crawlers in parallel)
    domain_results = await asyncio.gather(
        *[_process_domain(domain, progress_callback, topic_callback) for domain in domains],
        return_exceptions=True,
    )

    results = [r for r in domain_results if isinstance(r, ScoutTopicData)]
    results.sort(key=lambda t: t.score, reverse=True)
    await _log(f"Scout complete — {len(results)} topics discovered.")
    return results[:max_topics]
