"""
agents/crawlers/layer_d.py — Layer D: Tech & Social Signal crawlers (6 sources) - ENTERPRISE EDITION

Sources covered:
  1.  GitHubSearchCrawler   — GitHub Search API (repos, releases; 60 req/h no key)
  2.  RedditCrawler         — Reddit JSON API (r/MachineLearning, r/ESGInvesting…)
  3.  StackExchangeCrawler  — Stack Exchange API v2.3 (SO, AI, DataScience, Law)
  4.  ProductHuntCrawler    — Product Hunt GraphQL API + RSS fallback
  5.  PapersWithCodeCrawler — Papers With Code REST API (ML + code, no key)
  6.  MastodonCrawler       — Mastodon v2 search API (mastodon.social + sigmoid.social)

Ulepszenia PRO:
  - Bounded Concurrency (Semaphore): Limitowanie równoległych żądań do delikatnych i ściśle limitowanych API społecznościowych.
  - Global Layer Timeout: Ochrona przed tzw. "zombie connections" od sfederowanych instancji Mastodona.
  - Exception Unpacking: Jawne logowanie wyjątków w operacjach wsadowych (asyncio.gather).
  - Safe Payload Parsing: Ochrona przed nieprawidłowymi zwrotami JSON przy obciążeniu API (np. Reddit Overload).
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Set, Any
from urllib.parse import quote_plus

from agents.crawlers.base import CrawlerBase
from agents.topic_scout import ScoutSource, _get_source_tier

logger = logging.getLogger("foundry.agents.crawlers.layer_d")

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_FMTS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
)

def _to_iso(date_str: str) -> str:
    if not date_str:
        return ""
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(date_str[:32].strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return date_str

def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")

def _unix_to_iso(ts: int | float | str) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return ""

def _rss_text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None else ""


# ===========================================================================
# 1. GitHubSearchCrawler
# ===========================================================================

class GitHubSearchCrawler(CrawlerBase):
    """
    GitHub Search API — trending repositories and recently-updated repos.
    60 req/h unauthenticated. Uses Accept: application/vnd.github.v3+json.
    """

    source_id = "github"
    default_poll_interval = 900

    async def crawl(self, query: str) -> list[ScoutSource]:
        since = _days_ago(90)
        url = (
            f"https://api.github.com/search/repositories"
            f"?q={quote_plus(query)}+pushed:>{since}"
            f"&sort=updated&order=desc&per_page=8"
        )
        resp = await self._fetch(
            url,
            headers={"Accept": "application/vnd.github.v3+json"},
            use_cache_headers=False,
        )
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
        for repo in data.get("items", []):
            link = repo.get("html_url", "")
            if not link:
                continue
            pushed = repo.get("pushed_at", "")
            desc = (repo.get("description") or "")[:500]
            sources.append(ScoutSource(
                url=link,
                title=repo.get("full_name", "").strip(),
                published_at=_to_iso(pushed),
                source_type="github",
                verified=True,
                source_tier="B",
                snippet=desc,
            ))
        return sources


# ===========================================================================
# 2. RedditCrawler
# ===========================================================================

_REDDIT_SUBS = [
    "MachineLearning", "artificial", "AIEthics",
    "sustainability", "ESGInvesting", "ClimateOffensive",
    "govtech", "legaltech", "fintech", "datasciencenews",
]

class RedditCrawler(CrawlerBase):
    """Reddit JSON search API — no auth required for read-only queries."""

    source_id = "reddit"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        sub_str = "+".join(_REDDIT_SUBS)
        url = (
            f"https://www.reddit.com/r/{sub_str}/search.json"
            f"?q={quote_plus(query)}&sort=new&t=month&limit=8&restrict_sr=1"
        )
        resp = await self._fetch(
            url,
            headers={"User-Agent": "FoundryScout/2.0 (research-tool; non-commercial)"},
            use_cache_headers=False,
        )
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
        children = data.get("data", {}).get("children", [])
        for child in children:
            post = child.get("data", {})
            url_post = post.get("url") or post.get("permalink", "")
            if url_post and url_post.startswith("/"):
                url_post = "https://www.reddit.com" + url_post
            if not url_post:
                continue
            sources.append(ScoutSource(
                url=url_post,
                title=(post.get("title") or "").strip(),
                published_at=_unix_to_iso(post.get("created_utc", 0)),
                source_type="reddit",
                verified=True,
                source_tier="C",
                snippet=(post.get("selftext") or post.get("link_flair_text", ""))[:500],
            ))
        return sources


# ===========================================================================
# 3. StackExchangeCrawler
# ===========================================================================

_SE_SITES = ["stackoverflow", "datascience", "ai", "law", "economics"]

class StackExchangeCrawler(CrawlerBase):
    """Stack Exchange API v2.3 — high-voted questions across relevant sites."""

    source_id = "stackexchange"
    default_poll_interval = 1800

    async def crawl(self, query: str) -> list[ScoutSource]:
        since_ts = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp())
        sources: list[ScoutSource] = []
        seen: set[str] = set()

        for site in _SE_SITES[:3]:   # limit to 3 sites per tick to respect rate limits
            url = (
                f"https://api.stackexchange.com/2.3/search/advanced"
                f"?q={quote_plus(query)}&site={site}"
                f"&sort=votes&order=desc&fromdate={since_ts}"
                f"&pagesize=4&filter=default"
            )
            resp = await self._fetch(url, use_cache_headers=False)
            if resp.status_code != 200:
                continue
                
            try:
                data = resp.json()
                items = data.get("items", [])
                for item in items:
                    link = item.get("link", "")
                    if not link or link in seen:
                        continue
                    seen.add(link)
                    sources.append(ScoutSource(
                        url=link,
                        title=(item.get("title") or "").strip(),
                        published_at=_unix_to_iso(item.get("creation_date", 0)),
                        source_type="stackexchange",
                        verified=True,
                        source_tier="B",
                        snippet=f"Score: {item.get('score', 0)}, Answers: {item.get('answer_count', 0)}",
                    ))
            except Exception as exc:
                logger.debug("[stackexchange] parse error site=%s: %s", site, exc)

        return sources[:8]


# ===========================================================================
# 4. ProductHuntCrawler
# ===========================================================================

class ProductHuntCrawler(CrawlerBase):
    """Product Hunt — new products/tools via public RSS and GraphQL API."""

    source_id = "producthunt"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        from config.settings import settings
        sources: list[ScoutSource] = []

        # Primary: GraphQL (needs developer token)
        ph_token = getattr(settings, "producthunt_api_key", "")
        if ph_token:
            gql = """
            query($q: String!) {
              posts(search: {query: $q}, order: NEWEST, first: 8) {
                edges { node {
                  name tagline url createdAt
                  topics { edges { node { name } } }
                }}
              }
            }"""
            try:
                resp = await self._fetch(
                    "https://api.producthunt.com/v2/api/graphql",
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {ph_token}",
                        "Content-Type": "application/json",
                    },
                    json={"query": gql, "variables": {"q": query}},
                    use_cache_headers=False,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    edges = (
                        data.get("data", {})
                        .get("posts", {})
                        .get("edges", [])
                    )
                    for edge in edges:
                        node = edge.get("node", {})
                        link = node.get("url", "")
                        if not link:
                            continue
                        sources.append(ScoutSource(
                            url=link,
                            title=node.get("name", "").strip(),
                            published_at=_to_iso(node.get("createdAt", "")),
                            source_type="producthunt",
                            verified=True,
                            source_tier="C",
                            snippet=node.get("tagline", "")[:500],
                        ))
                    if sources:
                        return sources
            except Exception as exc:
                logger.debug("[producthunt] GraphQL error: %s", exc)

        # Fallback: RSS feed + title keyword filter
        rss_url = "https://www.producthunt.com/feed"
        resp2 = await self._fetch(rss_url, use_cache_headers=True)
        if resp2.status_code != 200:
            return []
        ql = query.lower().split()
        try:
            root = ET.fromstring(resp2.text)
            for item in root.findall(".//item"):
                title = _rss_text(item, "title")
                link  = _rss_text(item, "link") or _rss_text(item, "guid")
                pub   = _rss_text(item, "pubDate")
                desc  = _rss_text(item, "description")
                if not link:
                    continue
                combined = (title + " " + desc).lower()
                if ql and not any(kw in combined for kw in ql):
                    continue
                sources.append(ScoutSource(
                    url=link,
                    title=title.strip(),
                    published_at=_to_iso(pub),
                    source_type="producthunt",
                    verified=True,
                    source_tier="C",
                    snippet=desc[:500],
                ))
        except ET.ParseError as exc:
            logger.debug("[producthunt] RSS parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 5. PapersWithCodeCrawler
# ===========================================================================

class PapersWithCodeCrawler(CrawlerBase):
    """Papers With Code REST API — ML papers + GitHub repos. No key needed."""

    source_id = "paperswithcode"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            f"https://paperswithcode.com/api/v1/papers/"
            f"?q={quote_plus(query)}&page=1&items_per_page=8&ordering=-published"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
        for paper in data.get("results", []):
            # Prefer arxiv URL, then paper URL
            link = paper.get("arxiv_id", "")
            if link:
                link = f"https://arxiv.org/abs/{link}"
            else:
                link = paper.get("url_pdf") or paper.get("url_abs", "")
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(paper.get("title") or "").strip(),
                published_at=_to_iso(paper.get("published", "")),
                source_type="paperswithcode",
                verified="arxiv.org" in link,
                source_tier="B",
                snippet=(paper.get("abstract") or "")[:500],
            ))
        return sources


# ===========================================================================
# 6. MastodonCrawler
# ===========================================================================

_MASTODON_INSTANCES = [
    "mastodon.social",      # general (largest)
    "sigmoid.social",       # ML / AI research community
    "scholar.social",       # academics & researchers
]

class MastodonCrawler(CrawlerBase):
    """Mastodon API v2 search — academic and tech federated social signals."""

    source_id = "mastodon"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        sources: list[ScoutSource] = []
        seen: set[str] = set()

        for instance in _MASTODON_INSTANCES:
            url = (
                f"https://{instance}/api/v2/search"
                f"?q={quote_plus(query)}&type=statuses&limit=5&resolve=false"
            )
            try:
                resp = await self._fetch(url, use_cache_headers=False)
                if resp.status_code != 200:
                    continue
                    
                data = resp.json()
                statuses = data.get("statuses", [])
                
                for st in statuses:
                    link = st.get("url", "")
                    if not link or link in seen:
                        continue
                    # Skip boosts (reblogs)
                    if st.get("reblog"):
                        continue
                    seen.add(link)
                    account = st.get("account", {})
                    display = account.get("display_name") or account.get("username", "")
                    # Strip HTML tags from content
                    import re
                    content = re.sub(r"<[^>]+>", " ", st.get("content", "")).strip()
                    sources.append(ScoutSource(
                        url=link,
                        title=f"[{display}] {content[:120]}",
                        published_at=_to_iso(st.get("created_at", "")),
                        source_type="mastodon",
                        verified=True,
                        source_tier="C",
                        snippet=content[:500],
                    ))
            except Exception as exc:
                logger.debug("[mastodon] %s error: %s", instance, exc)

        return sources[:8]


# ===========================================================================
# Registry + Public Entry-Point (Enterprise Edition)
# ===========================================================================

_CRAWLERS: dict[str, CrawlerBase] = {
    c.source_id: c
    for c in [
        GitHubSearchCrawler(),
        RedditCrawler(),
        StackExchangeCrawler(),
        ProductHuntCrawler(),
        PapersWithCodeCrawler(),
        MastodonCrawler(),
    ]
}


async def _run_single_crawler_safe(crawler: CrawlerBase, query: str, semaphore: asyncio.Semaphore) -> List[ScoutSource]:
    """Wraper wykonujący zapytanie chronione Semaforem z bezbłędnym przechwyceniem awarii."""
    async with semaphore:
        try:
            return await crawler.safe_crawl(query)
        except Exception as exc:
            logger.error(f"[Layer D] Fatal exception in crawler {crawler.source_id}: {exc}", exc_info=True)
            return []


async def run_layer_d(
    query: str,
    enabled: Optional[list[str]] = None,
) -> list[ScoutSource]:
    """
    Run all (or a subset of) Layer D crawlers in parallel.
    Returns deduplicated ScoutSource list.
    Zabezpieczone przez Bounded Concurrency (Semaphore) i Global Timeout.
    """
    from config.settings import settings
    active = enabled
    if active is None:
        cfg_enabled = getattr(settings, "scout_sources_enabled", [])
        if cfg_enabled:
            active = [sid for sid in cfg_enabled if sid in _CRAWLERS]
        else:
            active = list(_CRAWLERS.keys())

    crawlers = [_CRAWLERS[sid] for sid in active if sid in _CRAWLERS]
    if not crawlers:
        return []

    # Limitujemy współbieżność (kluczowe przy API społecznościowych)
    concurrency_limit = getattr(settings, "crawler_concurrency_limit", 10)
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [_run_single_crawler_safe(c, query, semaphore) for c in crawlers]

    # Globalny limit czasu dla całej warstwy
    layer_timeout = getattr(settings, "layer_d_timeout_seconds", 60.0)

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=layer_timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"[Layer D] Przekroczono globalny limit czasu ({layer_timeout}s). Niektóre wyniki mogły zostać utracone.")
        return []

    seen_urls: set[str] = set()
    sources: list[ScoutSource] = []
    
    # Dekompozycja wyników
    for idx, batch in enumerate(results):
        if isinstance(batch, Exception):
            logger.error(f"[Layer D] Zgromadzono wyjątek z crawlera {crawlers[idx].source_id}: {batch}")
            continue
        if not isinstance(batch, list):
            continue
            
        for src in batch:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                sources.append(src)
                
    return sources


def get_crawler_status_d() -> list[dict]:
    """Return status dict for each Layer D crawler."""
    return [
        {
            "source_id": c.source_id,
            "layer": "D",
            "poll_interval": c.poll_interval,
            "is_paused": c.is_paused,
            "last_seen_id": c.last_seen_id,
            "consecutive_errors": c._consecutive_errors,
        }
        for c in _CRAWLERS.values()
    ]
