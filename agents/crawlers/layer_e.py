"""
agents/crawlers/layer_e.py — Layer E: Multimedia & Archive crawlers (6 sources)

Sources covered:
  1.  YouTubeCrawler         — YouTube Data API v3 (needs YOUTUBE_API_KEY)
  2.  PodcastIndexCrawler    — Podcast Index API (needs PODCAST_INDEX_API_KEY)
  3.  TEDCrawler             — TED talks RSS + JSON search (no key)
  4.  InternetArchiveCrawler — Internet Archive advancedsearch JSON (no key)
  5.  JSTORCrawler           — JSTOR Text Analyzer API + OpenAlex fallback
  6.  EuropeanaCrawler       — Europeana REST API v2 (needs EUROPEANA_API_KEY)

All classes inherit CrawlerBase for circuit breaker, backoff, ETag caching.

Public API:
    sources = await run_layer_e("CSRD sustainability conference talks")
    sources = await run_layer_e("AI governance", enabled=["youtube", "ted", "archive"])
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus

from agents.crawlers.base import CrawlerBase
from agents.topic_scout import ScoutSource, _get_source_tier

logger = logging.getLogger(__name__)

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
    "%B %d, %Y",
    "%Y/%m/%d",
    "%Y",
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


def _rss_text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None else ""


_ATOM_NS = "http://www.w3.org/2005/Atom"
_AT = f"{{{_ATOM_NS}}}"


def _atom_text(el: Optional[ET.Element], tag: str) -> str:
    child = el.find(f"{_AT}{tag}") if el is not None else None
    return (child.text or "").strip() if child is not None else ""


def _atom_link(el: ET.Element) -> str:
    for link in el.findall(f"{_AT}link"):
        if link.get("rel", "alternate") == "alternate":
            return link.get("href", "")
    lnk = el.find(f"{_AT}link")
    return lnk.get("href", "") if lnk is not None else ""


# ===========================================================================
# 1. YouTubeCrawler
# ===========================================================================


class YouTubeCrawler(CrawlerBase):
    """
    YouTube Data API v3 — conference talks, lectures, explainers.
    Requires YOUTUBE_API_KEY (free tier: 10,000 units/day).
    Gracefully skips when key is absent.
    """

    source_id = "youtube"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        from config.settings import settings
        api_key = getattr(settings, "youtube_api_key", "")
        if not api_key:
            logger.debug("[youtube] YOUTUBE_API_KEY not set — skipping")
            return []

        published_after = (
            datetime.now(timezone.utc) - timedelta(days=365)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        url = (
            "https://www.googleapis.com/youtube/v3/search"
            f"?part=snippet&q={quote_plus(query)}"
            f"&type=video&order=relevance&maxResults=8"
            f"&publishedAfter={published_after}"
            f"&key={api_key}"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
        sources: list[ScoutSource] = []
        for item in resp.json().get("items", []):
            vid_id = item.get("id", {}).get("videoId", "")
            if not vid_id:
                continue
            link = f"https://www.youtube.com/watch?v={vid_id}"
            snippet = item.get("snippet", {})
            sources.append(ScoutSource(
                url=link,
                title=(snippet.get("title") or "").strip(),
                published_at=_to_iso(snippet.get("publishedAt", "")),
                source_type="youtube",
                verified=True,
                source_tier="B",
                snippet=(snippet.get("description") or "")[:500],
            ))
        return sources


# ===========================================================================
# 2. PodcastIndexCrawler
# ===========================================================================

def _podcast_index_auth_headers(api_key: str, api_secret: str) -> dict:
    """Generate HMAC-SHA256 auth headers for Podcast Index API."""
    epoch = int(time.time())
    hash_input = f"{api_key}{api_secret}{epoch}".encode("utf-8")
    auth_hash = hashlib.sha1(hash_input).hexdigest()  # PI uses SHA-1
    return {
        "X-Auth-Key": api_key,
        "X-Auth-Date": str(epoch),
        "Authorization": auth_hash,
        "User-Agent": "FoundryScout/2.0",
    }


class PodcastIndexCrawler(CrawlerBase):
    """
    Podcast Index API — episode search by topic keyword.
    Free with registration at podcastindex.org.
    Requires PODCAST_INDEX_API_KEY + PODCAST_INDEX_API_SECRET.
    """

    source_id = "podcastindex"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        from config.settings import settings
        api_key    = getattr(settings, "podcast_index_api_key", "")
        api_secret = getattr(settings, "podcast_index_api_secret", "")
        if not api_key or not api_secret:
            logger.debug("[podcastindex] PODCAST_INDEX_API_KEY/SECRET not set — skipping")
            return []

        url = (
            f"https://api.podcastindex.org/api/1.0/search/byterm"
            f"?q={quote_plus(query)}&clean&max=8&pretty"
        )
        resp = await self._fetch(
            url,
            headers=_podcast_index_auth_headers(api_key, api_secret),
            use_cache_headers=False,
        )
        if resp.status_code != 200:
            return []
        sources: list[ScoutSource] = []
        for feed in resp.json().get("feeds", []):
            link = feed.get("url") or feed.get("link", "")
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(feed.get("title") or "").strip(),
                published_at=_to_iso(
                    str(feed.get("newestItemPublishTime", ""))
                    if feed.get("newestItemPublishTime") else ""
                ),
                source_type="podcastindex",
                verified=True,
                source_tier="C",
                snippet=(feed.get("description") or "")[:500],
            ))
        return sources


# ===========================================================================
# 3. TEDCrawler
# ===========================================================================


class TEDCrawler(CrawlerBase):
    """TED talks — RSS feeds for Science, Technology, Business, Global issues."""

    source_id = "ted"
    default_poll_interval = 3600

    _FEEDS = [
        "https://feeds.feedburner.com/TEDTalks_video",
        "https://www.ted.com/talks/rss",
    ]

    async def crawl(self, query: str) -> list[ScoutSource]:
        ql = query.lower().split()
        sources: list[ScoutSource] = []
        seen: set[str] = set()

        for feed_url in self._FEEDS:
            try:
                resp = await self._fetch(feed_url, use_cache_headers=True)
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.text)

                # Try Atom entries first
                entries = root.findall(f"{_AT}entry")
                if entries:
                    for entry in entries:
                        title = _atom_text(entry, "title")
                        link  = _atom_link(entry)
                        pub   = _atom_text(entry, "updated") or _atom_text(entry, "published")
                        summ  = _atom_text(entry, "summary")
                        if not link or link in seen:
                            continue
                        combined = (title + " " + summ).lower()
                        if ql and not any(kw in combined for kw in ql):
                            continue
                        seen.add(link)
                        sources.append(ScoutSource(
                            url=link,
                            title=title.strip(),
                            published_at=_to_iso(pub),
                            source_type="ted",
                            verified="ted.com" in link,
                            source_tier="B",
                            snippet=summ[:500],
                        ))
                else:
                    for item in root.findall(".//item"):
                        title = _rss_text(item, "title")
                        link  = _rss_text(item, "link") or _rss_text(item, "guid")
                        pub   = _rss_text(item, "pubDate")
                        desc  = _rss_text(item, "description")
                        if not link or link in seen:
                            continue
                        combined = (title + " " + desc).lower()
                        if ql and not any(kw in combined for kw in ql):
                            continue
                        seen.add(link)
                        sources.append(ScoutSource(
                            url=link,
                            title=title.strip(),
                            published_at=_to_iso(pub),
                            source_type="ted",
                            verified="ted.com" in link,
                            source_tier="B",
                            snippet=desc[:500],
                        ))
                if sources:
                    break
            except ET.ParseError as exc:
                logger.debug("[ted] XML parse error feed=%s: %s", feed_url, exc)

        return sources[:8]


# ===========================================================================
# 4. InternetArchiveCrawler
# ===========================================================================


class InternetArchiveCrawler(CrawlerBase):
    """
    Internet Archive full-text search — texts, audio, video, software.
    Public API, no key required. Rate limit: ~1 req/s.
    """

    source_id = "archive"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            "https://archive.org/advancedsearch.php"
            f"?q={quote_plus(query)}"
            f"+date:[{_days_ago(730)}+TO+9999-12-31]"
            "&fl[]=identifier,title,date,subject,mediatype,description"
            "&sort[]=date+desc&rows=8&page=1&output=json"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
        sources: list[ScoutSource] = []
        try:
            docs = resp.json().get("response", {}).get("docs", [])
            for doc in docs:
                ident = doc.get("identifier", "")
                if not ident:
                    continue
                link = f"https://archive.org/details/{ident}"
                media = doc.get("mediatype", "texts")
                # Map mediatype to our format taxonomy
                fmt = {
                    "audio": "audio",
                    "movies": "video",
                    "texts": "pdf",
                    "software": "html",
                    "image": "html",
                }.get(media, "html")
                title_raw = doc.get("title", ident)
                title = (title_raw[0] if isinstance(title_raw, list) else title_raw).strip()
                date_raw = doc.get("date", "")
                desc_raw = doc.get("description", "")
                desc = (desc_raw[0] if isinstance(desc_raw, list) else desc_raw)
                sources.append(ScoutSource(
                    url=link,
                    title=title,
                    published_at=_to_iso(date_raw[:10] if date_raw else ""),
                    source_type="archive",
                    verified=True,
                    source_tier="B",
                    snippet=str(desc)[:500],
                ))
        except Exception as exc:
            logger.debug("[archive] parse error: %s", exc)
        return sources


# ===========================================================================
# 5. JSTORCrawler
# ===========================================================================


class JSTORCrawler(CrawlerBase):
    """
    JSTOR content search — academic journals, books, primary sources.
    Uses JSTOR Text Analyzer public API; falls back to OpenAlex venue filter.
    """

    source_id = "jstor"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        sources: list[ScoutSource] = []

        # JSTOR public search (no key, limited results)
        url = (
            f"https://www.jstor.org/open/search?query={quote_plus(query)}"
            f"&sd=2022&ed=&la=en&rows=8&pageIndex=0&so=rel&searchType=faceted"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code == 200:
            try:
                data = resp.json()
                results = data.get("results", {}).get("items", [])
                for item in results[:8]:
                    stable_url = item.get("stableUrl") or item.get("doi", "")
                    if stable_url and not stable_url.startswith("http"):
                        stable_url = "https://www.jstor.org" + stable_url
                    if not stable_url:
                        continue
                    sources.append(ScoutSource(
                        url=stable_url,
                        title=(item.get("title") or "").strip(),
                        published_at=_to_iso(str(item.get("pubYear", ""))),
                        source_type="jstor",
                        verified="jstor.org" in stable_url,
                        source_tier="A",
                        snippet=(item.get("snippet") or "")[:500],
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: OpenAlex filtering by JSTOR publisher
        url2 = (
            f"https://api.openalex.org/works"
            f"?search={quote_plus(query)}"
            f"&filter=host_venue.publisher:JSTOR,publication_year:>2021"
            f"&sort=publication_date:desc&per-page=8"
            f"&select=id,title,doi,publication_date,open_access"
        )
        resp2 = await self._fetch(url2, use_cache_headers=False)
        if resp2.status_code != 200:
            return []
        for work in resp2.json().get("results", []):
            oa = work.get("open_access") or {}
            link = oa.get("oa_url") or ""
            if not link:
                doi = work.get("doi", "")
                link = f"https://doi.org/{doi}" if doi else ""
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(work.get("title") or "").strip(),
                published_at=_to_iso(work.get("publication_date", "")),
                source_type="jstor",
                verified=False,
                source_tier="A",
            ))
        return sources[:8]


# ===========================================================================
# 6. EuropeanaCrawler
# ===========================================================================


class EuropeanaCrawler(CrawlerBase):
    """
    Europeana REST API v2 — European cultural heritage (texts, audio, video).
    Free API key from pro.europeana.eu. Gracefully skips without key.
    """

    source_id = "europeana"
    default_poll_interval = 7200

    async def crawl(self, query: str) -> list[ScoutSource]:
        from config.settings import settings
        api_key = getattr(settings, "europeana_api_key", "")

        # Use demo key if none configured (heavily rate-limited but functional)
        key = api_key or "api2demo"

        url = (
            f"https://api.europeana.eu/record/v2/search.json"
            f"?wskey={key}&query={quote_plus(query)}"
            f"&rows=8&sort=score+desc&profile=standard"
            f"&reusability=open&qf=TYPE:TEXT"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
        sources: list[ScoutSource] = []
        try:
            for item in resp.json().get("items", []):
                guid = item.get("guid") or ""
                link = (
                    item.get("edmIsShownAt", [None])[0]
                    or item.get("edmIsShownBy", [None])[0]
                    or guid
                    or ""
                )
                if isinstance(link, list):
                    link = link[0] if link else ""
                if not link:
                    continue
                title_raw = item.get("title") or item.get("dcTitle") or []
                title = (title_raw[0] if isinstance(title_raw, list) else title_raw).strip()
                date_raw = item.get("year") or item.get("dcDate") or ""
                date_s = date_raw[0] if isinstance(date_raw, list) else str(date_raw)
                desc_raw = item.get("dcDescription") or []
                desc = (desc_raw[0] if isinstance(desc_raw, list) else str(desc_raw))
                sources.append(ScoutSource(
                    url=link,
                    title=title,
                    published_at=_to_iso(date_s),
                    source_type="europeana",
                    verified="europeana.eu" in link,
                    source_tier="B",
                    snippet=str(desc)[:500],
                ))
        except Exception as exc:
            logger.debug("[europeana] parse error: %s", exc)
        return sources


# ===========================================================================
# Registry + public entry-point
# ===========================================================================

_CRAWLERS: dict[str, CrawlerBase] = {
    c.source_id: c
    for c in [
        YouTubeCrawler(),
        PodcastIndexCrawler(),
        TEDCrawler(),
        InternetArchiveCrawler(),
        JSTORCrawler(),
        EuropeanaCrawler(),
    ]
}


async def run_layer_e(
    query: str,
    enabled: Optional[list[str]] = None,
) -> list[ScoutSource]:
    """
    Run all (or a subset of) Layer E crawlers in parallel.
    Returns deduplicated ScoutSource list.
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

    results = await asyncio.gather(
        *[c.safe_crawl(query) for c in crawlers],
        return_exceptions=True,
    )

    seen_urls: set[str] = set()
    sources: list[ScoutSource] = []
    for batch in results:
        if not isinstance(batch, list):
            continue
        for src in batch:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                sources.append(src)
    return sources


def get_crawler_status_e() -> list[dict]:
    """Return status dict for each Layer E crawler."""
    return [
        {
            "source_id": c.source_id,
            "layer": "E",
            "poll_interval": c.poll_interval,
            "is_paused": c.is_paused,
            "last_seen_id": c.last_seen_id,
            "consecutive_errors": c._consecutive_errors,
        }
        for c in _CRAWLERS.values()
    ]
