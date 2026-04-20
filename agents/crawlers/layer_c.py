"""
agents/crawlers/layer_c.py — Layer C: Finance & Economic Data crawlers (10 sources) - ENTERPRISE EDITION

Sources covered:
  1.  IMFCrawler            — IMF publications & working papers JSON
  2.  WorldBankCrawler      — World Bank Open Data search JSON
  3.  FREDCrawler           — Federal Reserve FRED economic data series
  4.  ECBCrawler            — European Central Bank publications Atom
  5.  BISCrawler            — Bank for International Settlements papers RSS
  6.  EurostatCrawler       — Eurostat news & data releases RSS
  7.  UNCrawler             — UN Digital Library REST search
  8.  IRENACrawler          — IRENA publications JSON
  9.  OurWorldInDataCrawler — OWID data catalog GitHub API
  10. HDXCrawler            — CKAN Humanitarian Data Exchange API

Ulepszenia PRO:
  - Bounded Concurrency (Semaphore): Limitowanie równoległych żądań HTTP do API finansowych (anti-DDoS).
  - Global Layer Timeout: Bezpieczny limit czasu na zebranie wszystkich informacji (zabezpieczenie przed zawieszeniem).
  - Exception Unpacking: Jawne logowanie wyjątków w operacjach wsadowych (asyncio.gather).
  - Safe Payload Parsing: Ochrona przed niestandardowymi odpowiedziami w API mimo kodu 200 OK.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from urllib.parse import quote_plus

from agents.crawlers.base import CrawlerBase
from agents.topic_scout import ScoutSource

logger = logging.getLogger("foundry.agents.crawlers.layer_c")

# ---------------------------------------------------------------------------
# Date helpers (mirror layer_a / layer_b)
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

def _rss_text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None else ""


# ===========================================================================
# 1. IMFCrawler
# ===========================================================================

class IMFCrawler(CrawlerBase):
    """IMF publications search — working papers, country reports, policy papers."""

    source_id = "imf"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        # Use the IMF API endpoint
        api_url = (
            f"https://www.imf.org/elibrary-angular-api/search"
            f"?q={quote_plus(query)}&lang=en&sort=date&n=8"
        )
        resp = await self._fetch(api_url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        if resp.status_code == 200:
            try:
                data = resp.json()
                items = data.get("docs", data.get("results", data.get("items", [])))
                for item in items[:8]:
                    link = item.get("url") or item.get("link", "")
                    if link and not link.startswith("http"):
                        link = "https://www.imf.org" + link
                    if not link:
                        continue
                    sources.append(ScoutSource(
                        url=link,
                        title=(item.get("title") or "").strip(),
                        published_at=_to_iso(item.get("date") or item.get("pubDate", "")),
                        source_type="imf",
                        verified="imf.org" in link,
                        source_tier="S",
                        snippet=(item.get("description") or item.get("abstract", ""))[:500],
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: IMF RSS feed
        rss_url = "https://www.imf.org/en/Publications/RSS"
        resp2 = await self._fetch(rss_url, use_cache_headers=True)
        if resp2.status_code == 200:
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
                        source_type="imf",
                        verified="imf.org" in link,
                        source_tier="S",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[imf] RSS parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 2. WorldBankCrawler
# ===========================================================================

class WorldBankCrawler(CrawlerBase):
    """World Bank Open Data — development publications, reports, datasets."""

    source_id = "worldbank"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        # World Bank documents search API
        url = (
            "https://search.worldbank.org/api/v2/wds"
            f"?qterm={quote_plus(query)}"
            f"&strdate={_days_ago(730)}&lang_exact=English"
            "&rows=8&os=0&format=json&srt=score&order=desc"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
        sources: list[ScoutSource] = []
        try:
            data = resp.json()
        except ValueError:
            return []
            
        try:
            docs = data.get("documents", {})
            for doc_id, doc in list(docs.items())[:8]:
                if not isinstance(doc, dict):
                    continue
                link = doc.get("url") or doc.get("pdfurl", "")
                if not link:
                    continue
                sources.append(ScoutSource(
                    url=link,
                    title=(doc.get("display_title") or doc.get("title") or "").strip(),
                    published_at=_to_iso(doc.get("docdt") or doc.get("repnb", "")),
                    source_type="worldbank",
                    verified="worldbank.org" in link or "openknowledge.worldbank.org" in link,
                    source_tier="A",
                    snippet=(doc.get("abstracts") or doc.get("abstract", ""))[:500],
                ))
        except Exception as exc:
            logger.debug("[worldbank] parse error: %s", exc)
        return sources


# ===========================================================================
# 3. FREDCrawler
# ===========================================================================

class FREDCrawler(CrawlerBase):
    """Federal Reserve FRED — economic data series and releases."""

    source_id = "fred"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        from config.settings import settings
        # FRED API is publicly accessible without a key for basic searches
        # Full data requires api_key; we search series metadata (public)
        api_key = getattr(settings, "fred_api_key", "")
        key_param = f"&api_key={api_key}" if api_key else ""

        url = (
            f"https://api.stlouisfed.org/fred/series/search"
            f"?search_text={quote_plus(query)}&limit=8&order_by=popularity&sort_order=desc"
            f"&file_type=json{key_param}"
        )
        search_resp = await self._fetch(url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        if search_resp.status_code != 200:
            logger.debug("[fred] primary search status: %s", search_resp.status_code)
        if search_resp.status_code == 200:
            try:
                data = search_resp.json()
            except ValueError:
                return []
                
            try:
                series_list = data.get("seriess", [])
                for s in series_list[:8]:
                    sid = s.get("id", "")
                    link = f"https://fred.stlouisfed.org/series/{sid}" if sid else ""
                    if not link:
                        continue
                    sources.append(ScoutSource(
                        url=link,
                        title=(s.get("title") or "").strip(),
                        published_at=_to_iso(s.get("last_updated", s.get("observation_end", ""))),
                        source_type="fred",
                        verified=True,
                        source_tier="S",
                        snippet=(s.get("notes") or "")[:500],
                    ))
            except Exception as exc:
                logger.debug("[fred] parse error: %s", exc)
        return sources


# ===========================================================================
# 4. ECBCrawler
# ===========================================================================

class ECBCrawler(CrawlerBase):
    """European Central Bank publications — working papers, research bulletins."""

    source_id = "ecb"
    default_poll_interval = 3600

    _FEEDS = [
        "https://www.ecb.europa.eu/rss/press.html",
        "https://www.ecb.europa.eu/rss/wps.html",    # working papers
        "https://www.ecb.europa.eu/rss/research.html",
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
                # ECB feeds may be Atom
                root = ET.fromstring(resp.text)
                # Try Atom first
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
                            source_type="ecb",
                            verified="ecb.europa.eu" in link,
                            source_tier="S",
                            snippet=summ[:500],
                        ))
                else:
                    # RSS fallback
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
                            source_type="ecb",
                            verified="ecb.europa.eu" in link,
                            source_tier="S",
                            snippet=desc[:500],
                        ))
            except ET.ParseError as exc:
                logger.debug("[ecb] XML parse error feed=%s: %s", feed_url, exc)
        return sources[:8]


# ===========================================================================
# 5. BISCrawler
# ===========================================================================

class BISCrawler(CrawlerBase):
    """BIS — Bank for International Settlements working papers and research."""

    source_id = "bis"
    default_poll_interval = 3600

    _FEEDS = [
        "https://www.bis.org/doclist/wp.rss",        # BIS working papers
        "https://www.bis.org/doclist/cgdfs.rss",     # CGFS papers
        "https://www.bis.org/doclist/fsi_insights.rss",
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
                for item in root.findall(".//item"):
                    title = _rss_text(item, "title")
                    link  = _rss_text(item, "link") or _rss_text(item, "guid")
                    pub   = _rss_text(item, "pubDate")
                    desc  = _rss_text(item, "description")
                    if not link or link in seen:
                        continue
                    if not link.startswith("http"):
                        link = "https://www.bis.org" + link
                    combined = (title + " " + desc).lower()
                    if ql and not any(kw in combined for kw in ql):
                        continue
                    seen.add(link)
                    sources.append(ScoutSource(
                        url=link,
                        title=title.strip(),
                        published_at=_to_iso(pub),
                        source_type="bis",
                        verified="bis.org" in link,
                        source_tier="A",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[bis] RSS parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 6. EurostatCrawler
# ===========================================================================

class EurostatCrawler(CrawlerBase):
    """Eurostat — EU statistical data releases and news."""

    source_id = "eurostat"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        ql = query.lower().split()
        sources: list[ScoutSource] = []

        # Eurostat news RSS
        rss_url = "https://ec.europa.eu/eurostat/en/rss/news"
        resp = await self._fetch(rss_url, use_cache_headers=True)
        if resp.status_code == 200:
            try:
                root = ET.fromstring(resp.text)
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
                        source_type="eurostat",
                        verified="europa.eu" in link or "eurostat" in link,
                        source_tier="S",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[eurostat] RSS parse error: %s", exc)

        if sources:
            return sources[:8]

        # Fallback: Eurostat JSON-stat dataset search
        api_url = (
            f"https://ec.europa.eu/eurostat/wdds/rest/data/v2.1/json/en"
            f"/search?query={quote_plus(query)}&hits=8"
        )
        resp2 = await self._fetch(api_url, use_cache_headers=False)
        if resp2.status_code == 200:
            try:
                datasets = resp2.json().get("value", {})
                for ds_id, ds in list(datasets.items())[:8]:
                    link = f"https://ec.europa.eu/eurostat/databrowser/view/{ds_id}/default/table"
                    label = ds.get("label", ds_id) if isinstance(ds, dict) else str(ds)
                    sources.append(ScoutSource(
                        url=link,
                        title=label.strip(),
                        published_at="",
                        source_type="eurostat",
                        verified=True,
                        source_tier="S",
                    ))
            except Exception as exc:
                logger.debug("[eurostat] API parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 7. UNCrawler
# ===========================================================================

class UNCrawler(CrawlerBase):
    """UN Digital Library — UN documents, resolutions, reports."""

    source_id = "undl"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        # UN Digital Library public search API
        url = (
            f"https://digitallibrary.un.org/search?p={quote_plus(query)}"
            f"&of=jm&action=search&rm=&ln=en&rg=8&sc=0&c=Resource+Type%3AResolution"
        )
        search_resp = await self._fetch(url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        if search_resp.status_code != 200:
            logger.debug("[undl] primary search status: %s", search_resp.status_code)

        # Fallback: UN News RSS
        rss_url = "https://news.un.org/en/feed/topic/sustainable-development-goals/feed.rss"
        resp2 = await self._fetch(rss_url, use_cache_headers=True)
        if resp2.status_code == 200:
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
                        source_type="undl",
                        verified="un.org" in link,
                        source_tier="A",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[undl] RSS parse error: %s", exc)

        if sources:
            return sources[:8]

        # UN iLibrary OpenAlex filter
        url3 = (
            f"https://api.openalex.org/works"
            f"?search={quote_plus(query)}"
            f"&filter=institutions.ror:https://ror.org/0021hn262,publication_year:>2022"
            f"&sort=publication_date:desc&per-page=8"
            f"&select=id,title,doi,publication_date,open_access"
        )
        resp3 = await self._fetch(url3, use_cache_headers=False)
        if resp3.status_code == 200:
            try:
                data = resp3.json()
            except ValueError:
                return []
                
            for work in data.get("results", []):
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
                    source_type="undl",
                    verified=False,
                    source_tier="A",
                ))
        return sources[:8]


# ===========================================================================
# 8. IRENACrawler
# ===========================================================================

class IRENACrawler(CrawlerBase):
    """IRENA — International Renewable Energy Agency publications."""

    source_id = "irena"
    default_poll_interval = 7200

    async def crawl(self, query: str) -> list[ScoutSource]:
        # IRENA publications search API
        url = (
            f"https://www.irena.org/api/publications"
            f"?q={quote_plus(query)}&pagesize=8&page=1&sort=date"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        if resp.status_code == 200:
            try:
                data = resp.json()
                items = data.get("publications", data.get("results", data.get("items", [])))
                for item in items[:8]:
                    link = item.get("url") or item.get("pdfUrl", "")
                    if link and not link.startswith("http"):
                        link = "https://www.irena.org" + link
                    if not link:
                        continue
                    sources.append(ScoutSource(
                        url=link,
                        title=(item.get("title") or "").strip(),
                        published_at=_to_iso(item.get("date") or item.get("year", "")),
                        source_type="irena",
                        verified="irena.org" in link,
                        source_tier="A",
                        snippet=(item.get("summary") or item.get("abstract", ""))[:500],
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: IRENA via OpenAlex
        url2 = (
            f"https://api.openalex.org/works"
            f"?search={quote_plus(query)}"
            f"&filter=institutions.ror:https://ror.org/02fsq5v83,publication_year:>2022"
            f"&sort=publication_date:desc&per-page=8"
            f"&select=id,title,doi,publication_date,open_access"
        )
        resp2 = await self._fetch(url2, use_cache_headers=False)
        if resp2.status_code != 200:
            return []
            
        try:
            data = resp2.json()
        except ValueError:
            return []
            
        for work in data.get("results", []):
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
                source_type="irena",
                verified=False,
                source_tier="A",
            ))
        return sources[:8]


# ===========================================================================
# 9. OurWorldInDataCrawler
# ===========================================================================

class OurWorldInDataCrawler(CrawlerBase):
    """Our World in Data — open-access data, charts, and research articles."""

    source_id = "owid"
    default_poll_interval = 7200

    async def crawl(self, query: str) -> list[ScoutSource]:
        sources: list[ScoutSource] = []

        # OWID posts/articles via their public Algolia API
        algolia_url = (
            "https://yh72u0qfh9-dsn.algolia.net/1/indexes/owid_prod"
            f"/query?query={quote_plus(query)}&hitsPerPage=8"
        )
        resp = await self._fetch(algolia_url, use_cache_headers=False)
        if resp.status_code == 200:
            try:
                data = resp.json()
                hits = data.get("hits", [])
                for hit in hits[:8]:
                    slug = hit.get("slug", "")
                    link = f"https://ourworldindata.org/{slug}" if slug else hit.get("url", "")
                    if not link:
                        continue
                    sources.append(ScoutSource(
                        url=link,
                        title=(hit.get("title") or "").strip(),
                        published_at=_to_iso(hit.get("publishedAt") or hit.get("date", "")),
                        source_type="owid",
                        verified="ourworldindata.org" in link,
                        source_tier="A",
                        snippet=(hit.get("excerpt") or hit.get("summary", ""))[:500],
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: OWID GitHub data catalog (curated datasets index)
        gh_url = (
            "https://api.github.com/search/code"
            f"?q={quote_plus(query)}+repo:owid/owid-datasets+extension:csv"
            "&sort=indexed&order=desc&per_page=8"
        )
        resp2 = await self._fetch(
            gh_url,
            headers={"Accept": "application/vnd.github.v3+json"},
            use_cache_headers=False,
        )
        if resp2.status_code == 200:
            try:
                data = resp2.json()
                for item in data.get("items", [])[:8]:
                    link = item.get("html_url", "")
                    name = item.get("name", "")
                    sources.append(ScoutSource(
                        url=link,
                        title=f"OWID dataset: {name}",
                        published_at="",
                        source_type="owid",
                        verified="github.com/owid" in link,
                        source_tier="A",
                    ))
            except Exception:
                pass
        return sources[:8]


# ===========================================================================
# 10. HDXCrawler
# ===========================================================================

class HDXCrawler(CrawlerBase):
    """HDX (Humanitarian Data Exchange) — CKAN-based humanitarian datasets."""

    source_id = "hdx"
    default_poll_interval = 7200

    async def crawl(self, query: str) -> list[ScoutSource]:
        # HDX CKAN API (public, no key required)
        url = (
            f"https://data.humdata.org/api/3/action/package_search"
            f"?q={quote_plus(query)}&rows=8&sort=score+desc"
            f"&fq=last_modified:[{_days_ago(365)}T00:00:00Z+TO+NOW]"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
        try:
            result = data.get("result", {})
            for pkg in result.get("results", []):
                name = pkg.get("name", "")
                link = f"https://data.humdata.org/dataset/{name}" if name else ""
                if not link:
                    continue
                sources.append(ScoutSource(
                    url=link,
                    title=(pkg.get("title") or name).strip(),
                    published_at=_to_iso(pkg.get("metadata_modified") or pkg.get("last_modified", "")),
                    source_type="hdx",
                    verified=True,
                    source_tier="A",
                    snippet=(pkg.get("notes") or "")[:500],
                ))
        except Exception as exc:
            logger.debug("[hdx] parse error: %s", exc)
        return sources


# ===========================================================================
# Registry + Public Entry-Point (Enterprise Edition)
# ===========================================================================

_CRAWLERS: dict[str, CrawlerBase] = {
    c.source_id: c
    for c in [
        IMFCrawler(),
        WorldBankCrawler(),
        FREDCrawler(),
        ECBCrawler(),
        BISCrawler(),
        EurostatCrawler(),
        UNCrawler(),
        IRENACrawler(),
        OurWorldInDataCrawler(),
        HDXCrawler(),
    ]
}


async def _run_single_crawler_safe(crawler: CrawlerBase, query: str, semaphore: asyncio.Semaphore) -> List[ScoutSource]:
    """Wraper wykonujący zapytanie chronione Semaforem z bezbłędnym przechwyceniem awarii."""
    async with semaphore:
        try:
            return await crawler.safe_crawl(query)
        except Exception as exc:
            logger.error(f"[Layer C] Fatal exception in crawler {crawler.source_id}: {exc}", exc_info=True)
            return []


async def run_layer_c(
    query: str,
    enabled: Optional[list[str]] = None,
) -> list[ScoutSource]:
    """
    Run all (or a subset of) Layer C crawlers in parallel.
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

    # Limitujemy współbieżność, chroniąc system przed dławieniem API
    concurrency_limit = getattr(settings, "crawler_concurrency_limit", 10)
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [_run_single_crawler_safe(c, query, semaphore) for c in crawlers]

    # Globalny limit czasu dla całej warstwy
    layer_timeout = getattr(settings, "layer_c_timeout_seconds", 60.0)

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=layer_timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"[Layer C] Przekroczono globalny limit czasu ({layer_timeout}s). Niektóre wyniki mogły zostać utracone.")
        return []

    seen_urls: set[str] = set()
    sources: list[ScoutSource] = []
    
    # Dekompozycja wyników i logowanie błędów
    for idx, batch in enumerate(results):
        if isinstance(batch, Exception):
            logger.error(f"[Layer C] Zgromadzono wyjątek z crawlera {crawlers[idx].source_id}: {batch}")
            continue
        if not isinstance(batch, list):
            continue
            
        for src in batch:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                sources.append(src)
                
    return sources


def get_crawler_status_c() -> list[dict]:
    """Return status dict for each Layer C crawler."""
    return [
        {
            "source_id": c.source_id,
            "layer": "C",
            "poll_interval": c.poll_interval,
            "is_paused": c.is_paused,
            "last_seen_id": c.last_seen_id,
            "consecutive_errors": c._consecutive_errors,
        }
        for c in _CRAWLERS.values()
    ]
