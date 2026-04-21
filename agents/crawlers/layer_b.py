"""
agents/crawlers/layer_b.py — Layer B: Legislation & Regulatory crawlers (10 sources) - ENTERPRISE EDITION

Sources covered:
  1.  EurLexCrawler          — EUR-Lex Atom/RSS (EU regulations, directives)
  2.  CURIACrawler           — CURIA Court of Justice recent judgments RSS
  3.  FederalRegisterCrawler — US Federal Register JSON API
  4.  SECEdgarCrawler        — SEC EDGAR full-text search JSON API
  5.  ESMACrawler            — ESMA press releases & consultations Atom
  6.  EBACrawler             — EBA publications RSS (banking regulation)
  7.  OECDCrawler            — OECD iLibrary search JSON
  8.  WTOCrawler             — WTO documents search RSS/JSON
  9.  WIPOCrawler            — WIPO PATENTSCOPE / IP Portal JSON
  10. EPOCrawler             — EPO Espacenet / OPS REST API

Ulepszenia PRO:
  - Bounded Concurrency (Semaphore): Limitowanie równoległych żądań do API instytucjonalnych.
  - Global Layer Timeout: Ochrona przed nieskończonym oczekiwaniem na odpowiedź rządowych serwerów.
  - Exception Unpacking: Dokładne logowanie błędów ze złączonych procesów (asyncio.gather).
  - Safe Payload Parsing: Wzmocniona obrona przed nieprawidłowym JSON/XML przy statusie 200 OK.
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

logger = logging.getLogger("foundry.agents.crawlers.layer_b")

# ---------------------------------------------------------------------------
# Date helpers (mirror layer_a)
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


# ---------------------------------------------------------------------------
# Atom/RSS helpers (mirror layer_a)
# ---------------------------------------------------------------------------

_ATOM_NS = "http://www.w3.org/2005/Atom"
_AT = f"{{{_ATOM_NS}}}"

def _atom_text(el: Optional[ET.Element], tag: str) -> str:
    child = el.find(f"{_AT}{tag}") if el is not None else None
    return (child.text or "").strip() if child is not None else ""

def _atom_link(el: ET.Element, rel: str = "alternate") -> str:
    for link in el.findall(f"{_AT}link"):
        if link.get("rel", "alternate") == rel:
            return link.get("href", "")
    for link in el.findall(f"{_AT}link"):
        if link.get("type", "") == "text/html":
            return link.get("href", "")
    lnk = el.find(f"{_AT}link")
    return lnk.get("href", "") if lnk is not None else ""

def _rss_text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None else ""


# ===========================================================================
# 1. EurLexCrawler
# ===========================================================================

class EurLexCrawler(CrawlerBase):
    """EUR-Lex RSS search — EU regulations, directives, decisions (Tier S)."""

    source_id = "eurlex"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            "https://eur-lex.europa.eu/search.html"
            f"?type=quick&lang=EN&text={quote_plus(query)}"
            "&scope=EURLEX&FM_CODED=REG%2CDIR%2CDEC"
            "&SORT=VALIDITY_DATE%3Adesc&HITS_PER_PAGE=10&RSS=true"
        )
        resp = await self._fetch(url, use_cache_headers=True)
        if resp.status_code != 200:
            return []
        sources: list[ScoutSource] = []
        try:
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item"):
                link = _rss_text(item, "link")
                if not link:
                    link = _rss_text(item, "guid")
                if not link or "eur-lex.europa.eu" not in link:
                    continue
                sources.append(ScoutSource(
                    url=link,
                    title=_rss_text(item, "title"),
                    published_at=_to_iso(_rss_text(item, "pubDate") or _rss_text(item, "dc:date")),
                    source_type="eurlex",
                    verified=True,
                    source_tier="S",
                    snippet=_rss_text(item, "description")[:500],
                ))
        except ET.ParseError as exc:
            logger.debug("[eurlex] XML parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 2. CURIACrawler
# ===========================================================================

class CURIACrawler(CrawlerBase):
    """CURIA — Court of Justice of the EU recent judgments and opinions RSS."""

    source_id = "curia"
    default_poll_interval = 1800

    _FEEDS = [
        "https://curia.europa.eu/jcms/jcms/p1_1382581/",   # recent judgments
        "https://curia.europa.eu/jcms/jcms/p1_1382582/",   # recent AG opinions
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
                    desc  = _rss_text(item, "description")
                    link  = _rss_text(item, "link") or _rss_text(item, "guid")
                    pub   = _rss_text(item, "pubDate")
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
                        source_type="curia",
                        verified="curia.europa.eu" in link,
                        source_tier="S",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[curia] RSS parse error feed=%s: %s", feed_url, exc)
        return sources[:8]


# ===========================================================================
# 3. FederalRegisterCrawler
# ===========================================================================

class FederalRegisterCrawler(CrawlerBase):
    """US Federal Register JSON API — proposed rules, final rules, notices."""

    source_id = "federalregister"
    default_poll_interval = 900

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            "https://www.federalregister.gov/api/v1/articles.json"
            f"?conditions[term]={quote_plus(query)}"
            f"&conditions[publication_date][gte]={_days_ago(365)}"
            "&conditions[type][]=RULE&conditions[type][]=PRORULE&conditions[type][]=NOTICE"
            "&per_page=8&order=newest"
            "&fields[]=title&fields[]=html_url&fields[]=publication_date"
            "&fields[]=abstract&fields[]=document_number"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
        for art in data.get("results", []):
            link = art.get("html_url", "")
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(art.get("title") or "").strip(),
                published_at=_to_iso(art.get("publication_date", "")),
                source_type="federalregister",
                verified=True,
                source_tier="S",
                snippet=(art.get("abstract") or "")[:500],
            ))
        return sources


# ===========================================================================
# 4. SECEdgarCrawler
# ===========================================================================

class SECEdgarCrawler(CrawlerBase):
    """SEC EDGAR full-text search — 10-K, 10-Q, 8-K, DEF 14A filings."""

    source_id = "secedgar"
    default_poll_interval = 900

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            "https://efts.sec.gov/LATEST/search-index"
            f"?q={quote_plus(query)}"
            f"&dateRange=custom&startdt={_days_ago(365)}&enddt={_days_ago(0)}"
            "&forms=10-K,10-Q,8-K,DEF+14A"
            "&hits.hits._source=period_of_report,file_date,display_names,biz_location"
        )
        # Alternative public endpoint
        url2 = (
            f"https://efts.sec.gov/LATEST/search-index?q={quote_plus(query)}"
            f"&dateRange=custom&startdt={_days_ago(365)}"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            resp = await self._fetch(url2, use_cache_headers=False)
            
        if resp.status_code != 200:
            return []
            
        sources: list[ScoutSource] = []
        try:
            hits = resp.json().get("hits", {}).get("hits", [])
            for hit in hits[:8]:
                src = hit.get("_source", {})
                file_id = hit.get("_id", "")
                link = f"https://www.sec.gov/Archives/edgar/full-index/{file_id}" if file_id else ""
                if not link:
                    continue
                names = src.get("display_names", [])
                entity = names[0].get("name", "") if names else ""
                title = f"{entity} SEC Filing".strip() if entity else "SEC Filing"
                sources.append(ScoutSource(
                    url=link,
                    title=title,
                    published_at=_to_iso(src.get("file_date", "")),
                    source_type="secedgar",
                    verified=True,
                    source_tier="S",
                ))
        except Exception as exc:
            logger.debug("[secedgar] parse error: %s", exc)
        return sources


# ===========================================================================
# 5. ESMACrawler
# ===========================================================================

class ESMACrawler(CrawlerBase):
    """ESMA — European Securities and Markets Authority publications & consultations."""

    source_id = "esma"
    default_poll_interval = 1800

    async def crawl(self, query: str) -> list[ScoutSource]:
        # ESMA search API (public JSON endpoint)
        url = (
            "https://www.esma.europa.eu/search-documents"
            f"?q={quote_plus(query)}&type=all&language=en&size=8"
        )
        search_resp = await self._fetch(url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        if search_resp.status_code != 200:
            logger.debug("[esma] primary search status: %s", search_resp.status_code)

        if search_resp.status_code == 200:
            try:
                data = search_resp.json()
                results = data.get("results", data.get("items", []))
                for r in results[:8]:
                    link = r.get("url") or r.get("link", "")
                    if link and not link.startswith("http"):
                        link = "https://www.esma.europa.eu" + link
                    if not link:
                        continue
                    sources.append(ScoutSource(
                        url=link,
                        title=(r.get("title") or r.get("name", "")).strip(),
                        published_at=_to_iso(r.get("date") or r.get("publication_date", "")),
                        source_type="esma",
                        verified="esma.europa.eu" in link,
                        source_tier="A",
                        snippet=(r.get("description") or r.get("summary", ""))[:500],
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: ESMA press release RSS
        rss_url = "https://www.esma.europa.eu/press-news/esma-news.rss"
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
                    source_type="esma",
                    verified="esma.europa.eu" in link,
                    source_tier="A",
                    snippet=desc[:500],
                ))
        except ET.ParseError as exc:
            logger.debug("[esma] RSS parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 6. EBACrawler
# ===========================================================================

class EBACrawler(CrawlerBase):
    """EBA — European Banking Authority publications and regulatory standards."""

    source_id = "eba"
    default_poll_interval = 1800

    _FEEDS = [
        "https://www.eba.europa.eu/newsroom/news.rss",
        "https://www.eba.europa.eu/regulation-and-policy.rss",
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
                    combined = (title + " " + desc).lower()
                    if ql and not any(kw in combined for kw in ql):
                        continue
                    seen.add(link)
                    sources.append(ScoutSource(
                        url=link,
                        title=title.strip(),
                        published_at=_to_iso(pub),
                        source_type="eba",
                        verified="eba.europa.eu" in link,
                        source_tier="A",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[eba] RSS parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 7. OECDCrawler
# ===========================================================================

class OECDCrawler(CrawlerBase):
    """OECD iLibrary search JSON — economic policy, governance, environment."""

    source_id = "oecd"
    default_poll_interval = 1800

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            f"https://www.oecd-ilibrary.org/search/search-results.json"
            f"?q={quote_plus(query)}&lang=en&cr=oecd&dy=2023&sort=relevance&n=8"
        )
        search_resp = await self._fetch(url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        if search_resp.status_code != 200:
            logger.debug("[oecd] primary search status: %s", search_resp.status_code)

        if search_resp.status_code == 200:
            try:
                data = search_resp.json()
                items = data.get("items", data.get("results", []))
                for item in items[:8]:
                    link = item.get("url") or item.get("link", "")
                    if link and not link.startswith("http"):
                        link = "https://www.oecd-ilibrary.org" + link
                    if not link:
                        continue
                    sources.append(ScoutSource(
                        url=link,
                        title=(item.get("title") or "").strip(),
                        published_at=_to_iso(item.get("date") or item.get("publicationDate", "")),
                        source_type="oecd",
                        verified="oecd" in link,
                        source_tier="A",
                        snippet=(item.get("summary") or item.get("description", ""))[:500],
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: OECD OpenAlex filter
        url2 = (
            f"https://api.openalex.org/works"
            f"?search={quote_plus(query)}"
            f"&filter=institutions.ror:https://ror.org/04k8n6289,publication_year:>2022"
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
                link = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else "")
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(work.get("title") or "").strip(),
                published_at=_to_iso(work.get("publication_date", "")),
                source_type="oecd",
                verified=False,
                source_tier="A",
            ))
        return sources[:8]


# ===========================================================================
# 8. WTOCrawler
# ===========================================================================

class WTOCrawler(CrawlerBase):
    """WTO document gateway — trade policy, dispute settlement, agreements."""

    source_id = "wto"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        # WTO public document search
        url = (
            "https://www.wto.org/english/res_e/search_e/search_e.htm"
            f"?q={quote_plus(query)}&langFlt=&maxRec=8&SortBy=date&SortOrder=desc"
        )
        search_resp = await self._fetch(url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        if search_resp.status_code != 200:
            logger.debug("[wto] primary search status: %s", search_resp.status_code)

        # Try WTO news RSS as more reliable fallback
        rss_url = "https://www.wto.org/rss/english/news_e.rss"
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
                        source_type="wto",
                        verified="wto.org" in link,
                        source_tier="A",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[wto] RSS parse error: %s", exc)

        if sources:
            return sources[:8]

        # Fallback: WTO documents via OpenAlex institution filter
        url3 = (
            f"https://api.openalex.org/works"
            f"?search={quote_plus(query)}"
            f"&filter=institutions.ror:https://ror.org/01h8c5k52,publication_year:>2022"
            f"&sort=publication_date:desc&per-page=8"
            f"&select=id,title,doi,publication_date,open_access"
        )
        resp3 = await self._fetch(url3, use_cache_headers=False)
        if resp3.status_code == 200:
            try:
                data = resp3.json()
            except ValueError:
                return sources[:8]
                
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
                    source_type="wto",
                    verified=False,
                    source_tier="A",
                ))
        return sources[:8]


# ===========================================================================
# 9. WIPOCrawler
# ===========================================================================

class WIPOCrawler(CrawlerBase):
    """WIPO PATENTSCOPE + IP Portal — international IP, patents, treaties."""

    source_id = "wipo"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        # WIPO PATENTSCOPE search API (public)
        url = (
            "https://patentscope.wipo.int/search/en/rest/servicejson.jsf"
            f"?office=PCT&query={quote_plus(query)}&_=1&maxRec=8"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        sources: list[ScoutSource] = []
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                results = data.get("resultsList", {}).get("wipo:PCT", [])
                if isinstance(results, dict):
                    results = [results]
                for r in results[:8]:
                    app_num = r.get("wipo:internationalApplicationNumber", "")
                    link = (
                        f"https://patentscope.wipo.int/search/en/detail.jsf"
                        f"?docId={app_num}"
                    ) if app_num else ""
                    if not link:
                        continue
                    sources.append(ScoutSource(
                        url=link,
                        title=(r.get("wipo:inventionTitle") or r.get("title", "")).strip(),
                        published_at=_to_iso(r.get("wipo:internationalFilingDate", "")),
                        source_type="wipo",
                        verified=True,
                        source_tier="S",
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: WIPO news RSS
        rss_url = "https://www.wipo.int/rss/en/wipo/news.rss"
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
                    source_type="wipo",
                    verified="wipo.int" in link,
                    source_tier="S",
                    snippet=desc[:500],
                ))
        except ET.ParseError as exc:
            logger.debug("[wipo] RSS parse error: %s", exc)
        return sources[:8]


# ===========================================================================
# 10. EPOCrawler
# ===========================================================================

class EPOCrawler(CrawlerBase):
    """EPO Espacenet — European Patent Office patent search (public OPS API)."""

    source_id = "epo"
    default_poll_interval = 3600

    async def crawl(self, query: str) -> list[ScoutSource]:
        # EPO OPS REST API (free, no key for basic search)
        url = (
            "https://ops.epo.org/3.2/rest-services/published-data/search"
            f"?q=txt%3D{quote_plus(query)}"
            "&Range=1-8"
        )
        resp = await self._fetch(
            url,
            headers={"Accept": "application/json"},
            use_cache_headers=False,
        )
        sources: list[ScoutSource] = []
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                results = (
                    data.get("ops:world-patent-data", {})
                    .get("ops:biblio-search", {})
                    .get("ops:search-result", {})
                    .get("ops:publication-reference", [])
                )
                if isinstance(results, dict):
                    results = [results]
                for ref in results[:8]:
                    doc = ref.get("document-id", {})
                    country = doc.get("country", {}).get("$", "")
                    docnum  = doc.get("doc-number", {}).get("$", "")
                    kind    = doc.get("kind", {}).get("$", "")
                    if not docnum:
                        continue
                    ep_num = f"{country}{docnum}{kind}".strip()
                    link = f"https://worldwide.espacenet.com/patent/search?q={ep_num}"
                    sources.append(ScoutSource(
                        url=link,
                        title=f"Patent {ep_num}",
                        published_at="",
                        source_type="epo",
                        verified=True,
                        source_tier="A",
                    ))
                if sources:
                    return sources
            except Exception:
                pass

        # Fallback: Espacenet CQL search (HTML but stable URL pattern)
        url2 = (
            f"https://worldwide.espacenet.com/patent/search"
            f"?q={quote_plus(query)}"
        )
        sources.append(ScoutSource(
            url=url2,
            title=f"EPO patent search: {query[:80]}",
            published_at="",
            source_type="epo",
            verified=True,
            source_tier="A",
        ))
        return sources[:8]


# ===========================================================================
# Registry + Public Entry-Point (Enterprise Edition)
# ===========================================================================

_CRAWLERS: dict[str, CrawlerBase] = {
    c.source_id: c
    for c in [
        EurLexCrawler(),
        CURIACrawler(),
        FederalRegisterCrawler(),
        SECEdgarCrawler(),
        ESMACrawler(),
        EBACrawler(),
        OECDCrawler(),
        WTOCrawler(),
        WIPOCrawler(),
        EPOCrawler(),
    ]
}


async def _run_single_crawler_safe(crawler: CrawlerBase, query: str, semaphore: asyncio.Semaphore) -> List[ScoutSource]:
    """Wraper wykonujący zapytanie chronione Semaforem z bezbłędnym przechwyceniem awarii."""
    async with semaphore:
        try:
            return await crawler.safe_crawl(query)
        except Exception as exc:
            logger.error(f"[Layer B] Fatal exception in crawler {crawler.source_id}: {exc}", exc_info=True)
            return []


async def run_layer_b(
    query: str,
    enabled: Optional[list[str]] = None,
) -> list[ScoutSource]:
    """
    Run all (or a subset of) Layer B crawlers in parallel.
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

    # Limitujemy współbieżność, chroniąc system przed limitowaniem API rządowych
    concurrency_limit = getattr(settings, "crawler_concurrency_limit", 10)
    semaphore = asyncio.Semaphore(concurrency_limit)

    task_map = {
        asyncio.create_task(_run_single_crawler_safe(c, query, semaphore)): c
        for c in crawlers
    }

    # Globalny limit czasu dla całej warstwy
    layer_timeout = getattr(settings, "layer_b_timeout_seconds", 60.0)

    done, pending = await asyncio.wait(task_map.keys(), timeout=layer_timeout)
    if pending:
        for task in pending:
            task.cancel()
        logger.warning(
            "[Layer B] Przekroczono globalny limit czasu (%.1fs). partial_success=%d/%d, pending=%d.",
            layer_timeout, len(done), len(task_map), len(pending),
        )

    seen_urls: set[str] = set()
    sources: list[ScoutSource] = []
    
    # Dekompozycja wyników
    for task in done:
        crawler = task_map[task]
        if task.cancelled():
            batch = []
        else:
            try:
                batch = task.result()
            except Exception as exc:
                batch = exc
        if isinstance(batch, Exception):
            logger.error(f"[Layer B] Zgromadzono wyjątek z crawlera {crawler.source_id}: {batch}")
            continue
        if not isinstance(batch, list):
            continue
            
        for src in batch:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                sources.append(src)
    logger.info(
        "[Layer B] health summary: ok=%d/%d timed_out=%d discovered=%d",
        len(done), len(task_map), len(pending), len(sources),
    )
                
    return sources


def get_crawler_status_b() -> list[dict]:
    """Return status dict for each Layer B crawler."""
    return [
        {
            "source_id": c.source_id,
            "layer": "B",
            "poll_interval": c.poll_interval,
            "is_paused": c.is_paused,
            "last_seen_id": c.last_seen_id,
            "consecutive_errors": c._consecutive_errors,
        }
        for c in _CRAWLERS.values()
    ]
