"""
agents/crawlers/layer_a.py — Layer A: Science & Research crawlers (14 sources) - ENTERPRISE EDITION

Sources covered:
  1.  ArxivCrawler          — Atom feed, cs.* / stat.* / econ.* / q-bio.*
  2.  BioRxivCrawler        — REST JSON /api/biorxiv (life sciences preprints)
  3.  MedRxivCrawler        — REST JSON /api/medrxiv (medical preprints)
  4.  SemanticScholarCrawler— REST /graph/v1/paper/search (free, no key)
  5.  PubMedCrawler         — E-utilities esearch + efetch (NCBI PMC)
  6.  EuroPMCCrawler        — REST /search (European biomedical)
  7.  CORECrawler           — API v3 /search/works (40M+ OA docs; needs key)
  8.  SSRNCrawler           — RSS per-network (economics, law, finance)
  9.  PhilPapersCrawler     — REST /api/search (philosophy, AI alignment)
  10. IEEEXploreCrawler     — REST /api/v1/search (EE, AI; needs key)
  11. ACMCrawler            — OpenAlex filtered by ACM venue (no auth needed)
  12. BASECrawler           — REST JSON search (350M+ cross-repo)
  13. ChemRxivCrawler       — REST JSON /api-gateway/chemrxiv (chemistry)
  14. EngrXivCrawler        — OSF SHARE API (engineering preprints)

Ulepszenia PRO:
  - Bounded Concurrency (Semaphore): Limitowanie równoległych żądań do API, ochrona łącza.
  - Global Layer Timeout: Gwarancja, że cała warstwa zwróci wyniki w skończonym czasie.
  - Exception Unpacking: Dokładne logowanie błędów ze złączonych procesów (asyncio.gather).
  - Safe Payload Parsing: Wzmocniona obrona przed nieprawidłowym JSON/XML przy statusie 200 OK.
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

logger = logging.getLogger("foundry.agents.crawlers.layer_a")

# ---------------------------------------------------------------------------
# Date helpers (Zoptymalizowane)
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
)

def _to_iso(date_str: str) -> str:
    """Normalise any recognised date string to ISO-8601. Returns '' on failure."""
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
    """Return YYYY-MM-DD for n days ago (UTC)."""
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Atom / RSS namespace constants (stdlib ET)
# ---------------------------------------------------------------------------

_ATOM_NS = "http://www.w3.org/2005/Atom"
_AT = f"{{{_ATOM_NS}}}"   # shorthand: f"{_AT}entry"

def _atom_text(el: Optional[ET.Element], tag: str) -> str:
    child = el.find(f"{_AT}{tag}") if el is not None else None
    return (child.text or "").strip() if child is not None else ""

def _atom_link(el: ET.Element, rel: str = "alternate") -> str:
    """Return href of first <link rel=rel> or first <link> if rel not found."""
    for link in el.findall(f"{_AT}link"):
        if link.get("rel", "alternate") == rel:
            return link.get("href", "")
    # fallback: any link with type text/html
    for link in el.findall(f"{_AT}link"):
        if link.get("type", "") == "text/html":
            return link.get("href", "")
    lnk = el.find(f"{_AT}link")
    return lnk.get("href", "") if lnk is not None else ""

def _rss_text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None else ""


# ===========================================================================
# 1. ArxivCrawler
# ===========================================================================

class ArxivCrawler(CrawlerBase):
    """arXiv Atom API — cs.*, stat.*, econ.*, q-bio.* categories."""

    source_id = "arxiv"
    default_poll_interval = 120

    _CATS = "cat:cs.* OR cat:stat.* OR cat:econ.* OR cat:q-bio.*"

    async def crawl(self, query: str) -> list[ScoutSource]:
        q = quote_plus(f"all:{query} AND ({self._CATS})")
        url = (
            f"https://export.arxiv.org/api/query"
            f"?search_query={q}&sortBy=submittedDate&sortOrder=descending&max_results=10"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []

        sources: list[ScoutSource] = []
        try:
            root = ET.fromstring(resp.text)
            for entry in root.findall(f"{_AT}entry"):
                link = _atom_link(entry)
                if not link or "arxiv.org" not in link:
                    continue
                arxiv_id = _atom_text(entry, "id").split("/")[-1]
                if arxiv_id and self.last_seen_id and arxiv_id <= self.last_seen_id:
                    continue
                sources.append(ScoutSource(
                    url=link,
                    title=_atom_text(entry, "title").replace("\n", " "),
                    published_at=_to_iso(_atom_text(entry, "published")),
                    source_type="arxiv",
                    verified=True,
                    source_tier="S",
                ))
                if arxiv_id and (not self.last_seen_id or arxiv_id > self.last_seen_id):
                    self.last_seen_id = arxiv_id
        except ET.ParseError as exc:
            logger.warning("[arxiv] XML parse error: %s", exc)
        return sources


# ===========================================================================
# 2+3. BioRxiv + MedRxiv
# ===========================================================================

class _BioMedRxivBase(CrawlerBase):
    _SERVER: str  # "biorxiv" or "medrxiv"

    async def crawl(self, query: str) -> list[ScoutSource]:
        since = _days_ago(14)
        until = _days_ago(0)
        url = f"https://api.biorxiv.org/details/{self._SERVER}/{since}/{until}/0/json"
        resp = await self._fetch(url)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            logger.warning(f"[{self._SERVER}] Otrzymano nieprawidłowy JSON mimo statusu 200.")
            return []
            
        collection = data.get("collection", [])
        sources: list[ScoutSource] = []
        ql = query.lower().split()
        
        for paper in collection[:20]:
            title: str = paper.get("title", "")
            abstract: str = paper.get("abstract", "")
            combined = (title + " " + abstract).lower()
            if not any(kw in combined for kw in ql):
                continue
            doi = paper.get("doi", "")
            paper_url = f"https://doi.org/{doi}" if doi else ""
            if not paper_url:
                continue
            sources.append(ScoutSource(
                url=paper_url,
                title=title.strip(),
                published_at=_to_iso(paper.get("date", "")),
                source_type=self._SERVER,
                verified=False,           # DOI URL verified downstream
                source_tier="A",
                snippet=abstract[:500],
            ))
        return sources[:8]


class BioRxivCrawler(_BioMedRxivBase):
    source_id = "biorxiv"
    default_poll_interval = 300
    _SERVER = "biorxiv"


class MedRxivCrawler(_BioMedRxivBase):
    source_id = "medrxiv"
    default_poll_interval = 300
    _SERVER = "medrxiv"


# ===========================================================================
# 4. SemanticScholarCrawler
# ===========================================================================

class SemanticScholarCrawler(CrawlerBase):
    """Semantic Scholar REST API — free, no key, 100 req/5 min."""

    source_id = "semanticscholar"
    default_poll_interval = 180

    _FIELDS = "title,year,publicationDate,openAccessPdf,externalIds,abstract"

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={quote_plus(query)}&fields={self._FIELDS}&limit=10"
            f"&publicationDateOrYear=2024:"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            logger.warning("[semanticscholar] Invalid JSON payload received.")
            return []
            
        sources: list[ScoutSource] = []
        for paper in data.get("data", []):
            oa = paper.get("openAccessPdf") or {}
            paper_url = oa.get("url", "")
            if not paper_url:
                ext = paper.get("externalIds") or {}
                doi = ext.get("DOI", "")
                paper_url = f"https://doi.org/{doi}" if doi else ""
            if not paper_url:
                continue
            s2_id = paper.get("paperId", "")
            if s2_id and self.last_seen_id and s2_id == self.last_seen_id:
                break
            sources.append(ScoutSource(
                url=paper_url,
                title=(paper.get("title") or "").strip(),
                published_at=_to_iso(paper.get("publicationDate") or str(paper.get("year", ""))),
                source_type="semanticscholar",
                verified=False,
                source_tier=_get_source_tier(paper_url),
                snippet=(paper.get("abstract") or "")[:500],
            ))
            
        if sources:
            first_id = (data.get("data") or [{}])[0].get("paperId", "")
            if first_id:
                self.last_seen_id = first_id
        return sources


# ===========================================================================
# 5. PubMedCrawler
# ===========================================================================

class PubMedCrawler(CrawlerBase):
    """NCBI PubMed Central via E-utilities (esearch + efetch)."""

    source_id = "pubmed"
    default_poll_interval = 300

    _ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    _ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    async def crawl(self, query: str) -> list[ScoutSource]:
        # Step 1 — esearch: get PMC IDs
        search_url = (
            f"{self._ESEARCH}?db=pubmed&term={quote_plus(query)}"
            f"&sort=pub_date&retmax=8&retmode=json&mindate={_days_ago(365)}&datetype=pdat"
        )
        resp = await self._fetch(search_url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            esearch = resp.json()
        except ValueError:
            return []
            
        ids = esearch.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2 — esummary: get titles + dates
        id_str = ",".join(ids[:8])
        summ_url = f"{self._ESUMMARY}?db=pubmed&id={id_str}&retmode=json"
        resp2 = await self._fetch(summ_url, use_cache_headers=False)
        if resp2.status_code != 200:
            return []
            
        try:
            summ = resp2.json().get("result", {})
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
        for pmid in ids[:8]:
            doc = summ.get(str(pmid), {})
            title = doc.get("title", "").strip()
            if not title:
                continue
            pub_date = doc.get("pubdate", "")
            sources.append(ScoutSource(
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                title=title,
                published_at=_to_iso(pub_date),
                source_type="pubmed",
                verified=True,
                source_tier="S",
            ))
        return sources


# ===========================================================================
# 6. EuroPMCCrawler
# ===========================================================================

class EuroPMCCrawler(CrawlerBase):
    """Europe PMC REST search — European biomedical literature."""

    source_id = "europepmc"
    default_poll_interval = 300

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={quote_plus(query)}"
            f"&resultType=lite&pageSize=8&format=json&sort=%25RECENTLY_ADDED"
        )
        resp = await self._fetch(url)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        results = data.get("resultList", {}).get("result", [])
        sources: list[ScoutSource] = []
        
        for r in results:
            pmid = r.get("pmid") or r.get("pmcid") or ""
            if r.get("pmcid"):
                link = f"https://europepmc.org/article/PMC/{r['pmcid']}"
            elif pmid:
                link = f"https://europepmc.org/article/med/{pmid}"
            else:
                doi = r.get("doi", "")
                link = f"https://doi.org/{doi}" if doi else ""
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(r.get("title") or "").strip().rstrip("."),
                published_at=_to_iso(r.get("firstPublicationDate", "")),
                source_type="europepmc",
                verified="europepmc.org" in link,
                source_tier="A",
            ))
        return sources


# ===========================================================================
# 7. CORECrawler
# ===========================================================================

class CORECrawler(CrawlerBase):
    """CORE.ac.uk API v3 — 40M+ open-access documents. Needs CORE_API_KEY."""

    source_id = "core"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        from config.settings import settings
        api_key = getattr(settings, "core_api_key", "")
        if not api_key:
            logger.debug("[core] CORE_API_KEY not set — skipping")
            return []

        url = (
            f"https://api.core.ac.uk/v3/search/works"
            f"?q={quote_plus(query)}&limit=8&sort=publishedDate:desc"
        )
        resp = await self._fetch(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            use_cache_headers=False,
        )
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
        for work in data.get("results", []):
            link = (work.get("downloadUrl") or work.get("doi")
                    or work.get("identifiers", [{}])[0].get("identifier", ""))
            if not link:
                continue
            if link.startswith("10."):
                link = f"https://doi.org/{link}"
            sources.append(ScoutSource(
                url=link,
                title=(work.get("title") or "").strip(),
                published_at=_to_iso(work.get("publishedDate", "")),
                source_type="core",
                verified=False,
                source_tier="A",
                snippet=(work.get("abstract") or "")[:500],
            ))
        return sources


# ===========================================================================
# 8. SSRNCrawler
# ===========================================================================

class SSRNCrawler(CrawlerBase):
    """SSRN multi-network RSS (economics, law, finance, management)."""

    source_id = "ssrn"
    default_poll_interval = 900   # 15 min — low-frequency

    _FEEDS = [
        "https://papers.ssrn.com/sol3/JELJOUR_Results.cfm"
        "?form_name=journalBrowse&journal_id=1005628&Network=no&lim=false&count=10&sort=ab_approval_date&RSS=true",
        "https://papers.ssrn.com/sol3/JELJOUR_Results.cfm"
        "?form_name=journalBrowse&journal_id=1004611&Network=no&lim=false&count=10&sort=ab_approval_date&RSS=true",
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
                    link  = _rss_text(item, "link")
                    pub   = _rss_text(item, "pubDate")
                    combined = (title + " " + desc).lower()
                    if not link or link in seen:
                        continue
                    if ql and not any(kw in combined for kw in ql):
                        continue
                    seen.add(link)
                    sources.append(ScoutSource(
                        url=link,
                        title=title.strip(),
                        published_at=_to_iso(pub),
                        source_type="ssrn",
                        verified="ssrn.com" in link,
                        source_tier="A",
                        snippet=desc[:500],
                    ))
            except ET.ParseError as exc:
                logger.debug("[ssrn] RSS parse error: %s", exc)
                
        return sources[:8]


# ===========================================================================
# 9. PhilPapersCrawler
# ===========================================================================

class PhilPapersCrawler(CrawlerBase):
    """PhilPapers.org REST API — philosophy, AI ethics, alignment."""

    source_id = "philpapers"
    default_poll_interval = 900

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            f"https://philpapers.org/api/search.json"
            f"?query={quote_plus(query)}&limit=8&method=title+abstract"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        entries = data if isinstance(data, list) else data.get("entries", data.get("results", []))
        sources: list[ScoutSource] = []
        
        for entry in entries[:8]:
            link = entry.get("url") or entry.get("link", "")
            if not link:
                doi = entry.get("doi", "")
                link = f"https://doi.org/{doi}" if doi else ""
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(entry.get("title") or "").strip(),
                published_at=_to_iso(entry.get("date") or entry.get("year", "")),
                source_type="philpapers",
                verified="philpapers.org" in link,
                source_tier="A",
            ))
        return sources


# ===========================================================================
# 10. IEEEXploreCrawler
# ===========================================================================

class IEEEXploreCrawler(CrawlerBase):
    """IEEE Xplore REST API — electrical engineering, AI, robotics. Needs key."""

    source_id = "ieee"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        from config.settings import settings
        api_key = getattr(settings, "ieee_api_key", "")
        if not api_key:
            logger.debug("[ieee] IEEE_API_KEY not set — skipping")
            return []

        url = (
            f"https://ieeexploreapi.ieee.org/api/v1/search/articles"
            f"?apikey={api_key}&querytext={quote_plus(query)}"
            f"&max_records=8&sort_order=desc&sort_field=publication_date"
            f"&start_year=2023"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        articles = data.get("articles", [])
        sources: list[ScoutSource] = []
        
        for art in articles:
            link = art.get("html_url") or art.get("pdf_url", "")
            if not link:
                doi = art.get("doi", "")
                link = f"https://doi.org/{doi}" if doi else ""
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(art.get("title") or "").strip(),
                published_at=_to_iso(art.get("publication_date", "")),
                source_type="ieee",
                verified="ieee.org" in link or "ieeexplore" in link,
                source_tier="A",
                snippet=(art.get("abstract") or "")[:500],
            ))
        return sources


# ===========================================================================
# 11. ACMCrawler
# ===========================================================================

class ACMCrawler(CrawlerBase):
    """ACM Digital Library — via OpenAlex venue filter (no auth required)."""

    source_id = "acm"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        # Use OpenAlex with host_venue filter for ACM
        url = (
            f"https://api.openalex.org/works"
            f"?search={quote_plus(query)}"
            f"&filter=host_venue.publisher:ACM,publication_year:>2022"
            f"&sort=publication_date:desc&per-page=8"
            f"&select=id,title,doi,publication_date,open_access,primary_location"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        sources: list[ScoutSource] = []
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
                source_type="acm",
                verified=False,
                source_tier=_get_source_tier(link),
            ))
        return sources


# ===========================================================================
# 12. BASECrawler
# ===========================================================================

class BASECrawler(CrawlerBase):
    """BASE (Bielefeld Academic Search Engine) — 350M+ cross-repo documents."""

    source_id = "base"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            f"https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
            f"?func=PerformSearch&query={quote_plus(query)}"
            f"&hits=8&offset=0&sortfield=dcyear&sortorder=1&format=json"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        docs = data.get("response", {}).get("docs", [])
        sources: list[ScoutSource] = []
        
        for doc in docs:
            links = doc.get("dclink") or doc.get("dcidentifier") or []
            if isinstance(links, str):
                links = [links]
            link = next((l for l in links if l.startswith("http")), "")
            if not link:
                continue
            title_raw = doc.get("dctitle") or doc.get("dcsubject") or ""
            title = (title_raw[0] if isinstance(title_raw, list) else title_raw).strip()
            date_raw = doc.get("dcyear") or doc.get("dcdate") or ""
            if isinstance(date_raw, list):
                date_raw = date_raw[0] if date_raw else ""
            sources.append(ScoutSource(
                url=link,
                title=title,
                published_at=_to_iso(str(date_raw)),
                source_type="base",
                verified=False,
                source_tier=_get_source_tier(link),
            ))
        return sources


# ===========================================================================
# 13. ChemRxivCrawler
# ===========================================================================

class ChemRxivCrawler(CrawlerBase):
    """ChemRxiv REST API — chemistry, materials science preprints."""

    source_id = "chemrxiv"
    default_poll_interval = 600

    async def crawl(self, query: str) -> list[ScoutSource]:
        url = (
            f"https://chemrxiv.org/engage/api-gateway/chemrxiv/v1/items"
            f"?searchTerm={quote_plus(query)}&limit=8&skip=0&sort=published_date_desc"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            return []
            
        try:
            data = resp.json()
        except ValueError:
            return []
            
        items = data.get("itemHits", [])
        sources: list[ScoutSource] = []
        
        for hit in items:
            item = hit.get("item", hit)
            doi = item.get("doi", "")
            link = f"https://doi.org/{doi}" if doi else item.get("url", "")
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=(item.get("title") or "").strip(),
                published_at=_to_iso(item.get("statusDate", item.get("publishedDate", ""))),
                source_type="chemrxiv",
                verified=False,
                source_tier="A",
                snippet=(item.get("abstract") or "")[:500],
            ))
        return sources


# ===========================================================================
# 14. EngrXivCrawler
# ===========================================================================

class EngrXivCrawler(CrawlerBase):
    """engrXiv via OSF SHARE API — open engineering preprints."""

    source_id = "engrxiv"
    default_poll_interval = 900

    async def crawl(self, query: str) -> list[ScoutSource]:
        # OSF SHARE search, filtered to engrXiv provider
        url = (
            f"https://share.osf.io/api/v2/creativeworks/"
            f"?q={quote_plus(query)}&filter[types]=preprint"
            f"&filter[sources.name]=engrXiv&sort=-date_created&page[size]=8"
        )
        resp = await self._fetch(url, use_cache_headers=False)
        if resp.status_code != 200:
            # Fallback: OSF preprint search without provider filter
            url2 = (
                f"https://api.osf.io/v2/preprints/"
                f"?filter[provider]=engrxiv&filter[date_created][gte]={_days_ago(180)}"
                f"&page[size]=8&sort=-date_created"
            )
            resp = await self._fetch(url2, use_cache_headers=False)
            if resp.status_code != 200:
                return []

        try:
            data = resp.json()
        except ValueError:
            return []
            
        items = data.get("data", [])
        sources: list[ScoutSource] = []
        ql = query.lower().split()
        
        for item in items:
            attrs = item.get("attributes", {})
            title = attrs.get("title", "").strip()
            if not title:
                title = (attrs.get("description") or "")[:100]
            combined = (title + " " + (attrs.get("description") or "")).lower()
            if ql and not any(kw in combined for kw in ql):
                continue
            link = attrs.get("preprint_doi_created") or ""
            if not link:
                osf_id = item.get("id", "")
                link = f"https://osf.io/{osf_id}/" if osf_id else ""
            if not link:
                continue
            sources.append(ScoutSource(
                url=link,
                title=title,
                published_at=_to_iso(attrs.get("date_created", "")),
                source_type="engrxiv",
                verified=False,
                source_tier="A",
            ))
        return sources


# ===========================================================================
# Registry + Public Entry-Point (Enterprise Edition)
# ===========================================================================

_CRAWLERS: dict[str, CrawlerBase] = {
    c.source_id: c
    for c in [
        ArxivCrawler(),
        BioRxivCrawler(),
        MedRxivCrawler(),
        SemanticScholarCrawler(),
        PubMedCrawler(),
        EuroPMCCrawler(),
        CORECrawler(),
        SSRNCrawler(),
        PhilPapersCrawler(),
        IEEEXploreCrawler(),
        ACMCrawler(),
        BASECrawler(),
        ChemRxivCrawler(),
        EngrXivCrawler(),
    ]
}


async def _run_single_crawler_safe(crawler: CrawlerBase, query: str, semaphore: asyncio.Semaphore) -> List[ScoutSource]:
    """Wraper wykonujący zapytanie chronione Semaforem."""
    async with semaphore:
        try:
            return await crawler.safe_crawl(query)
        except Exception as exc:
            # W środowisku PRO przechwytujemy krytyczne usterki klasy by nie ubić całej warstwy
            logger.error(f"[Layer A] Fatal exception in crawler {crawler.source_id}: {exc}", exc_info=True)
            return []


async def run_layer_a(
    query: str,
    enabled: Optional[list[str]] = None,
) -> list[ScoutSource]:
    """
    Run all (or a subset of) Layer A crawlers in parallel.
    Zabezpieczone przez Bounded Concurrency (Semaphore) i Global Timeout.
    
    Args:
        query:   search terms / topic string
        enabled: explicit list of source_ids to run; None = all
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

    # Limitujemy współbieżność, chroniąc system przed uderzeniem np. 50 crawlerów na raz
    concurrency_limit = getattr(settings, "crawler_concurrency_limit", 10)
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [_run_single_crawler_safe(c, query, semaphore) for c in crawlers]

    # Globalny limit czasu dla całej warstwy (zapobiega zwieszeniu się Scouta w nieskończoność)
    layer_timeout = getattr(settings, "layer_a_timeout_seconds", 60.0)
    
    try:
        # wait_for dba, by niezależnie od zachowania API zewn. proces się nie zablokował
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=layer_timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"[Layer A] Przekroczono globalny limit czasu ({layer_timeout}s). Niektóre wyniki mogły zostać utracone.")
        return []

    seen_urls: set[str] = set()
    sources: list[ScoutSource] = []
    
    # Dekompozycja wyników i szczegółowa identyfikacja błędów
    for idx, batch in enumerate(results):
        if isinstance(batch, Exception):
            logger.error(f"[Layer A] Zgromadzono wyjątek z crawlera {crawlers[idx].source_id}: {batch}")
            continue
        if not isinstance(batch, list):
            continue
            
        for src in batch:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                sources.append(src)
                
    return sources


def get_crawler_status() -> list[dict]:
    """Return status dict for each crawler (used by GET /api/scout/sources)."""
    return [
        {
            "source_id": c.source_id,
            "poll_interval": c.poll_interval,
            "is_paused": c.is_paused,
            "last_seen_id": c.last_seen_id,
            "consecutive_errors": c._consecutive_errors,
        }
        for c in _CRAWLERS.values()
    ]
