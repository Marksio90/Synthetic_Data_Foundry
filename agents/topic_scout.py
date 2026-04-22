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
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx
import openai

from config.settings import settings
from agents.scout_contract import topic_priority_score

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
    dataset_category: str = "general"         # legal_regulatory | climate_finance | ...
    dataset_purpose: str = "qa_reasoning"     # compliance_qa | policy_tracking | ...
    demand_score: float = 0.0                 # estimated practical demand [0..1]
    uniqueness_score: float = 0.0             # novelty/rarity [0..1]
    quality_score: float = 0.0                # evidence quality for dataset usefulness [0..1]
    quality_gate_passed: bool = False         # hard gate per category
    quality_gate_reasons: list[str] = field(default_factory=list)


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

# ---------------------------------------------------------------------------
# Candidate domain pool — 120+ topics spanning LLM knowledge-gap blind spots.
# Domains are drawn randomly each run; DB history prevents repetition.
# ---------------------------------------------------------------------------

_CANDIDATE_DOMAINS: list[str] = [
    # ── EU Digital & AI Regulation ───────────────────────────────────────────
    "EU AI Act high-risk systems conformity assessment requirements",
    "AI liability directive EU strict liability algorithmic harm",
    "Digital Markets Act DMA gatekeeper enforcement Big Tech 2025",
    "Digital Services Act DSA very-large-platform content moderation audit",
    "Cyber Resilience Act CRA product security CE marking",
    "NIS2 directive cybersecurity critical infrastructure national implementation",
    "European Health Data Space EHDS secondary use personal health data",
    "Data Governance Act DGA data intermediary altruism certification",
    "EU Data Act cloud switching portability obligations",
    "eIDAS 2.0 European digital identity wallet implementation",
    "AI Act general purpose AI GPAI systemic risk model evaluation",
    "Ecodesign regulation software digital products energy efficiency",
    "EU Chips Act semiconductor first-in-class fabrication subsidies",
    "biometric surveillance prohibition real-time remote AI enforcement",
    "open source AI model regulation liability exemptions EU",
    "foundation model transparency GPAI code of practice obligations",
    "quantum computing export controls dual-use regulation",
    "AI Act prohibited practices social scoring emotion recognition ban",
    "digital product passport regulation circular economy implementation",
    "platform-to-business regulation P2B ranking transparency enforcement",
    # ── EU ESG & Sustainability Reporting ────────────────────────────────────
    "CSRD corporate sustainability reporting ESRS sector-specific standards",
    "SFDR sustainable finance disclosure RTS level 2 PAI indicators",
    "EU taxonomy green finance technical screening nuclear gas controversy",
    "corporate sustainability due diligence CSDDD civil liability chain",
    "ISSB IFRS S1 S2 sustainability accounting national adoption schedule",
    "TNFD biodiversity nature-related financial disclosure taskforce",
    "double materiality assessment VSME SME proportionality reporting",
    "ESG rating providers regulation EU methodology transparency conflicts",
    "greenwashing enforcement ESG claims substantiation EU case law",
    "transition finance credibility criteria financial institutions alignment",
    "gender pay gap reporting directive EU equal pay audit enforcement",
    "supply chain ESG due diligence forced labor import ban",
    "scope 3 category 15 financed emissions PCAF methodology banks",
    "carbon credits voluntary market integrity VCMI ICVCM standards",
    "nature positive targets Kunming-Montreal post-2020 framework corporate",
    "corporate biodiversity footprint methodology TNFD disclosure",
    # ── EU Climate & Energy ──────────────────────────────────────────────────
    "carbon border adjustment mechanism CBAM implementation verification 2026",
    "EU Emissions Trading System ETS aviation maritime reform 2024",
    "net zero transition planning financial institutions scenario analysis",
    "carbon removal certification framework EU methodology standards",
    "nature restoration law EU implementation biodiversity 2030 targets",
    "EU hydrogen strategy renewable energy directive RED III delegated acts",
    "building renovation wave EPBD nearly zero energy performance",
    "energy efficiency directive recast EED 2024 national implementation",
    "methane regulation oil gas EU monitoring enforcement 2025",
    "sustainable aviation fuel SAF mandate RefuelEU aviation blending",
    "FuelEU Maritime green shipping GHG intensity regulation",
    "critical raw materials act strategic stockpiling EU autonomy",
    "land use LULUCF regulation forest carbon sink accounting",
    "soil health monitoring law EU restoration degraded land",
    "plastic tax extended producer responsibility EU packaging waste",
    "water stewardship corporate disclosure EU Water Framework Directive",
    "deforestation-free supply chain regulation EUDR palm soy beef timber",
    "net metering prosumer self-consumption regulation EU member states",
    "just transition fund implementation coal regions reskilling workers",
    "stranded assets climate scenario analysis bank stress test ECB",
    # ── EU Financial & Capital Markets ──────────────────────────────────────
    "Basel IV capital requirements output floor implementation 2025",
    "DORA ICT third-party risk concentration cloud financial services",
    "MiCA crypto asset markets regulation stablecoin e-money token",
    "DeFi decentralized finance regulatory framework ESMA approach",
    "CBDC central bank digital currency design privacy offline use",
    "anti-money laundering AMLA new EU authority 2025 Frankfurt",
    "FIDA financial data access framework open finance EU",
    "ELTIF 2.0 long-term investment funds retail access liquidity",
    "PRIIPS KID revision retail investor protection disclosure reform",
    "MiFID III retail investment strategy inducements ban reform",
    "CCP central counterparty EMIR 3.0 active accounts EU clearing",
    "tokenization securities digital bond DLT pilot regime EU",
    "insurance recovery resolution regulation IRRD solvency",
    "crowdfunding regulation EU ECSPR cross-border scaling limits",
    # ── US Regulatory ────────────────────────────────────────────────────────
    "SEC climate disclosure rules Scope 3 litigation injunction 2025",
    "US AI executive order NIST AI RMF federal agency implementation",
    "FTC AI algorithmic fairness deceptive design enforcement action",
    "US critical infrastructure cybersecurity CISA binding directive",
    "EPA methane waste emissions rule oil gas 2024 enforcement",
    "US IRA Inflation Reduction Act clean energy tax credit transferability",
    "CFTC digital asset derivatives prediction market DeFi regulation",
    "SEC cybersecurity material incident 4-day disclosure rule enforcement",
    "FDA AI medical device continuous learning SaMD regulation",
    "US antitrust AI competition enforcement DOJ FTC consent decree",
    "US forced labor Uyghur Prevention Act UFLPA supply chain enforcement",
    "CFPB open banking rule 1033 implementation fintech",
    # ── International & Cross-Border ─────────────────────────────────────────
    "OECD Pillar Two global minimum tax 15 percent GloBE implementation",
    "UK sustainability disclosure requirements SDR labelling 2025",
    "Australia mandatory climate disclosure ASRS AASB implementation",
    "Singapore TCFD mandatory climate reporting MAS transition plan",
    "Japan climate disclosure SSBJ sustainability standards adoption",
    "India ESG BRSR core sustainability reporting SEBI listed companies",
    "South Korea ESG disclosure KOSPI mandatory timeline phased",
    "China ESG disclosure CSRC Shanghai Shenzhen standards GRI alignment",
    "Brazil sustainable taxonomy green finance BCB implementation",
    "OECD responsible business conduct guidelines 2024 update MNE",
    "FATF crypto virtual asset travel rule VASP cross-border",
    "anti-coercion instrument EU trade retaliation implementation",
    "WTO fisheries subsidies agreement implementation enforcement",
    "Basel III emerging market bank implementation challenges",
    # ── Health, Pharma & Biotech ─────────────────────────────────────────────
    "EU pharmaceutical legislation reform 2024 critical medicines shortage",
    "medical device regulation MDR conformity assessment backlog notified body",
    "in vitro diagnostic regulation IVDR performance evaluation transition",
    "clinical trials regulation CTR EU portal CTIS implementation",
    "advanced therapy medicinal products ATMP regulatory pathway EMA",
    "health technology assessment HTA EU joint clinical assessment",
    "orphan drug designation reform EU incentives critical revision",
    "antimicrobial resistance AMR action plan EU one health approach",
    "AI medical diagnosis regulatory pathway FDA EMA SaMD approval",
    "pandemic preparedness health emergency HERA strategic reserve EU",
    "synthetic biology biotech dual-use regulation containment EU",
    # ── Labor, Social & Governance ───────────────────────────────────────────
    "platform workers directive EU algorithmic management employment status",
    "pay transparency directive EU gender pay gap reporting audit",
    "whistleblower protection directive EU national transposition enforcement",
    "AI workplace surveillance monitoring workers rights regulation EU",
    "skills agenda digital reskilling EU funding programmes upskilling",
    "European works council directive reform information consultation rights",
    "remote work telework right regulation EU cross-border bilateral",
    "corporate governance diversity board composition mandatory quota EU",
    # ── Food, Agriculture & Environment ──────────────────────────────────────
    "sustainable use of pesticides regulation SUR reduction target EU",
    "new genomic techniques NGT precision breeding regulation EU approval",
    "novel foods cultured meat cell-based regulation approval EU",
    "CAP strategic plans 2023-2027 eco-scheme agri-environment implementation",
    "animal welfare transport slaughter regulation EU reform initiative",
    "food labelling nutriscore front-of-pack EU harmonisation",
    "fisheries EU CFP landing obligation discard ban enforcement",
    # ── Transport & Infrastructure ────────────────────────────────────────────
    "alternative fuels infrastructure AFIR EV charging deployment mandate",
    "road transport CO2 standards trucks vans zero emission 2030",
    "autonomous vehicles type approval EU uncrewed certification framework",
    "TEN-T trans-European transport network revised regulation corridors",
    "urban access regulation low emission zone harmonisation EU cities",
    "drone regulation EASA urban air mobility corridor U-space",
    # ── Tax & Trade ──────────────────────────────────────────────────────────
    "Pillar Two GloBE income inclusion qualified domestic minimum top-up",
    "digital services tax OECD Pillar One Amount A reallocation",
    "transfer pricing OECD BEPS 2.0 Amount B simplified approach",
    "carbon leakage trade adjustment mechanism WTO compatibility",
    "export controls semiconductor equipment restrictions enforcement 2024",
    "EU customs union reform single window electronic declaration",
    # ── AI, Data & Privacy ───────────────────────────────────────────────────
    "AI training data copyright web scraping regulation EU US",
    "GDPR AI profiling automated decision-making enforcement fines 2025",
    "federated learning differential privacy compliance GDPR",
    "cross-border data transfer adequacy decision EU US DPF",
    "children online protection age verification regulation EU",
    "cloud computing concentration risk regulatory guidance FSB",
    "Internet of Things IoT security labelling regulation EU RED",
]

# ---------------------------------------------------------------------------
# Query angle modifiers — appended to crawler queries to vary search angle
# each run, ensuring crawlers surface different aspects of the same domain.
# ---------------------------------------------------------------------------

_QUERY_ANGLES: list[str] = [
    "",                                          # baseline — no modifier
    "2025 enforcement case law ruling",
    "implementation guidance technical standard",
    "compliance deadline SME impact proportionality",
    "recent amendment update revision delegated act",
    "third-country equivalence extraterritorial scope",
    "litigation appeal court judgment annulment",
    "industry sector-specific guidance FAQ",
    "academic empirical study evidence research",
    "NGO civil society position paper critique",
    "consultation stakeholder response feedback",
    "cross-border harmonisation conflict challenge",
    "penalty sanction enforcement action fine",
    "transition period exemption derogation scope",
    "reporting template disclosure format standard",
    "cost-benefit impact assessment analysis",
    "SME simplification proportionality threshold",
    "supply chain upstream downstream obligation",
    "financial materiality double materiality assessment",
    "benchmark index methodology update",
]


def _select_query_angles(count: int) -> list[str]:
    """Select mostly unique query angles to improve topical diversity per run."""
    if count <= 0:
        return []

    # Prefer non-baseline angles so searches explore specific implementation facets.
    non_empty = [a for a in _QUERY_ANGLES if a]
    baseline = [a for a in _QUERY_ANGLES if not a]

    # Up to 1 baseline per run; rest sampled without replacement for uniqueness.
    baseline_slots = 1 if baseline and count >= 4 else 0
    non_empty_needed = max(0, count - baseline_slots)

    if non_empty_needed <= len(non_empty):
        chosen = random.sample(non_empty, non_empty_needed)
    else:
        # If we ever request more than available unique angles, recycle with shuffle.
        chosen = list(non_empty)
        while len(chosen) < non_empty_needed:
            chosen.extend(random.sample(non_empty, min(len(non_empty), non_empty_needed - len(chosen))))

    if baseline_slots:
        chosen.extend(random.sample(baseline, baseline_slots))

    random.shuffle(chosen)
    return chosen[:count]


async def _select_domains(n: int = 6) -> list[str]:
    """Pick domains with the highest LLM knowledge gaps using DB-backed history.

    Draws from a 120+ candidate pool, excludes domains the user has already
    ingested (permanent) and domains scanned in recent runs (rolling window),
    then asks the LLM to rank the remainder for gap potential.
    """
    from agents.scout_history import (
        get_excluded_domains,
        get_recent_domains,
        record_selected_domains,
    )

    excluded = await get_excluded_domains()
    recent = set(await get_recent_domains(window=40))  # ~6-7 runs of 6 domains

    # Filter: not permanently excluded AND not scanned recently
    available = [
        d for d in _CANDIDATE_DOMAINS
        if d not in excluded and d not in recent
    ]
    if len(available) < n:
        # Rotation complete — keep permanent exclusions but reset recency memory
        available = [d for d in _CANDIDATE_DOMAINS if d not in excluded]
    if len(available) < n:
        # All permanently excluded (extreme edge case) — use full pool
        available = list(_CANDIDATE_DOMAINS)

    # Present a random subset so LLM ordering bias never fixes the result
    subset_size = min(len(available), n + 10)
    candidates = random.sample(available, subset_size)

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    run_id = uuid.uuid4().hex[:8]

    try:
        resp = await _OPENAI.chat.completions.create(
            model=settings.openai_primary_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Today is {today}. You are a research analyst identifying where large "
                        "language models have the deepest knowledge gaps — topics with significant "
                        "recent developments that post-date LLM training cutoffs. "
                        f"Select exactly {n} domains from the list below that maximise "
                        "knowledge-gap potential. Prioritise:\n"
                        "  • Rapid 2024-2025 regulatory or technical changes not yet in training data\n"
                        "  • Niche, less-mainstream topics LLMs are unlikely to know deeply\n"
                        "  • Topics with real enforcement actions, case law, or delegated acts\n"
                        "  • Topics from diverse regulatory areas (not all ESG, not all EU)\n"
                        "Avoid picking the same obvious top-tier topics — choose for variety, depth, "
                        "and genuine novelty.\n"
                        "Return ONLY a JSON array of strings copied exactly from the list, "
                        "no other text.\n"
                        "Domains:\n" + "\n".join(f"- {d}" for d in candidates)
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Select {n} domains with the highest LLM knowledge gaps as of {today}. "
                        "Favour niche, less-obvious choices with recent enforcement or technical updates."
                    ),
                },
            ],
            temperature=0.95,
            max_tokens=500,
        )
        raw = _strip_json_fences(resp.choices[0].message.content.strip())
        selected = json.loads(raw)
        if isinstance(selected, list) and selected:
            result = [str(d) for d in selected[:n]]
            await record_selected_domains(result, run_id=run_id)
            return result
    except Exception as exc:
        logger.warning("Domain auto-selection failed: %s — using random fallback", exc)

    # Fallback: pure random from available candidates — never the same fixed head
    fallback = random.sample(available, min(n, len(available)))
    await record_selected_domains(fallback, run_id=run_id)
    return fallback


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


_CATEGORY_RULES: list[tuple[str, tuple[str, ...], str]] = [
    ("legal_regulatory", ("regulation", "directive", "act", "compliance", "liability", "court"), "compliance_qa"),
    ("climate_finance", ("taxonomy", "sfdr", "csrd", "carbon", "cbam", "esg", "emissions"), "disclosure_reporting"),
    ("health_medtech", ("fda", "medical device", "ivdr", "mdr", "clinical", "ema"), "safety_validation"),
    ("ai_data_governance", ("ai act", "gdpr", "privacy", "model", "data transfer", "copyright"), "risk_controls"),
    ("capital_markets", ("mifid", "mica", "basel", "aml", "tokenization", "derivatives"), "market_surveillance"),
]

_LEGISLATIVE_SOURCES = frozenset({
    "eurlex", "curia", "federalregister", "secedgar", "esma", "eba", "oecd", "wto", "wipo", "epo",
})

_ECON_DATA_SOURCES = frozenset({
    "imf", "worldbank", "ecb", "bis", "eurostat", "owid", "irena", "undl", "hdx",
})

_RESEARCH_SOURCES = frozenset({
    "arxiv", "openalex", "semanticscholar", "pubmed", "core", "ieee", "ssrn",
    "europepmc", "philpapers", "acm", "base", "chemrxiv", "engrxiv",
})


def _infer_dataset_profile(domain: str) -> tuple[str, str]:
    d = domain.lower()
    for category, markers, purpose in _CATEGORY_RULES:
        if any(m in d for m in markers):
            return category, purpose
    return "general", "qa_reasoning"


def _evaluate_quality_gate(
    category: str,
    sources: list[ScoutSource],
    quality_score: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    src_types = {s.source_type for s in sources}
    high_tier = sum(1 for s in sources if s.source_tier in {"S", "A"})
    n_sources = len(sources)

    if quality_score < 0.45:
        reasons.append("quality_score<0.45")

    if category == "legal_regulatory":
        if n_sources < 3:
            reasons.append("min_3_sources")
        if high_tier < 2:
            reasons.append("min_2_high_tier_sources")
        if not (src_types & _LEGISLATIVE_SOURCES):
            reasons.append("min_1_legislative_source")
    elif category == "climate_finance":
        if n_sources < 4:
            reasons.append("min_4_sources")
        if high_tier < 2:
            reasons.append("min_2_high_tier_sources")
        if not (src_types & (_LEGISLATIVE_SOURCES | _ECON_DATA_SOURCES)):
            reasons.append("min_1_policy_or_econ_source")
    elif category == "health_medtech":
        if n_sources < 4:
            reasons.append("min_4_sources")
        if high_tier < 2:
            reasons.append("min_2_high_tier_sources")
        if not (src_types & _RESEARCH_SOURCES):
            reasons.append("min_1_research_source")
    elif category == "ai_data_governance":
        if n_sources < 3:
            reasons.append("min_3_sources")
        if high_tier < 2:
            reasons.append("min_2_high_tier_sources")
        if not (src_types & (_LEGISLATIVE_SOURCES | _RESEARCH_SOURCES)):
            reasons.append("min_1_regulatory_or_research_source")
    elif category == "capital_markets":
        if n_sources < 3:
            reasons.append("min_3_sources")
        if high_tier < 2:
            reasons.append("min_2_high_tier_sources")
        if not (src_types & (_ECON_DATA_SOURCES | _LEGISLATIVE_SOURCES)):
            reasons.append("min_1_market_regulatory_source")
    else:
        if n_sources < 3:
            reasons.append("min_3_sources")
        if high_tier < 1:
            reasons.append("min_1_high_tier_source")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Per-domain processing (runs in parallel)
# ---------------------------------------------------------------------------


async def _process_domain(
    domain: str,
    progress_callback: Optional[ProgressCallback] = None,
    topic_callback: Optional[TopicCallback] = None,
    query_suffix: str = "",
) -> Optional[ScoutTopicData]:
    """Crawl all sources for one domain, verify URLs, score, and return a topic.

    `domain` is the canonical identifier used for topic title/ID and history.
    `query_suffix` is an angle modifier appended to crawler queries only, so
    consecutive runs of the same domain surface different aspects of the topic.
    """
    crawler_query = f"{domain} {query_suffix}".strip() if query_suffix else domain

    async def _log(msg: str) -> None:
        logger.info("[Scout] %s", msg)
        if progress_callback:
            try:
                await progress_callback(msg)
            except Exception:
                pass

    angle_note = f" [{query_suffix}]" if query_suffix else ""
    await _log(f"Scanning: {domain[:55]}{angle_note}")

    # Lazy imports avoid circular dependency (layers import ScoutSource from this module)
    from agents.crawlers.layer_a import run_layer_a
    from agents.crawlers.layer_b import run_layer_b
    from agents.crawlers.layer_c import run_layer_c
    from agents.crawlers.layer_d import run_layer_d
    from agents.crawlers.layer_e import run_layer_e

    gathered = await asyncio.gather(
        run_layer_a(crawler_query),    # Layer A: 14 science/research crawlers
        run_layer_b(crawler_query),    # Layer B: 10 legislation/regulatory crawlers
        run_layer_c(crawler_query),    # Layer C: 10 finance/economic data crawlers
        run_layer_d(crawler_query),    # Layer D:  6 tech/social signal crawlers
        run_layer_e(crawler_query),    # Layer E:  6 multimedia/archive crawlers
        _fetch_openalex(crawler_query),  # cross-layer aggregator (kept for coverage)
        _probe_llm_uncertainty(domain),  # uncertainty probe on canonical domain name
        return_exceptions=True,
    )

    layer_a_srcs = gathered[0] if not isinstance(gathered[0], Exception) else []
    layer_b_srcs = gathered[1] if not isinstance(gathered[1], Exception) else []
    layer_c_srcs = gathered[2] if not isinstance(gathered[2], Exception) else []
    layer_d_srcs = gathered[3] if not isinstance(gathered[3], Exception) else []
    layer_e_srcs = gathered[4] if not isinstance(gathered[4], Exception) else []
    oa_srcs      = gathered[5] if not isinstance(gathered[5], Exception) else []
    uncertainty  = gathered[6] if not isinstance(gathered[6], Exception) else 0.5

    # Merge: A (science) → B (legislation) → C (finance) → D (social) → E (multimedia) → OA
    all_sources: list[ScoutSource] = [
        *layer_a_srcs, *layer_b_srcs, *layer_c_srcs,
        *layer_d_srcs, *layer_e_srcs, *oa_srcs,
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
    _SOCIAL_TYPES = frozenset({"hackernews", "reddit", "mastodon", "producthunt"})
    social_count = sum(1 for s in verified if s.source_type in _SOCIAL_TYPES)
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
    source_type_diversity = min(1.0, len({s.source_type for s in verified}) / 6.0)
    high_tier_ratio = sum(1 for s in verified if s.source_tier in {"S", "A"}) / max(1, len(verified))
    cutoff_coverage = min(1.0, len(post_cutoff_models) / 8.0)

    score = round(
        0.40 * recency
        + 0.30 * float(uncertainty)
        + 0.20 * density
        + 0.10 * social,
        3,
    )
    # Backward-compatible uplift: slightly reward stronger evidence quality and source spread.
    score = round(min(1.0, score + 0.04 * source_type_diversity + 0.04 * high_tier_ratio), 3)

    # New quality-and-demand profile used for stronger dataset triage.
    evidence_quality = min(1.0, 0.45 * high_tier_ratio + 0.35 * source_type_diversity + 0.20 * density)
    uniqueness_score = min(1.0, 0.50 * scoring.knowledge_gap_score + 0.30 * scoring.llm_uncertainty + 0.20 * cutoff_coverage)
    demand_score = min(1.0, 0.55 * recency + 0.30 * cutoff_coverage + 0.15 * evidence_quality)
    quality_score = min(1.0, 0.60 * evidence_quality + 0.40 * uniqueness_score)
    dataset_category, dataset_purpose = _infer_dataset_profile(domain)
    quality_gate_passed, quality_gate_reasons = _evaluate_quality_gate(
        dataset_category, verified, quality_score,
    )

    await _log(
        f"  {domain[:40]}: gap={scoring.knowledge_gap_score:.3f} score={score:.2f} "
        f"tier={best_tier} sources={len(verified)} "
        f"uncertainty={scoring.llm_uncertainty:.2f} cutoff_models={len(post_cutoff_models)} "
        f"quality={quality_score:.2f} uniqueness={uniqueness_score:.2f} gate={'pass' if quality_gate_passed else 'fail'}"
    )
    if not quality_gate_passed:
        await _log(f"  {domain[:40]}: quality gate failed ({', '.join(quality_gate_reasons[:4])})")

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
        ingest_ready=quality_gate_passed,
        dataset_category=dataset_category,
        dataset_purpose=dataset_purpose,
        demand_score=round(demand_score, 3),
        uniqueness_score=round(uniqueness_score, 3),
        quality_score=round(quality_score, 3),
        quality_gate_passed=quality_gate_passed,
        quality_gate_reasons=quality_gate_reasons,
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

    # Assign mostly unique angle modifiers so each domain explores a distinct facet.
    angles = _select_query_angles(len(domains))
    angle_preview = [a if a else "(baseline)" for a in angles[:4]]
    await _log(f"Query angles: {angle_preview}...")

    # Process all domains concurrently (each internally already fans out crawlers in parallel)
    domain_results = await asyncio.gather(
        *[
            _process_domain(domain, progress_callback, topic_callback, query_suffix=angle)
            for domain, angle in zip(domains, angles)
        ],
        return_exceptions=True,
    )

    results = [r for r in domain_results if isinstance(r, ScoutTopicData)]
    passed_count = sum(1 for t in results if t.quality_gate_passed)
    if results:
        await _log(f"Quality gate passed: {passed_count}/{len(results)} topics")
    # Prioritize topics with high practical usefulness for dataset generation.
    results.sort(
        key=lambda t: (
            0.20 * (1.0 if t.quality_gate_passed else 0.0)
            + 0.45 * t.quality_score
            + 0.25 * t.uniqueness_score
            + 0.20 * t.knowledge_gap_score
            + 0.10 * t.demand_score
        ),
        reverse=True,
    )
    await _log(f"Scout complete — {len(results)} topics discovered.")
    return results[:max_topics]
