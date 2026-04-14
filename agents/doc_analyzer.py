"""
agents/doc_analyzer.py — Automatyczna analiza dokumentów.

Wykrywa bez żadnych LLM-ów (zero kosztu):
  1. Język źródłowy — analiza częstotliwości liter charakterystycznych
  2. Domenę tematyczną — dopasowanie słów kluczowych do profili domenowych
  3. Perspektywy — wynikają z domeny, dobierane automatycznie
  4. Potrzebę tłumaczenia — jeśli wykryty język ≠ 'pl'

Używany przez AutoPilot Orchestrator przed uruchomieniem pipeline'u.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain profiles: keywords → domain → perspectives
# ---------------------------------------------------------------------------

_DOMAIN_PROFILES: dict[str, dict] = {
    "esg_sustainability": {
        "keywords": [
            "csrd", "sfdr", "taxonomy", "taksonomia", "esg", "sustainability",
            "non-financial", "raportowanie niefinansowe", "zrównoważony",
            "csddd", "due diligence", "łańcuch wartości", "emisje",
            "climate", "klimat", "biodiversity", "różnorodność biologiczna",
            "paris agreement", "porozumienie paryskie", "scope 1", "scope 2",
        ],
        "perspectives": ["cfo", "prawnik", "audytor"],
        "label": "ESG / Zrównoważony Rozwój UE",
    },
    "banking_compliance": {
        "keywords": [
            "crd", "crr", "dora", "mica", "basel", "capital requirement",
            "wymóg kapitałowy", "credit risk", "ryzyko kredytowe",
            "liquidity", "płynność", "leverage ratio", "wskaźnik dźwigni",
            "srep", "nsfr", "lcr", "tier 1", "tier 2",
            "pillar", "filar", "bazylea", "resolution", "bail-in",
        ],
        "perspectives": ["compliance_officer", "risk_manager", "prawnik"],
        "label": "Compliance Bankowy / Fintech",
    },
    "data_privacy": {
        "keywords": [
            "gdpr", "rodo", "data protection", "ochrona danych", "personal data",
            "dane osobowe", "data subject", "osoba, której dane", "processor",
            "administrator", "controller", "consent", "zgoda", "breach",
            "naruszenie", "dpo", "ai act", "ustawa o ai", "data governance",
            "data act", "nis2", "cybersecurity",
        ],
        "perspectives": ["dpo", "prawnik_it", "cto"],
        "label": "Ochrona Danych / Cyberbezpieczeństwo",
    },
    "tax_law": {
        "keywords": [
            "cit", "vat", "pit", "podatek", "tax", "mdr", "dac6", "beps",
            "pillar ii", "global minimum tax", "globalny podatek minimalny",
            "transfer pricing", "ceny transferowe", "permanent establishment",
            "zakład", "withholding tax", "podatek u źródła", "fatca", "crs",
        ],
        "perspectives": ["doradca_podatkowy", "ksiegowy", "prawnik_podatkowy"],
        "label": "Prawo Podatkowe",
    },
    "labor_law": {
        "keywords": [
            "kodeks pracy", "labour", "employment", "zatrudnienie",
            "work time", "czas pracy", "leave", "urlop", "dismissal",
            "zwolnienie", "remuneration", "wynagrodzenie", "workplace",
            "work-life balance", "parental leave", "urlop rodzicielski",
            "posted worker", "delegowanie pracowników",
        ],
        "perspectives": ["hr_director", "prawnik_pracy", "inspektor_pip"],
        "label": "Prawo Pracy",
    },
    "public_procurement": {
        "keywords": [
            "pzp", "zamówienie publiczne", "public procurement", "przetarg",
            "tender", "zamawiający", "wykonawca", "oferta", "siwz", "swz",
            "concession", "koncesja", "utility", "sektorowy",
            "national security", "bezpieczeństwo narodowe",
        ],
        "perspectives": ["zamawiajacy", "wykonawca", "prawnik_pzp"],
        "label": "Zamówienia Publiczne",
    },
    "pharma_medical": {
        "keywords": [
            "mdr", "ivdr", "medical device", "wyroby medyczne", "gmp",
            "good manufacturing", "ema", "clinical trial", "badanie kliniczne",
            "marketing authorisation", "pozwolenie na dopuszczenie",
            "pharmacovigilance", "nadzór nad bezpieczeństwem",
            "hta", "health technology assessment",
        ],
        "perspectives": ["regulatory_affairs", "quality_manager", "audytor_gmp"],
        "label": "Farmaceutyka / Wyroby Medyczne",
    },
    "accounting_ifrs": {
        "keywords": [
            "mssf", "ifrs", "ias", "fair value", "wartość godziwa",
            "impairment", "utrata wartości", "revenue recognition",
            "przychody", "lease", "leasing", "financial instruments",
            "instrumenty finansowe", "consolidation", "konsolidacja",
            "audit", "rewizja finansowa", "going concern",
        ],
        "perspectives": ["biegly_rewident", "kontroler_finansowy", "cfo"],
        "label": "Rachunkowość / MSSF",
    },
    "construction_real_estate": {
        "keywords": [
            "prawo budowlane", "construction law", "pozwolenie na budowę",
            "building permit", "nieruchomość", "real estate", "deweloper",
            "warunki techniczne", "technical conditions", "architect",
            "urbanistyka", "mpzp", "study uwarunkowań",
        ],
        "perspectives": ["inspektor_nadzoru", "prawnik_nieruchomosci", "deweloper"],
        "label": "Prawo Budowlane / Nieruchomości",
    },
}

# Fallback when no domain matches well enough
_DEFAULT_DOMAIN = "general_legal"
_DEFAULT_PERSPECTIVES = ["cfo", "prawnik", "audytor"]
_DEFAULT_LABEL = "Prawo / Regulacje (ogólne)"

# Minimum keyword hit-rate to claim a domain
_DOMAIN_MIN_SCORE = 0.003  # 0.3% of all words


# ---------------------------------------------------------------------------
# Language detection (zero external deps — character-frequency analysis)
# ---------------------------------------------------------------------------

# Characteristic letters per language (normalised, lowercase)
_LANG_SIGNATURES: dict[str, set[str]] = {
    "pl": set("ąćęłńóśźż"),
    "de": set("äöüß"),
    "fr": set("àâæçéèêëîïôœùûüÿ"),
    "es": set("áéíóúüñ¿¡"),
    "it": set("àèéìîòóùú"),
    "pt": set("ãõàáâçéêíóôú"),
    "ru": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    "uk": set("іїєґ"),
}

# Common stopwords per language — used as secondary signal
_STOPWORDS: dict[str, list[str]] = {
    "en": ["the", "of", "and", "to", "in", "is", "that", "for", "on", "with", "shall", "may"],
    "pl": ["w", "i", "z", "na", "do", "że", "się", "nie", "o", "przez", "po", "jest", "są"],
    "de": ["der", "die", "das", "und", "in", "zu", "von", "den", "mit", "des", "ist", "ein"],
    "fr": ["le", "de", "et", "en", "les", "des", "du", "un", "une", "la", "est", "que"],
}


def detect_language(text: str) -> str:
    """
    Detect document language using character-frequency + stopword heuristics.
    Returns ISO-639-1 code: 'pl', 'en', 'de', 'fr', 'es', 'it', 'ru', or 'unknown'.
    No external libraries needed.
    """
    if not text:
        return "unknown"

    # Sample first 5000 chars — sufficient for language detection
    sample = text[:5000].lower()
    total_alpha = max(sum(1 for c in sample if c.isalpha()), 1)

    # Score each language by frequency of characteristic characters
    char_scores: dict[str, float] = {}
    for lang, chars in _LANG_SIGNATURES.items():
        hits = sum(1 for c in sample if c in chars)
        char_scores[lang] = hits / total_alpha

    best_char_lang = max(char_scores, key=lambda k: char_scores[k])
    best_char_score = char_scores[best_char_lang]

    # If strong character signal → use it
    if best_char_score > 0.01:
        logger.debug("Language detected by chars: %s (score=%.3f)", best_char_lang, best_char_score)
        return best_char_lang

    # Fallback: stopword frequency
    words = re.findall(r"\b[a-z]{2,}\b", sample)
    word_count = max(len(words), 1)
    word_set = set(words)

    stopword_scores: dict[str, float] = {}
    for lang, stops in _STOPWORDS.items():
        hits = sum(1 for w in stops if w in word_set)
        stopword_scores[lang] = hits / len(stops)

    best_sw_lang = max(stopword_scores, key=lambda k: stopword_scores[k])
    best_sw_score = stopword_scores[best_sw_lang]

    if best_sw_score > 0.3:
        logger.debug("Language detected by stopwords: %s (score=%.2f)", best_sw_lang, best_sw_score)
        return best_sw_lang

    # Cannot determine — default to English (most common in legal docs)
    return "en"


# ---------------------------------------------------------------------------
# Domain detection (keyword matching)
# ---------------------------------------------------------------------------

def detect_domain(text: str) -> tuple[str, str, list[str], float]:
    """
    Detect domain from document text using keyword frequency matching.
    Returns: (domain_key, domain_label, perspectives, confidence_score)
    """
    if not text:
        return _DEFAULT_DOMAIN, _DEFAULT_LABEL, _DEFAULT_PERSPECTIVES, 0.0

    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    total_words = max(len(words), 1)

    domain_scores: dict[str, float] = {}
    for domain_key, profile in _DOMAIN_PROFILES.items():
        hits = sum(1 for kw in profile["keywords"] if kw in text_lower)
        domain_scores[domain_key] = hits / total_words

    best_domain = max(domain_scores, key=lambda k: domain_scores[k])
    best_score = domain_scores[best_domain]

    if best_score < _DOMAIN_MIN_SCORE:
        logger.info(
            "Domain detection: no strong match (best=%.4f) → using default", best_score
        )
        return _DEFAULT_DOMAIN, _DEFAULT_LABEL, _DEFAULT_PERSPECTIVES, best_score

    profile = _DOMAIN_PROFILES[best_domain]
    logger.info(
        "Domain detected: %s — %s (score=%.4f, confidence=%.0f%%)",
        best_domain, profile["label"], best_score,
        min(best_score / _DOMAIN_MIN_SCORE * 20, 100),
    )
    # Confidence: normalised relative to minimum threshold, capped at 1.0
    confidence = min(best_score / (_DOMAIN_MIN_SCORE * 5), 1.0)
    return best_domain, profile["label"], profile["perspectives"], round(confidence, 2)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DocumentAnalysis:
    """Result of analysing one or more documents."""
    language: str                        # ISO-639-1 detected language
    translation_required: bool           # True when language != 'pl'
    domain: str                          # domain_key
    domain_label: str                    # human-readable domain label
    perspectives: list[str]             # auto-selected perspectives
    domain_confidence: float            # 0.0–1.0
    total_chars: int                    # raw text size analysed
    auto_decisions: list[str] = field(default_factory=list)  # audit trail

    def summary(self) -> str:
        lines = [
            f"Język: {self.language.upper()} {'(tłumaczenie na PL)' if self.translation_required else '(PL ✓)'}",
            f"Domena: {self.domain_label} (pewność: {self.domain_confidence:.0%})",
            f"Perspektywy: {', '.join(self.perspectives)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_documents(file_paths: list[str | Path]) -> DocumentAnalysis:
    """
    Analyse a list of PDF/text files (already in data/ directory).
    Reads the first ~20 000 chars from each file for analysis.
    Returns a DocumentAnalysis with all auto-detected parameters.
    """
    combined_text = ""
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            logger.warning("DocAnalyzer: file not found — %s", p)
            continue
        try:
            # PDFs are read as binary; we extract readable ASCII/UTF-8 text
            raw = p.read_bytes()
            # Try UTF-8 first, fallback to latin-1
            try:
                text_chunk = raw.decode("utf-8", errors="ignore")
            except Exception:
                text_chunk = raw.decode("latin-1", errors="ignore")
            combined_text += text_chunk[:20_000] + "\n"
        except Exception as exc:
            logger.warning("DocAnalyzer: could not read %s — %s", p, exc)

    if not combined_text.strip():
        logger.warning("DocAnalyzer: no readable text extracted from files")
        return DocumentAnalysis(
            language="pl",
            translation_required=False,
            domain=_DEFAULT_DOMAIN,
            domain_label=_DEFAULT_LABEL,
            perspectives=_DEFAULT_PERSPECTIVES,
            domain_confidence=0.0,
            total_chars=0,
            auto_decisions=["Brak tekstu do analizy — użyto wartości domyślnych"],
        )

    lang = detect_language(combined_text)
    domain, label, perspectives, confidence = detect_domain(combined_text)
    translation_required = lang != "pl"

    decisions = [
        f"Wykryty język: {lang.upper()} → {'tłumaczenie wymagane' if translation_required else 'brak tłumaczenia'}",
        f"Wykryta domena: {label} (pewność {confidence:.0%})",
        f"Automatycznie wybrane perspektywy: {', '.join(perspectives)}",
    ]
    if confidence < 0.3:
        decisions.append(
            "⚠ Niska pewność domeny — rozważ ręczny wybór perspektyw w trybie manualnym"
        )

    return DocumentAnalysis(
        language=lang,
        translation_required=translation_required,
        domain=domain,
        domain_label=label,
        perspectives=perspectives,
        domain_confidence=confidence,
        total_chars=len(combined_text),
        auto_decisions=decisions,
    )
