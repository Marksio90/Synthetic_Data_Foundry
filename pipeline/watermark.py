"""
pipeline/watermark.py — B2B Linguistic Watermarking (Self-Check patch).

Two complementary techniques applied every WATERMARK_INTERVAL records:

1. Double-space injection: inserts a second ASCII space character after the
   word "dyrektywa" (or its English equivalent "directive"). This is
   imperceptible to human readers but detectable programmatically.

2. Rare-synonym substitution: replaces the common word "wymogi" with the
   rarer legal synonym "wymogi normatywne" in the assistant turn text.

The unique hash is derived from:
    SHA-256(client_id || batch_id || FOUNDRY_SECRET || str(record_index))

The hash is stored in watermark_registry so Anthropic can later prove
which client received which dataset copy.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid

logger = logging.getLogger(__name__)

# Secret pepper — override via FOUNDRY_WATERMARK_SECRET env var
_SECRET = os.getenv("FOUNDRY_WATERMARK_SECRET", "esg-foundry-default-secret-change-me")

# ----------------------------------------------------------------------------
# Synonym map: common → rare legal equivalent (Polish ESG vocabulary)
# ----------------------------------------------------------------------------
_SYNONYM_MAP: dict[str, str] = {
    "wymogi": "wymogi normatywne",
    "obowiązki": "powinności prawne",
    "raportowanie": "sprawozdawczość regulacyjna",
    "ujawnienia": "ujawnienia informacyjne",
}

# Regex: word boundary matches
_DOUBLE_SPACE_PATTERN = re.compile(
    r"\b(dyrektywa|directive|rozporządzenie|regulation)\b",
    flags=re.IGNORECASE,
)


def compute_watermark_hash(
    client_id: str,
    batch_id: str,
    record_index: int,
) -> str:
    """Return a 64-char hex SHA-256 tied to this client + batch + position."""
    payload = f"{client_id}|{batch_id}|{record_index}|{_SECRET}"
    return hashlib.sha256(payload.encode()).hexdigest()


def inject_watermark(text: str, technique_index: int) -> str:
    """
    Apply one of the two watermark techniques to *text* and return the
    modified string.  technique_index selects which technique:
        0 → double-space after keyword
        1 → rare-synonym substitution
    """
    if technique_index % 2 == 0:
        # Technique 1: double-space after directive keyword
        return _DOUBLE_SPACE_PATTERN.sub(lambda m: m.group(0) + "  ", text, count=1)
    else:
        # Technique 2: rare synonym substitution (first match only)
        for common, rare in _SYNONYM_MAP.items():
            if common in text:
                return text.replace(common, rare, 1)
        # Fallback: double-space if no synonym found
        return _DOUBLE_SPACE_PATTERN.sub(lambda m: m.group(0) + "  ", text, count=1)


def build_watermark_description(technique_index: int) -> str:
    if technique_index % 2 == 0:
        return "double-space-after-directive-keyword"
    return "rare-synonym-substitution"
