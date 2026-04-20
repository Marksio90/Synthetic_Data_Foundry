"""
agents/calibrator.py — Automatyczna kalibracja parametrów pipeline'u.

Analizuje charakterystykę chunków dokumentu i dobiera optymalne parametry
BEZ wykonywania żadnych wywołań LLM (zero kosztu API, działa w milisekundach).

Metryki:
  - Średnia i mediana długości chunków (chars)
  - Bogactwo słownictwa (unique_words / total_words)
  - Gęstość informacji (artykuły/ustępy, liczby, daty, akronimy)
  - Liczba chunków z nagłówkami sekcji

Na tej podstawie dobierane są:
  quality_threshold   — próg akceptacji odpowiedzi przez sędziego
  max_turns           — maksymalna liczba tur rozmowy per chunk
  adversarial_ratio   — odsetek pytań adversarialnych

Wynik kalibracji jest logowany i przekazywany do orchestratora,
który nadpisuje ustawienia z .env.
"""

from __future__ import annotations

import logging
import re
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for information-density scoring
# ---------------------------------------------------------------------------
_ARTICLE_RE = re.compile(
    r"\b(art(?:ykuł)?\.?\s*\d+|article\s+\d+|§\s*\d+|ustęp|ust\.|paragraph|section)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:[.,]\d+)*\s*(?:%|EUR|PLN|mln|mld|tys\.)?\b")
_DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")


def _vocabulary_richness(text: str) -> float:
    """Type-token ratio (unique/total words) — proxy for domain specificity."""
    words = re.findall(r"\b[a-zA-ZąćęłńóśźżÄÖÜäöüàâéèêëîïôœùûü]{3,}\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _information_density(text: str) -> float:
    """
    Score 0.0–1.0 how information-dense the text is.
    Legal/regulatory documents score high; boilerplate/TOC scores low.
    """
    if not text:
        return 0.0
    chars = max(len(text), 1)
    article_hits = len(_ARTICLE_RE.findall(text))
    number_hits = len(_NUMBER_RE.findall(text))
    date_hits = len(_DATE_RE.findall(text))
    acronym_hits = len(_ACRONYM_RE.findall(text))
    # Normalise by text length
    raw = (article_hits * 3 + number_hits * 1 + date_hits * 2 + acronym_hits * 1) / (chars / 100)
    return min(raw / 10.0, 1.0)  # cap at 1.0


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    quality_threshold: float
    max_turns: int
    adversarial_ratio: float
    reasoning: list[str] = field(default_factory=list)

    # Diagnostic stats
    n_chunks: int = 0
    avg_chunk_len: float = 0.0
    vocab_richness: float = 0.0
    info_density: float = 0.0
    heading_ratio: float = 0.0

    def as_env_overrides(self) -> dict[str, str]:
        """Return dict suitable for subprocess env injection."""
        return {
            "QUALITY_THRESHOLD": str(self.quality_threshold),
            "MAX_TURNS": str(self.max_turns),
            "ADVERSARIAL_RATIO": str(self.adversarial_ratio),
        }

    def summary(self) -> str:
        lines = [
            f"quality_threshold = {self.quality_threshold} "
            f"(avg_chunk_len={self.avg_chunk_len:.0f}, vocab_richness={self.vocab_richness:.2f})",
            f"max_turns = {self.max_turns}",
            f"adversarial_ratio = {self.adversarial_ratio}",
        ]
        for r in self.reasoning:
            lines.append(f"  → {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------

def calibrate(chunks: list[dict[str, Any]]) -> CalibrationResult:
    """
    Compute optimal pipeline parameters from chunk characteristics.

    Args:
        chunks: List of chunk dicts with at least 'content' key.
                Accepts both ORM objects (with .content attribute)
                and plain dicts.

    Returns:
        CalibrationResult with auto-tuned parameters.
    """
    if not chunks:
        logger.warning("Calibrator: no chunks provided — using safe defaults")
        return CalibrationResult(
            quality_threshold=0.82,
            max_turns=3,
            adversarial_ratio=0.10,
            reasoning=["Brak chunków — zastosowano wartości domyślne"],
        )

    # Extract content strings (handle both ORM objects and plain dicts)
    contents: list[str] = []
    headings: list[str] = []
    for c in chunks:
        if hasattr(c, "content"):
            contents.append(c.content or "")
            headings.append(c.section_heading or "")
        else:
            contents.append(c.get("content", ""))
            headings.append(c.get("section_heading", "") or "")

    # ── Core metrics ─────────────────────────────────────────────────────────
    lengths = [len(t) for t in contents if t]
    avg_len = statistics.mean(lengths) if lengths else 300

    # Vocabulary richness on a combined sample (first 10k chars per chunk, max 30 chunks)
    sample_text = " ".join(c[:10_000] for c in contents[:30])
    vocab = _vocabulary_richness(sample_text)

    # Information density on a sample
    density_scores = [_information_density(c) for c in contents[:50]]
    avg_density = statistics.mean(density_scores) if density_scores else 0.5

    # Ratio of chunks that have a section heading (structural richness)
    heading_ratio = sum(1 for h in headings if h) / max(len(headings), 1)

    reasoning: list[str] = []

    # ── quality_threshold ────────────────────────────────────────────────────
    # Start at baseline 0.82
    threshold = 0.82

    if avg_len < 250:
        threshold -= 0.04
        reasoning.append(f"Krótkie chunki (avg={avg_len:.0f}) → obniżono próg o 0.04")
    elif avg_len > 900:
        threshold += 0.02
        reasoning.append(f"Długie chunki (avg={avg_len:.0f}) → podwyższono próg o 0.02")

    if vocab > 0.65:
        threshold += 0.02
        reasoning.append(f"Wysokie bogactwo słownictwa ({vocab:.2f}) → +0.02")
    elif vocab < 0.35:
        threshold -= 0.02
        reasoning.append(f"Niskie bogactwo słownictwa ({vocab:.2f}) → -0.02")

    if avg_density > 0.6:
        threshold += 0.02
        reasoning.append(f"Wysoka gęstość informacji ({avg_density:.2f}) → +0.02")
    elif avg_density < 0.2:
        threshold -= 0.03
        reasoning.append(
            f"Niska gęstość informacji ({avg_density:.2f}) → -0.03 "
            "(możliwy boilerplate lub spis treści)"
        )

    threshold = round(max(0.72, min(threshold, 0.92)), 2)

    # ── max_turns ────────────────────────────────────────────────────────────
    if avg_len < 300:
        max_turns = 2
        reasoning.append("Krótkie chunki → max_turns=2 (mniej treści do eksploracji)")
    elif avg_density > 0.55 and avg_len > 600:
        max_turns = 3
        reasoning.append("Gęste, długie chunki → max_turns=3 (wiele aspektów do omówienia)")
    else:
        max_turns = 3

    # ── adversarial_ratio ────────────────────────────────────────────────────
    # Higher density → model less likely to hallucinate → standard ratio
    # Lower density → more boilerplate → slightly reduce adversarial (fewer "traps")
    if avg_density < 0.15:
        adversarial_ratio = 0.07
        reasoning.append("Niska gęstość treści → adversarial_ratio=0.07")
    else:
        adversarial_ratio = 0.10

    result = CalibrationResult(
        quality_threshold=threshold,
        max_turns=max_turns,
        adversarial_ratio=adversarial_ratio,
        reasoning=reasoning,
        n_chunks=len(chunks),
        avg_chunk_len=round(avg_len, 1),
        vocab_richness=round(vocab, 3),
        info_density=round(avg_density, 3),
        heading_ratio=round(heading_ratio, 2),
    )

    logger.info(
        "Calibrator: %d chunks analysed → threshold=%.2f, max_turns=%d, adversarial=%.2f",
        len(chunks), threshold, max_turns, adversarial_ratio,
    )
    for line in reasoning:
        logger.debug("  %s", line)

    return result


# ---------------------------------------------------------------------------
# Adaptive calibrator — adjusts threshold in real time from judge scores
# ---------------------------------------------------------------------------

class AdaptiveCalibrator:
    """
    Tracks live judge quality scores and adjusts quality_threshold between chunks.
    Thread-safe: designed to be updated from the main thread after each chunk batch.

    Logic:
      - avg_score well above threshold (>+0.15) → raise threshold by 0.01 (stricter)
      - avg_score barely above threshold (<+0.03) → lower threshold by 0.01 (permissive)
      - Re-calibrates every `recalibrate_every` recorded scores.
    """

    def __init__(self, initial_threshold: float, window: int = 30, recalibrate_every: int = 10) -> None:
        self.threshold = initial_threshold
        self._window = window
        self._every = recalibrate_every
        self._scores: deque[float] = deque(maxlen=window)
        self._count = 0
        self._lock = threading.Lock()

    def record(self, score: float) -> None:
        with self._lock:
            self._scores.append(score)
            self._count += 1
            if self._count % self._every == 0:
                self._recalibrate()

    def _recalibrate(self) -> None:
        if len(self._scores) < self._window // 2:
            return
        avg = sum(self._scores) / len(self._scores)
        old = self.threshold
        if avg > self.threshold + 0.15:
            self.threshold = min(round(self.threshold + 0.01, 2), 0.92)
        elif avg < self.threshold + 0.03:
            self.threshold = max(round(self.threshold - 0.01, 2), 0.72)
        if self.threshold != old:
            logger.info(
                "AdaptiveCalibrator: threshold %.2f → %.2f (window_avg=%.3f, n=%d)",
                old, self.threshold, avg, len(self._scores),
            )

    @property
    def current_threshold(self) -> float:
        with self._lock:
            return self.threshold
