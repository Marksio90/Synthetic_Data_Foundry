"""
utils/dedup.py — Band-based LSH near-duplicate question detection.

Używa MinHash (word-trigram shingles) z band LSH do O(1) amortyzowanego
wyszukiwania duplikatów — zamiast O(N²) naiwnego porównania.

Parametry LSH:
  _NUM_PERM  = 128 permutacji MinHash (~1% błędu szacowania Jaccard)
  _NUM_BANDS = 16 pasm
  _ROWS_BAND = 8 wierszy na pasmo (16 × 8 = 128)

Prawdopodobieństwo detekcji duplikatu przy Jaccard=0.85:
  P(wykryty) = 1 - (1 - 0.85^8)^16 ≈ 99.97%
Fałszywe alarmy dla Jaccard=0.5:
  P(fałszywy) = 1 - (1 - 0.5^8)^16 ≈ 6% → weryfikowane dokładnym Jaccard
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_NUM_PERM = 128   # MinHash permutations
_NUM_BANDS = 16   # LSH bands
_ROWS_BAND = 8    # rows per band (must equal _NUM_PERM // _NUM_BANDS)


class MinHashDeduplicator:
    """
    Thread-safe near-duplicate detector z band-LSH.

    Każde wywołanie is_duplicate(text):
      - Zwraca True  (znaleziono podobny — pomiń rekord)
      - Zwraca False (unikalny — dodaje do indeksu jako efekt uboczny)

    Złożoność: O(_NUM_BANDS) na zapytanie, O(N) pamięci.

    Użycie:
        dedup = MinHashDeduplicator(threshold=0.85)
        dedup.load_from_jsonl("output/dataset.jsonl")
        if dedup.is_duplicate(question):
            skip ...
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self._threshold = threshold
        self._signatures: list[list[int]] = []
        # _band_tables[b][band_hash] = lista indeksów sygnatur w _signatures
        self._band_tables: list[dict[int, list[int]]] = [
            {} for _ in range(_NUM_BANDS)
        ]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Signature computation
    # ------------------------------------------------------------------

    def _signature(self, text: str) -> list[int]:
        """Oblicz sygnaturę MinHash na bazie word-trigram shingles."""
        words = text.lower().split()
        if len(words) >= 3:
            shingles: set[str] = {
                f"{words[i]}_{words[i+1]}_{words[i+2]}"
                for i in range(len(words) - 2)
            }
        elif len(words) >= 2:
            shingles = {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}
        else:
            shingles = {text.lower()} if text.strip() else {"_empty_"}

        sig: list[int] = []
        for seed in range(_NUM_PERM):
            min_h = min(
                int(hashlib.md5(f"{seed}|{s}".encode()).hexdigest(), 16)
                for s in shingles
            )
            sig.append(min_h)
        return sig

    def _band_hashes(self, sig: list[int]) -> list[int]:
        """Oblicz jeden hash na pasmo (band) — klucze do tablicy LSH."""
        hashes: list[int] = []
        for b in range(_NUM_BANDS):
            start = b * _ROWS_BAND
            band = tuple(sig[start : start + _ROWS_BAND])
            hashes.append(hash(band))
        return hashes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, text: str) -> bool:
        """
        Zwraca True jeśli *text* jest podobny do wcześniej zarejestrowanego.
        Jeśli False — rejestruje *text* w indeksie (efekt uboczny).
        """
        sig = self._signature(text)
        band_hashes = self._band_hashes(sig)

        with self._lock:
            # Zbierz kandydatów przez wszystkie pasma (O(_NUM_BANDS))
            candidates: set[int] = set()
            for b, bh in enumerate(band_hashes):
                for idx in self._band_tables[b].get(bh, []):
                    candidates.add(idx)

            # Weryfikuj dokładnym Jaccard (eliminuje false-positives LSH)
            for idx in candidates:
                existing = self._signatures[idx]
                jaccard = sum(a == b for a, b in zip(sig, existing)) / _NUM_PERM
                if jaccard >= self._threshold:
                    return True

            # Nowy — dodaj do indeksu
            new_idx = len(self._signatures)
            self._signatures.append(sig)
            for b, bh in enumerate(band_hashes):
                self._band_tables[b].setdefault(bh, []).append(new_idx)
            return False

    def register(self, text: str) -> None:
        """Dodaj *text* do indeksu bez sprawdzania duplikatów (do pre-ładowania)."""
        sig = self._signature(text)
        band_hashes = self._band_hashes(sig)
        with self._lock:
            new_idx = len(self._signatures)
            self._signatures.append(sig)
            for b, bh in enumerate(band_hashes):
                self._band_tables[b].setdefault(bh, []).append(new_idx)

    def load_from_jsonl(self, path: str | Path) -> int:
        """
        Wstępnie załaduj indeks z istniejącego pliku JSONL, żeby wznowienie
        przebiegu nie generowało duplikatów pytań już obecnych w datasecie.
        Zwraca liczbę załadowanych pytań.
        """
        path = Path(path)
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    for msg in record.get("messages", []):
                        if msg.get("role") == "user":
                            self.register(msg["content"])
                            count += 1
                            break
                except Exception:
                    pass
        if count:
            logger.info(
                "Deduplicator pre-loaded %d questions from %s (threshold=%.2f, LSH bands=%d)",
                count, path.name, self._threshold, _NUM_BANDS,
            )
        return count

    @property
    def size(self) -> int:
        """Liczba sygnatur w indeksie."""
        return len(self._signatures)


class SemanticDeduplicator:
    """
    Embedding-based semantic duplicate detector (Stage 4 of the dedup pipeline).

    Catches paraphrases that share similar meaning but differ in surface form —
    not caught by MinHash LSH. Uses cosine similarity on dense text embeddings.

    Controlled by settings.semantic_dedup_enabled and settings.semantic_dedup_threshold.

    Usage (inject embed_fn to avoid circular imports):
        from agents.expert import embed_batch

        def _embed(texts):
            vecs, _ = embed_batch(texts)
            return vecs

        dedup = SemanticDeduplicator(embed_fn=_embed)
        if dedup.is_duplicate("Co CSRD nakłada na spółki?"):
            skip ...
    """

    def __init__(
        self,
        threshold: float = 0.88,
        embed_fn=None,
        max_history: int = 5000,
    ) -> None:
        self._threshold = threshold
        self._embed_fn = embed_fn
        self._embeddings: list[list[float]] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def is_duplicate(self, text: str) -> bool:
        """Return True if a semantically similar text was seen before; register otherwise."""
        if self._embed_fn is None:
            return False
        try:
            vec = self._embed_fn([text])[0]
        except Exception as exc:
            logger.debug("SemanticDeduplicator: embed failed (%s)", exc)
            return False
        with self._lock:
            for existing in self._embeddings:
                if self._cosine(vec, existing) >= self._threshold:
                    return True
            # Not a duplicate — register
            self._embeddings.append(vec)
            if len(self._embeddings) > self._max_history:
                self._embeddings.pop(0)
        return False

    @property
    def size(self) -> int:
        return len(self._embeddings)
