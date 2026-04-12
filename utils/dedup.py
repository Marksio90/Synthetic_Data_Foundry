"""
utils/dedup.py — MinHash-based near-duplicate question detection.

Uses word-trigram MinHash signatures to identify questions whose Jaccard
similarity exceeds *threshold*.  Prevents the same question from appearing
multiple times in the dataset under minor paraphrase variations.

No external dependencies — pure stdlib hashlib.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_NUM_PERM = 128  # MinHash permutations; 128 gives ~1% estimation error


class MinHashDeduplicator:
    """
    Thread-safe near-duplicate detector.

    Each call to is_duplicate(text) either:
      - Returns True  (near-duplicate found — caller should skip this record)
      - Returns False (new/unique — signature added to index as a side-effect)

    Usage:
        dedup = MinHashDeduplicator(threshold=0.85)
        dedup.load_from_jsonl("output/dataset.jsonl")   # pre-populate from existing file
        if dedup.is_duplicate(question):
            skip ...
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self._threshold = threshold
        self._signatures: list[list[int]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Signature computation
    # ------------------------------------------------------------------

    def _signature(self, text: str) -> list[int]:
        """Compute MinHash signature using word-trigram shingles."""
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, text: str) -> bool:
        """
        Returns True if *text* is too similar to a previously registered text.
        If False, registers *text* in the index (side-effect).
        """
        sig = self._signature(text)
        with self._lock:
            for existing in self._signatures:
                similarity = sum(a == b for a, b in zip(sig, existing)) / _NUM_PERM
                if similarity >= self._threshold:
                    return True
            self._signatures.append(sig)
            return False

    def register(self, text: str) -> None:
        """Add *text* to the index without a duplicate check (for pre-loading)."""
        sig = self._signature(text)
        with self._lock:
            self._signatures.append(sig)

    def load_from_jsonl(self, path: str | Path) -> int:
        """
        Pre-populate the index from an existing JSONL output file so that
        resume runs don't re-add questions already in the dataset.
        Returns the number of questions loaded.
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
                "Deduplicator pre-loaded %d questions from %s (threshold=%.2f)",
                count, path.name, self._threshold,
            )
        return count

    @property
    def size(self) -> int:
        """Number of signatures currently in the index."""
        return len(self._signatures)
