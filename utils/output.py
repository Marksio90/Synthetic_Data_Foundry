"""
utils/output.py — JSONL writer with ACID guarantees + B2B watermarking.

Design:
  - The file is opened in append mode; each record is a single JSON line.
  - Writing is guarded by a threading.Lock so concurrent callers don't
    interleave partial lines.
  - A record is written inside the same DB transaction as the sample status
    update — if either fails, neither persists (ACID idempotency).
  - Every WATERMARK_INTERVAL-th record has its assistant turn modified with
    the linguistic watermark before writing.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from config.settings import settings
from pipeline.watermark import (
    build_watermark_description,
    compute_watermark_hash,
    inject_watermark,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
    "Odpowiadasz wyłącznie na podstawie dostarczonych fragmentów dyrektyw. "
    "Jeśli informacja nie wynika z tekstu, odpowiedz: \"Brak danych w dyrektywie\"."
)


class JSONLWriter:
    """Thread-safe, append-only JSONL writer with watermark injection."""

    def __init__(
        self,
        output_path: str = settings.output_file,
        client_id: str = settings.client_id,
        batch_id: str = "",
        watermark_interval: int = settings.watermark_interval,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.client_id = client_id
        self.batch_id = batch_id
        self.watermark_interval = watermark_interval
        self._lock = threading.Lock()
        self._record_count = self._count_existing_records()
        self._watermark_positions: list[int] = []

    def _count_existing_records(self) -> int:
        """Resume from the last written line so record_index is monotonic."""
        if not self.output_path.exists():
            return 0
        count = 0
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        logger.info("Resuming from record %d in %s", count, self.output_path)
        return count

    def write_sample(
        self,
        question: str,
        answer: str,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> tuple[int, Optional[str]]:
        """
        Append one ChatML record.  Returns (record_index, watermark_hash).
        watermark_hash is non-None only when a watermark was injected.
        """
        with self._lock:
            idx = self._record_count
            watermark_hash: Optional[str] = None

            # Apply watermark every N records (Self-Check B2B patch)
            if self.watermark_interval > 0 and idx % self.watermark_interval == 0 and idx > 0:
                technique = idx // self.watermark_interval
                answer = inject_watermark(answer, technique)
                watermark_hash = compute_watermark_hash(self.client_id, self.batch_id, idx)
                self._watermark_positions.append(idx)
                logger.debug(
                    "Watermark injected at record %d (technique=%s, hash=%s)",
                    idx,
                    build_watermark_description(technique),
                    watermark_hash[:8] + "...",
                )

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            # Validate round-trip before appending (no truncated JSON)
            json.loads(line)

            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

            self._record_count += 1
            return idx, watermark_hash

    @property
    def watermark_positions(self) -> list[int]:
        return list(self._watermark_positions)

    @property
    def record_count(self) -> int:
        return self._record_count
