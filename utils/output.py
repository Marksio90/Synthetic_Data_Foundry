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

_REFUSAL_PHRASE = "Brak danych w dyrektywie"


class JSONLWriter:
    """Thread-safe, append-only JSONL writer with watermark injection."""

    def __init__(
        self,
        output_path: str = settings.output_file,
        client_id: str = settings.client_id,
        batch_id: str = "",
        watermark_interval: int = settings.watermark_interval,
        max_refusal_ratio: float = settings.max_refusal_ratio,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.client_id = client_id
        self.batch_id = batch_id
        self.watermark_interval = watermark_interval
        self.max_refusal_ratio = max_refusal_ratio
        self._lock = threading.Lock()
        self._record_count, self._refusal_count = self._count_existing_records()
        self._watermark_positions: list[int] = []

    def _count_existing_records(self) -> tuple[int, int]:
        """Resume from the last written line; also count existing refusals."""
        if not self.output_path.exists():
            return 0, 0
        total = 0
        refusals = 0
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
                    if _REFUSAL_PHRASE in line:
                        refusals += 1
        logger.info(
            "Resuming from record %d in %s (%d refusals, %.1f%%)",
            total, self.output_path, refusals,
            100.0 * refusals / total if total else 0,
        )
        return total, refusals

    def write_sample(
        self,
        question: str,
        answer: str,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> tuple[int, Optional[str]]:
        """
        Append one ChatML record.  Returns (record_index, watermark_hash).
        Returns (-1, None) when the record is silently skipped because the
        refusal cap (max_refusal_ratio) has been reached.
        watermark_hash is non-None only when a watermark was injected.
        """
        with self._lock:
            # Cap "Brak danych" refusals at max_refusal_ratio of total output
            is_refusal = _REFUSAL_PHRASE in answer
            if is_refusal and self._record_count > 0:
                current_ratio = self._refusal_count / self._record_count
                if current_ratio >= self.max_refusal_ratio:
                    logger.debug(
                        "Refusal cap %.0f%% reached (%.1f%%) — skipping record",
                        self.max_refusal_ratio * 100,
                        current_ratio * 100,
                    )
                    return -1, None

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
            if is_refusal:
                self._refusal_count += 1
            return idx, watermark_hash

    def write_conversation(self, messages: list[dict]) -> tuple[int, Optional[str]]:
        """
        Write a multi-turn conversation as a single JSONL record.
        messages includes system + user/assistant turns.
        Returns (record_index, watermark_hash).
        Returns (-1, None) when skipped due to refusal cap.
        """
        with self._lock:
            # Check refusal cap: count assistant turns that are pure refusals
            assistant_turns = [m for m in messages if m["role"] == "assistant"]
            refusal_turns = sum(1 for m in assistant_turns if _REFUSAL_PHRASE in m["content"])
            is_all_refusals = len(assistant_turns) > 0 and refusal_turns == len(assistant_turns)

            if is_all_refusals and self._record_count > 0:
                current_ratio = self._refusal_count / self._record_count
                if current_ratio >= self.max_refusal_ratio:
                    logger.debug("Refusal cap reached — skipping multi-turn record")
                    return -1, None

            idx = self._record_count
            watermark_hash: Optional[str] = None

            # Apply watermark to last assistant turn every N records
            if self.watermark_interval > 0 and idx % self.watermark_interval == 0 and idx > 0:
                technique = idx // self.watermark_interval
                # Find last assistant message and inject watermark
                messages = list(messages)  # copy to avoid mutating caller's list
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "assistant":
                        messages[i] = {
                            **messages[i],
                            "content": inject_watermark(messages[i]["content"], technique),
                        }
                        break
                watermark_hash = compute_watermark_hash(self.client_id, self.batch_id, idx)
                self._watermark_positions.append(idx)
                logger.debug(
                    "Watermark injected at record %d (technique=%s, hash=%s)",
                    idx,
                    build_watermark_description(technique),
                    watermark_hash[:8] + "...",
                )

            record = {"messages": messages}
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            # Validate round-trip before appending (no truncated JSON)
            json.loads(line)

            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

            self._record_count += 1
            if is_all_refusals:
                self._refusal_count += 1
            return idx, watermark_hash

    @property
    def watermark_positions(self) -> list[int]:
        return list(self._watermark_positions)

    @property
    def record_count(self) -> int:
        return self._record_count
