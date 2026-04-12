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

    def should_skip(self, messages: list[dict]) -> bool:
        """
        Thread-safe pre-check: would write_conversation skip this record?
        Call this BEFORE DB commit to avoid committing a record we won't write.
        """
        with self._lock:
            return self._is_refusal_capped(messages)

    def _is_refusal_capped(self, messages: list[dict]) -> bool:
        """Check if record is a refusal and cap has been reached (not thread-safe, call under lock)."""
        assistant_turns = [m for m in messages if m["role"] == "assistant"]
        if not assistant_turns:
            return False
        refusal_turns = sum(1 for m in assistant_turns if _REFUSAL_PHRASE in m["content"])
        # Count as refusal if majority of turns are "Brak danych"
        is_mostly_refusals = refusal_turns / len(assistant_turns) > 0.5
        if is_mostly_refusals and self._record_count > 0:
            return (self._refusal_count / self._record_count) >= self.max_refusal_ratio
        return False

    def write_conversation(self, messages: list[dict], metadata: Optional[dict] = None) -> tuple[int, Optional[str]]:
        """
        Write a multi-turn conversation as a single JSONL record.
        messages includes system + user/assistant turns.
        metadata: optional dict written as a "metadata" field (non-standard but
                  ignored by OpenAI fine-tuning; usable by HuggingFace TRL).
        Returns (record_index, watermark_hash).
        Returns (-1, None) when skipped due to refusal cap.
        """
        with self._lock:
            # Check refusal cap (majority-based: >50% refusal turns = refusal record)
            if self._is_refusal_capped(messages):
                logger.debug("Refusal cap reached — skipping multi-turn record")
                return -1, None
            assistant_turns = [m for m in messages if m["role"] == "assistant"]
            refusal_turns = sum(1 for m in assistant_turns if _REFUSAL_PHRASE in m["content"])
            is_mostly_refusals = len(assistant_turns) > 0 and refusal_turns / len(assistant_turns) > 0.5

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

            record: dict = {"messages": messages}
            if metadata:
                record["metadata"] = metadata

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            # Validate round-trip before appending (no truncated JSON)
            json.loads(line)

            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

            self._record_count += 1
            if is_mostly_refusals:
                self._refusal_count += 1
            return idx, watermark_hash

    @property
    def watermark_positions(self) -> list[int]:
        return list(self._watermark_positions)

    @property
    def record_count(self) -> int:
        return self._record_count


class DPOWriter:
    """
    Thread-safe writer for DPO (Direct Preference Optimization) preference pairs.

    Output format — one JSON line per pair, compatible with HuggingFace TRL
    DPOTrainer (https://huggingface.co/docs/trl/dpo_trainer):

        {
          "prompt":   [system_msg, user_msg],
          "chosen":   [{"role": "assistant", "content": "<good answer>"}],
          "rejected": [{"role": "assistant", "content": "<bad answer>"}]
        }

    Pairs are generated whenever an initial answer scores below the quality
    threshold and a retry succeeds — the failed answer becomes "rejected",
    the successful one becomes "chosen".
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = self._count_existing()

    def _count_existing(self) -> int:
        if not self.output_path.exists():
            return 0
        count = sum(1 for line in self.output_path.open("r", encoding="utf-8") if line.strip())
        if count:
            logger.info("DPOWriter: resuming from %d existing pairs in %s", count, self.output_path.name)
        return count

    def write_pair(
        self,
        prompt_messages: list[dict],
        chosen_answer: str,
        rejected_answer: str,
    ) -> int:
        """
        Write one DPO preference pair.  Returns pair index, or -1 if skipped.

        prompt_messages: [system_msg, user_msg] without the assistant turn.
        chosen_answer:   the higher-quality answer (score >= threshold).
        rejected_answer: the lower-quality answer (score < threshold).
        """
        if not chosen_answer.strip() or not rejected_answer.strip():
            return -1
        if chosen_answer.strip() == rejected_answer.strip():
            return -1

        record = {
            "prompt": prompt_messages,
            "chosen":   [{"role": "assistant", "content": chosen_answer}],
            "rejected": [{"role": "assistant", "content": rejected_answer}],
        }

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        json.loads(line)  # validate round-trip

        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            idx = self._count
            self._count += 1

        logger.debug("DPO pair #%d written", idx)
        return idx

    @property
    def pair_count(self) -> int:
        return self._count
