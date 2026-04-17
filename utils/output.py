"""
utils/output.py — JSONL writers: SFT + DPO + ORPO + KTO + watermarking.

Formaty treningowe:
  JSONLWriter  — SFT ChatML (OpenAI fine-tuning, HF TRL SFTTrainer)
  DPOWriter    — DPO preference pairs (TRL DPOTrainer)
  ORPOWriter   — ORPO preference pairs (TRL ORPOTrainer) — identyczny format co DPO
  KTOWriter    — KTO labeled completions (TRL KTOTrainer)

Wszystkie writery są thread-safe (threading.Lock) i ACID (round-trip JSON validation).
Watermarking: co N rekordów → B2B linguistic watermark w ostatniej turze asystenta.
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


# ---------------------------------------------------------------------------
# SFT Writer
# ---------------------------------------------------------------------------

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
        if not self.output_path.exists():
            return 0, 0
        total = refusals = 0
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
        with self._lock:
            return self._is_refusal_capped(messages)

    def _is_refusal_capped(self, messages: list[dict]) -> bool:
        assistant_turns = [m for m in messages if m["role"] == "assistant"]
        if not assistant_turns:
            return False
        refusal_turns = sum(1 for m in assistant_turns if _REFUSAL_PHRASE in m["content"])
        is_mostly_refusals = refusal_turns / len(assistant_turns) > 0.5
        if is_mostly_refusals and self._record_count > 0:
            return (self._refusal_count / self._record_count) >= self.max_refusal_ratio
        return False

    def write_conversation(
        self, messages: list[dict], metadata: Optional[dict] = None
    ) -> tuple[int, Optional[str]]:
        """
        Write multi-turn conversation as single JSONL record.
        Returns (record_index, watermark_hash). Returns (-1, None) when skipped.
        """
        with self._lock:
            if self._is_refusal_capped(messages):
                logger.debug("Refusal cap reached — skipping record")
                return -1, None

            assistant_turns = [m for m in messages if m["role"] == "assistant"]
            refusal_turns = sum(1 for m in assistant_turns if _REFUSAL_PHRASE in m["content"])
            is_mostly_refusals = len(assistant_turns) > 0 and refusal_turns / len(assistant_turns) > 0.5

            idx = self._record_count
            watermark_hash: Optional[str] = None

            if self.watermark_interval > 0 and idx % self.watermark_interval == 0 and idx > 0:
                technique = idx // self.watermark_interval
                messages = list(messages)
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
                    "Watermark injected at record %d (technique=%s, hash=%s…)",
                    idx, build_watermark_description(technique), watermark_hash[:8],
                )

            record: dict = {"messages": messages}
            if metadata:
                record["metadata"] = metadata

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            json.loads(line)  # round-trip validation

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


# ---------------------------------------------------------------------------
# DPO Writer (also used as ORPO — same format, different trainer)
# ---------------------------------------------------------------------------

class DPOWriter:
    """
    Thread-safe DPO/ORPO preference pair writer.

    Format (HuggingFace TRL DPOTrainer / ORPOTrainer):
    {
      "prompt":   [system_msg, user_msg],
      "chosen":   [{"role": "assistant", "content": "<good answer>"}],
      "rejected": [{"role": "assistant", "content": "<bad answer>"}]
    }
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
        """Write one DPO/ORPO preference pair. Returns pair index, or -1 if skipped."""
        if not chosen_answer.strip() or not rejected_answer.strip():
            return -1
        if chosen_answer.strip() == rejected_answer.strip():
            return -1

        record = {
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": chosen_answer}],
            "rejected": [{"role": "assistant", "content": rejected_answer}],
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        json.loads(line)

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


# ORPO format = DPO format (różny tylko trainer)
ORPOWriter = DPOWriter


# ---------------------------------------------------------------------------
# KTO Writer (Kahneman-Tversky Optimization)
# ---------------------------------------------------------------------------

class KTOWriter:
    """
    Thread-safe KTO labeled completion writer.

    Format (HuggingFace TRL KTOTrainer):
    {
      "prompt":     [system_msg, user_msg],
      "completion": [{"role": "assistant", "content": "..."}],
      "label":      true   ← true=dobre, false=złe
    }

    KTO nie wymaga par — każda próbka jest niezależna.
    Generuje 2 rekordy na parę DPO: jeden chosen(true) + jeden rejected(false).
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
            logger.info("KTOWriter: resuming from %d existing records in %s", count, self.output_path.name)
        return count

    def write_sample(
        self,
        prompt_messages: list[dict],
        completion: str,
        label: bool,
    ) -> int:
        """Write one KTO labeled sample. Returns index, or -1 if skipped."""
        if not completion.strip():
            return -1

        record = {
            "prompt": prompt_messages,
            "completion": [{"role": "assistant", "content": completion}],
            "label": label,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        json.loads(line)

        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            idx = self._count
            self._count += 1

        return idx

    def write_pair(
        self,
        prompt_messages: list[dict],
        chosen_answer: str,
        rejected_answer: str,
    ) -> tuple[int, int]:
        """
        Write DPO pair as 2 KTO records: chosen(true) + rejected(false).
        Returns (chosen_idx, rejected_idx).
        """
        chosen_idx = self.write_sample(prompt_messages, chosen_answer, label=True)
        rejected_idx = self.write_sample(prompt_messages, rejected_answer, label=False)
        if chosen_idx >= 0 and rejected_idx >= 0:
            logger.debug("KTO pair written: #%d (true) + #%d (false)", chosen_idx, rejected_idx)
        return chosen_idx, rejected_idx

    @property
    def sample_count(self) -> int:
        return self._count
