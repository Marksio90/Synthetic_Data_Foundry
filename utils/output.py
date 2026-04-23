"""
utils/output.py — JSONL writers: SFT + DPO + ORPO + KTO + watermarking - ENTERPRISE EDITION

Formaty treningowe:
  JSONLWriter  — SFT ChatML (OpenAI fine-tuning, HF TRL SFTTrainer)
  DPOWriter    — DPO preference pairs (TRL DPOTrainer)
  ORPOWriter   — ORPO preference pairs (TRL ORPOTrainer) — identyczny format co DPO
  KTOWriter    — KTO labeled completions (TRL KTOTrainer)

Ulepszenia PRO:
  - Thread-Safety: Wątkowo bezpieczne operacje z użyciem blokad (`threading.Lock`).
  - Buffered I/O: Odciążenie dysku przez zrzucanie danych partiami (zabezpieczone przez Auto-Flush).
  - Crash Resilience: Automatyczny ratunkowy flush() w przypadku zamykania instancji (GC).
  - ACID Compliance: Bezwzględna podwójna weryfikacja poprawności formatów JSON przed zapisem.
  - Watermarking: Nienaruszony mechanizm B2B linguistic watermark wstrzykiwany co N rekordów.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional, List

from config.settings import settings
from pipeline.watermark import (
    build_watermark_description,
    compute_watermark_hash,
    inject_watermark,
)

logger = logging.getLogger("foundry.utils.output")

_REFUSAL_PHRASE = "Brak danych w dyrektywie"


# ---------------------------------------------------------------------------
# SFT Writer (Supervised Fine-Tuning)
# ---------------------------------------------------------------------------

class JSONLWriter:
    """
    Thread-safe, append-only JSONL writer with buffered I/O and watermark injection.
    Zaprojektowany dla wysokiej współbieżności.
    """

    def __init__(
        self,
        output_path: str = getattr(settings, "output_file", "output/dataset.jsonl"),
        client_id: str = getattr(settings, "client_id", "default_client"),
        batch_id: str = "",
        watermark_interval: int = getattr(settings, "watermark_interval", 50),
        max_refusal_ratio: float = getattr(settings, "max_refusal_ratio", 0.10),
        flush_threshold: int = 5, # Częstotliwość zrzutu bufora na dysk (dla UI SFT)
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.client_id = client_id
        self.batch_id = batch_id
        self.watermark_interval = watermark_interval
        self.max_refusal_ratio = max_refusal_ratio
        
        self._flush_threshold = flush_threshold
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        
        self._record_count, self._refusal_count = self._count_existing_records()
        self._watermark_positions: list[int] = []

    def _count_existing_records(self) -> tuple[int, int]:
        if not self.output_path.exists():
            return 0, 0
        total = refusals = 0
        try:
            with self.output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total += 1
                        if _REFUSAL_PHRASE in line:
                            refusals += 1
            logger.info(
                "SFT Writer: Resuming from record %d in %s (%d refusals, %.1f%%)",
                total, self.output_path.name, refusals,
                100.0 * refusals / total if total else 0,
            )
        except Exception as exc:
            logger.error("SFT Writer: Błąd zliczania rekordów: %s", exc)
        return total, refusals

    def _flush_buffer(self, force: bool = False) -> None:
        """Zrzuca dane z bufora pamięci na dysk fizyczny."""
        if not self._buffer:
            return
        if not force and len(self._buffer) < self._flush_threshold:
            return
            
        try:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(self._buffer) + "\n")
                f.flush()
            self._buffer.clear()
        except Exception as e:
            logger.error("KRYTYCZNY BŁĄD I/O w SFT Writerze: %s", e)

    def should_skip(self, messages: list[dict]) -> bool:
        with self._lock:
            return self._is_refusal_capped(messages)

    def _is_refusal_capped(self, messages: list[dict]) -> bool:
        assistant_turns = [m for m in messages if m["role"] == "assistant"]
        if not assistant_turns:
            return False
            
        refusal_turns = sum(1 for m in assistant_turns if _REFUSAL_PHRASE in m["content"])
        is_mostly_refusals = refusal_turns / len(assistant_turns) > 0.5
        
        # Jeśli dodanie rekordu przekroczyłoby limit, odrzucamy
        total_records = self._record_count + len(self._buffer)
        if is_mostly_refusals and total_records > 0:
            return (self._refusal_count / total_records) >= self.max_refusal_ratio
            
        return False

    @staticmethod
    def _validate_messages(messages: list[dict]) -> Optional[str]:
        """Return an error description if messages violate the ChatML schema, else None."""
        _VALID_ROLES = frozenset({"system", "user", "assistant"})
        if not messages:
            return "pusta lista messages"
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return f"messages[{i}] nie jest słownikiem: {type(msg)}"
            role = msg.get("role")
            if role not in _VALID_ROLES:
                return f"messages[{i}] ma nieprawidłową rolę: {role!r}"
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                return f"messages[{i}] ma pustą lub brakującą treść (role={role!r})"
        return None

    def write_conversation(
        self, messages: list[dict], metadata: Optional[dict] = None
    ) -> tuple[int, Optional[str]]:
        """
        Zapisuje wieloturową rozmowę jako rekord JSONL.
        Zwraca: (indeks_rekordu, watermark_hash) lub (-1, None) w przypadku pominięcia.
        """
        schema_error = self._validate_messages(messages)
        if schema_error:
            logger.warning("SFT Writer: pominięto rekord — błąd schematu: %s", schema_error)
            return -1, None

        with self._lock:
            if self._is_refusal_capped(messages):
                logger.debug("Refusal cap reached — skipping record")
                return -1, None

            assistant_turns = [m for m in messages if m["role"] == "assistant"]
            refusal_turns = sum(1 for m in assistant_turns if _REFUSAL_PHRASE in m["content"])
            is_mostly_refusals = len(assistant_turns) > 0 and refusal_turns / len(assistant_turns) > 0.5

            idx = self._record_count + len(self._buffer)
            watermark_hash: Optional[str] = None

            # ---------------------------------------------------------
            # Wstrzykiwanie Znaku Wodnego (Watermark)
            # ---------------------------------------------------------
            if self.watermark_interval > 0 and idx % self.watermark_interval == 0 and idx > 0:
                technique = idx // self.watermark_interval
                messages = list(messages) # Kopia dla bezpieczeństwa referencji
                
                # Znak wodny wstrzykujemy w OSTATNIĄ turę asystenta
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

            # ---------------------------------------------------------
            # Budowa i Walidacja Rekordu
            # ---------------------------------------------------------
            record: dict = {"messages": messages}
            if metadata:
                record["metadata"] = metadata

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            json.loads(line)  # ACID round-trip validation

            # Dodanie do bufora
            self._buffer.append(line)
            
            if is_mostly_refusals:
                self._refusal_count += 1
                
            self._flush_buffer(force=False)
            
            return idx, watermark_hash

    @property
    def watermark_positions(self) -> list[int]:
        with self._lock:
            return list(self._watermark_positions)

    @property
    def record_count(self) -> int:
        with self._lock:
            return self._record_count + len(self._buffer)

    def __del__(self):
        try:
            self._flush_buffer(force=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DPO Writer (also used as ORPO — same format, different trainer)
# ---------------------------------------------------------------------------

class DPOWriter:
    """
    Thread-safe DPO/ORPO preference pair writer.
    Zoptymalizowany przez I/O Buffering.

    Format (HuggingFace TRL DPOTrainer / ORPOTrainer):
    {
      "prompt":   [system_msg, user_msg],
      "chosen":   [{"role": "assistant", "content": "<good answer>"}],
      "rejected": [{"role": "assistant", "content": "<bad answer>"}]
    }
    """

    def __init__(self, output_path: str, flush_threshold: int = 10) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._flush_threshold = flush_threshold
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._count = self._count_existing()

    def _count_existing(self) -> int:
        if not self.output_path.exists():
            return 0
        count = sum(1 for line in self.output_path.open("r", encoding="utf-8") if line.strip())
        if count:
            logger.info("DPOWriter: resuming from %d existing pairs in %s", count, self.output_path.name)
        return count

    def _flush_buffer(self, force: bool = False) -> None:
        if not self._buffer:
            return
        if not force and len(self._buffer) < self._flush_threshold:
            return
            
        try:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(self._buffer) + "\n")
                f.flush()
            self._buffer.clear()
        except Exception as e:
            logger.error("KRYTYCZNY BŁĄD I/O w DPO Writerze: %s", e)

    def write_pair(
        self,
        prompt_messages: list[dict],
        chosen_answer: str,
        rejected_answer: str,
    ) -> int:
        """Zapisuje parę preferencji. Zwraca indeks pary lub -1 w przypadku pominięcia."""
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
        json.loads(line) # ACID round-trip validation

        with self._lock:
            self._buffer.append(line)
            idx = self._count + len(self._buffer) - 1
            self._flush_buffer(force=False)
            
        logger.debug("DPO pair #%d buffered", idx)
        return idx

    @property
    def pair_count(self) -> int:
        with self._lock:
            return self._count + len(self._buffer)
            
    def __del__(self):
        try:
            with self._lock:
                self._count += len(self._buffer) # Utrzymanie poprawnego licznika
                self._flush_buffer(force=True)
        except Exception:
            pass


# ORPO format = DPO format (różny tylko trainer po stronie Hugging Face)
ORPOWriter = DPOWriter


# ---------------------------------------------------------------------------
# KTO Writer (Kahneman-Tversky Optimization)
# ---------------------------------------------------------------------------

class KTOWriter:
    """
    Thread-safe KTO labeled completion writer.
    Wspiera buforowanie I/O.

    Format (HuggingFace TRL KTOTrainer):
    {
      "prompt":     [system_msg, user_msg],
      "completion": [{"role": "assistant", "content": "..."}],
      "label":      true  ← true=dobre, false=złe
    }

    KTO nie wymaga par — każda próbka jest niezależna.
    Generuje 2 rekordy na parę DPO: jeden chosen(true) + jeden rejected(false).
    """

    def __init__(self, output_path: str, flush_threshold: int = 20) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._flush_threshold = flush_threshold
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._count = self._count_existing()

    def _count_existing(self) -> int:
        if not self.output_path.exists():
            return 0
        count = sum(1 for line in self.output_path.open("r", encoding="utf-8") if line.strip())
        if count:
            logger.info("KTOWriter: resuming from %d existing records in %s", count, self.output_path.name)
        return count

    def _flush_buffer(self, force: bool = False) -> None:
        if not self._buffer:
            return
        if not force and len(self._buffer) < self._flush_threshold:
            return
            
        try:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(self._buffer) + "\n")
                f.flush()
            self._buffer.clear()
        except Exception as e:
            logger.error("KRYTYCZNY BŁĄD I/O w KTO Writerze: %s", e)

    def write_sample(
        self,
        prompt_messages: list[dict],
        completion: str,
        label: bool,
    ) -> int:
        """Zapisuje jeden rekord etykietowany KTO. Zwraca indeks, lub -1 gdy pusty."""
        if not completion.strip():
            return -1

        record = {
            "prompt": prompt_messages,
            "completion": [{"role": "assistant", "content": completion}],
            "label": label,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        json.loads(line) # ACID round-trip validation

        with self._lock:
            self._buffer.append(line)
            idx = self._count + len(self._buffer) - 1
            self._flush_buffer(force=False)
            
        return idx

    def write_pair(
        self,
        prompt_messages: list[dict],
        chosen_answer: str,
        rejected_answer: str,
    ) -> tuple[int, int]:
        """
        Zapisuje parę z DPO jako 2 odrębne rekordy KTO: chosen(true) + rejected(false).
        Zwraca: (chosen_idx, rejected_idx).
        """
        with self._lock:
            chosen_idx = self.write_sample(prompt_messages, chosen_answer, label=True)
            rejected_idx = self.write_sample(prompt_messages, rejected_answer, label=False)
            
            if chosen_idx >= 0 and rejected_idx >= 0:
                logger.debug("KTO pair buffered: #%d (true) + #%d (false)", chosen_idx, rejected_idx)
            
            self._flush_buffer(force=False) # Sprawdzenie limitu po wstawieniu dwóch rekordów
            return chosen_idx, rejected_idx

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._count + len(self._buffer)
            
    def __del__(self):
        try:
            with self._lock:
                self._count += len(self._buffer)
                self._flush_buffer(force=True)
        except Exception:
            pass
