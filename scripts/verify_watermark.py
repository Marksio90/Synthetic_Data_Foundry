#!/usr/bin/env python3
"""
scripts/verify_watermark.py — Weryfikacja watermarku lingwistycznego.

Sprawdza, czy plik JSONL zawiera prawidłowe watermarki na oczekiwanych pozycjach
(co WATERMARK_INTERVAL rekordów). Używane przez Foundry do potwierdzenia
autentyczności datasetu dostarczonego klientowi B2B.

Użycie:
    python scripts/verify_watermark.py \\
        --jsonl output/dataset_esg_v1.jsonl \\
        --batch-id praw_euro_run_1 \\
        --client-id acme_corp \\
        --interval 50

Kody wyjścia:
    0 — wszystkie watermarki zweryfikowane
    1 — błąd: brak pliku, zbyt mało rekordów
    2 — ostrzeżenie: część watermarków nie pasuje
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("foundry.verify_watermark")

# Secret pepper — musi być taka sama jak w pipeline/watermark.py
_SECRET = os.getenv("FOUNDRY_WATERMARK_SECRET", "esg-foundry-default-secret-change-me")

# Regex wzorce watermarków (muszą pasować do pipeline/watermark.py)
_DOUBLE_SPACE_RE = re.compile(
    r"\b(dyrektywa|directive|rozporządzenie|regulation)  ",  # podwójna spacja
    flags=re.IGNORECASE,
)
_SYNONYM_VARIANTS = [
    "wymogi normatywne",
    "powinności prawne",
    "sprawozdawczość regulacyjna",
    "ujawnienia informacyjne",
]


def _compute_hash(client_id: str, batch_id: str, record_index: int) -> str:
    payload = f"{client_id}|{batch_id}|{record_index}|{_SECRET}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _detect_watermark_in_text(text: str) -> bool:
    """Sprawdź, czy tekst zawiera oznakę watermarku (podwójna spacja LUB synonim)."""
    if _DOUBLE_SPACE_RE.search(text):
        return True
    return any(variant in text for variant in _SYNONYM_VARIANTS)


def _get_assistant_text(messages: list[dict]) -> str:
    """Pobierz tekst z ostatniej odpowiedzi asystenta."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def verify(
    jsonl_path: Path,
    batch_id: str,
    client_id: str,
    interval: int,
) -> tuple[int, int, int]:
    """
    Weryfikuje watermarki w pliku JSONL.

    Zwraca (total_records, expected_watermarks, verified_watermarks).
    """
    if not jsonl_path.exists():
        logger.error("Plik nie istnieje: %s", jsonl_path)
        sys.exit(1)

    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Linia %d: błąd parsowania JSON: %s", line_no, exc)

    total = len(records)
    logger.info("Załadowano %d rekordów z %s", total, jsonl_path.name)

    if total < interval:
        logger.warning(
            "Za mało rekordów (%d) — pierwszy watermark oczekiwany co %d", total, interval
        )

    # Pozycje, gdzie powinny być watermarki (co interval rekordów, zaczynając od interval-1)
    watermark_positions = list(range(interval - 1, total, interval))
    expected = len(watermark_positions)

    if expected == 0:
        logger.info("Brak oczekiwanych watermarków przy %d rekordach i interval=%d", total, interval)
        return total, 0, 0

    verified = 0
    failed_positions: list[int] = []

    for pos in watermark_positions:
        record = records[pos]
        messages = record.get("messages", [])
        metadata = record.get("metadata", {})

        # Sprawdź hash w metadanych (jeśli dostępny)
        stored_hash = metadata.get("watermark_hash", "")
        expected_hash = _compute_hash(client_id, batch_id, pos)

        # Sprawdź wizualny watermark w tekście asystenta
        assistant_text = _get_assistant_text(messages)
        has_visual_mark = _detect_watermark_in_text(assistant_text)

        hash_ok = stored_hash == expected_hash if stored_hash else None
        visual_ok = has_visual_mark

        if hash_ok is True or (hash_ok is None and visual_ok):
            verified += 1
            logger.debug(
                "✓ Pozycja %d: hash=%s visual=%s",
                pos,
                "OK" if hash_ok else "brak",
                "OK" if visual_ok else "NIE",
            )
        else:
            failed_positions.append(pos)
            logger.warning(
                "✗ Pozycja %d: hash=%s visual=%s",
                pos,
                ("OK" if hash_ok else "NIE PASUJE") if hash_ok is not None else "brak",
                "OK" if visual_ok else "NIE ZNALEZIONO",
            )

    logger.info(
        "Wynik: %d/%d watermarków zweryfikowanych (pozycje co %d rekordów)",
        verified, expected, interval,
    )

    if failed_positions:
        logger.warning("Pozycje z problemami: %s", failed_positions[:20])

    return total, expected, verified


def main() -> None:
    p = argparse.ArgumentParser(
        description="Weryfikacja watermarku lingwistycznego w datasecie JSONL"
    )
    p.add_argument("--jsonl", required=True, help="Ścieżka do pliku dataset.jsonl")
    p.add_argument("--batch-id", required=True, help="Identyfikator batcha (batch_id)")
    p.add_argument("--client-id", default="dev_client", help="Identyfikator klienta (client_id)")
    p.add_argument("--interval", type=int, default=50, help="Interwał watermarku (domyślnie 50)")
    args = p.parse_args()

    total, expected, verified = verify(
        jsonl_path=Path(args.jsonl),
        batch_id=args.batch_id,
        client_id=args.client_id,
        interval=args.interval,
    )

    if expected == 0:
        sys.exit(0)

    if verified == expected:
        logger.info("✅ Wszystkie watermarki prawidłowe (%d/%d)", verified, expected)
        sys.exit(0)
    elif verified > 0:
        logger.warning("⚠️  Częściowa weryfikacja: %d/%d watermarków OK", verified, expected)
        sys.exit(2)
    else:
        logger.error("❌ Brak prawidłowych watermarków")
        sys.exit(2)


if __name__ == "__main__":
    main()
