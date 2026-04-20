"""
training/quality_gate.py — Automatyczna walidacja datasetu przed treningiem.

Sprawdza zestaw minimalnych wymagań jakościowych.
Jeśli wszystkie pasy → trening startuje automatycznie.
Jeśli jakiś fail → sygnalizuje problem + pyta użytkownika (1 decyzja TAK/NIE).

Minimalne wymagania (konfigurowalne):
  MIN_RECORDS         = 500  (SFT samples)
  MIN_DPO_PAIRS       = 50
  MIN_AVG_SCORE       = 0.82
  MAX_PERSPECTIVE_DOMINANCE = 0.60 (żadna perspektywa nie zajmuje > 60%)
  MIN_MULTI_TURN_PCT  = 0.15 (15% próbek wieloturowych)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

MIN_RECORDS: int = 500
MIN_DPO_PAIRS: int = 50
MIN_AVG_SCORE: float = 0.82
MAX_PERSPECTIVE_DOMINANCE: float = 0.60
MIN_MULTI_TURN_PCT: float = 0.15


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: object
    threshold: object
    message: str


@dataclass
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            icon = "✓" if c.passed else "✗"
            lines.append(f"  {icon} {c.name}: {c.value} (min: {c.threshold}) — {c.message}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        status = "PASS" if self.passed else "FAIL"
        lines.insert(0, f"Quality Gate: {status}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def check_dataset(
    jsonl_path: str,
    dpo_path: Optional[str] = None,
    min_records: int = MIN_RECORDS,
    min_dpo_pairs: int = MIN_DPO_PAIRS,
    min_avg_score: float = MIN_AVG_SCORE,
) -> GateResult:
    """
    Validate a JSONL dataset file before training.
    Returns GateResult with individual check results.
    """
    import json
    import statistics

    checks: list[GateCheck] = []
    warnings: list[str] = []
    path = Path(jsonl_path)
    invalid_json_lines = 0

    if not path.exists():
        return GateResult(
            passed=False,
            checks=[GateCheck("file_exists", False, "missing", "exists", "Plik JSONL nie istnieje")],
        )

    # --- Parse JSONL ---
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_json_lines += 1

    n = len(records)
    if invalid_json_lines:
        warnings.append(f"Pominięto {invalid_json_lines} uszkodzonych linii JSONL")

    # Check 1: minimum records
    checks.append(GateCheck(
        "min_records", n >= min_records, n, min_records,
        "OK" if n >= min_records else f"Za mało próbek ({n} < {min_records})",
    ))

    if n == 0:
        return GateResult(passed=False, checks=checks, warnings=warnings)

    # Check 2: DPO pairs
    dpo_count = 0
    if dpo_path:
        dp = Path(dpo_path)
        if dp.exists():
            with dp.open("r", encoding="utf-8") as f:
                dpo_count = sum(1 for line in f if line.strip())
    checks.append(GateCheck(
        "min_dpo_pairs", dpo_count >= min_dpo_pairs, dpo_count, min_dpo_pairs,
        "OK" if dpo_count >= min_dpo_pairs else "Za mało par DPO",
    ))

    # Check 3: average quality score from metadata
    scores = []
    perspectives: dict[str, int] = {}
    multi_turn_count = 0
    refusal_count = 0
    REFUSAL = "Brak danych w dyrektywie"

    for rec in records:
        msgs = rec.get("messages", [])
        meta = rec.get("metadata", {})

        # Score from metadata if available
        # (records don't store score directly — use metadata heuristic)

        # Perspective
        p = meta.get("perspective", "unknown")
        perspectives[p] = perspectives.get(p, 0) + 1

        # Multi-turn
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        if len(user_msgs) > 1:
            multi_turn_count += 1

        # Refusal
        asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
        if any(REFUSAL in m.get("content", "") for m in asst_msgs):
            refusal_count += 1

        # Score from metadata (preferowane do gate)
        score = meta.get("quality_score")
        if score is None:
            score = meta.get("judge_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        elif isinstance(score, str):
            try:
                scores.append(float(score))
            except ValueError:
                pass

    # Avg score check — use metadata quality score if available
    if scores:
        avg_score = statistics.mean(scores)
    else:
        avg_score = min_avg_score  # assume OK if no scores in JSONL
        warnings.append("Brak score w rekordach JSONL — pominięto sprawdzanie avg_score")

    checks.append(GateCheck(
        "min_avg_score",
        avg_score >= min_avg_score,
        round(avg_score, 3),
        min_avg_score,
        "OK" if avg_score >= min_avg_score else f"Zbyt niski średni score ({avg_score:.3f})",
    ))

    # Check 4: perspective balance
    if perspectives:
        max_pct = max(perspectives.values()) / n
        checks.append(GateCheck(
            "perspective_balance",
            max_pct <= MAX_PERSPECTIVE_DOMINANCE,
            f"{max(perspectives, key=perspectives.get)}: {max_pct:.0%}",
            f"≤{MAX_PERSPECTIVE_DOMINANCE:.0%}",
            "OK" if max_pct <= MAX_PERSPECTIVE_DOMINANCE else
            f"Perspektywa dominuje ({max_pct:.0%} > {MAX_PERSPECTIVE_DOMINANCE:.0%})",
        ))

    # Check 5: multi-turn percentage
    mt_pct = multi_turn_count / n
    checks.append(GateCheck(
        "multi_turn_pct",
        mt_pct >= MIN_MULTI_TURN_PCT,
        f"{mt_pct:.0%}",
        f"≥{MIN_MULTI_TURN_PCT:.0%}",
        "OK" if mt_pct >= MIN_MULTI_TURN_PCT else
        "Za mało próbek wieloturowych",
    ))

    # Warnings (non-blocking)
    refusal_pct = refusal_count / n
    if refusal_pct > 0.20:
        warnings.append(f"Wysoki odsetek odmów: {refusal_pct:.0%} (rekomendowane < 20%)")
    if n < 1000:
        warnings.append(f"Dataset mały ({n} próbek) — rekomendowane 1000+ dla dobrej jakości")

    all_passed = all(c.passed for c in checks)
    result = GateResult(passed=all_passed, checks=checks, warnings=warnings)

    logger.info("QualityGate: %s (%d records)", "PASS" if all_passed else "FAIL", n)
    for c in checks:
        logger.info("  %s %s: %s", "✓" if c.passed else "✗", c.name, c.message)

    return result
