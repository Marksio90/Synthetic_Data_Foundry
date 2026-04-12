"""
utils/datacard.py — Dataset statistics and datacard generator.

Reads the output JSONL file at the end of a run and produces:
  1. A human-readable datacard dict (returned)
  2. A <dataset>.datacard.json sidecar file

The datacard documents the dataset for B2B delivery — buyers need this
to evaluate the data quality before purchasing a fine-tuning licence.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

_REFUSAL_PHRASE = "Brak danych w dyrektywie"

# Perspective heuristics based on system prompt substrings
_PERSPECTIVE_HINTS: list[tuple[str, str]] = [
    ("cfo", "cfo"),
    ("prawnik", "radcy prawnego"),
    ("audytor", "biegłego rewidenta"),
    ("cross_doc", "wielu aktów"),
]


def _extract_perspective(system_content: str) -> str:
    lower = system_content.lower()
    for name, hint in _PERSPECTIVE_HINTS:
        if hint in lower:
            return name
    return "unknown"


def _dist(values: list[float | int]) -> dict:
    if not values:
        return {}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "count": n,
        "min": round(sorted_v[0], 1),
        "mean": round(statistics.mean(sorted_v), 1),
        "median": round(statistics.median(sorted_v), 1),
        "p75": round(sorted_v[int(n * 0.75)], 1),
        "p90": round(sorted_v[int(n * 0.90)], 1),
        "max": round(sorted_v[-1], 1),
    }


def generate_datacard(
    jsonl_path: str,
    batch_id: str = "",
    extra_meta: dict | None = None,
) -> dict:
    """
    Parse *jsonl_path* and write a companion .datacard.json file.
    Returns the datacard dict.
    """
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning("Datacard: output file not found: %s", jsonl_path)
        return {}

    total = 0
    refusals = 0
    turns_list: list[int] = []
    q_len_list: list[int] = []
    a_len_list: list[int] = []
    perspectives: dict[str, int] = {}
    has_reasoning: int = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = record.get("messages", [])
            total += 1

            user_msgs = [m for m in messages if m["role"] == "user"]
            asst_msgs = [m for m in messages if m["role"] == "assistant"]
            sys_msgs  = [m for m in messages if m["role"] == "system"]

            turns_list.append(len(user_msgs))

            if any(_REFUSAL_PHRASE in m["content"] for m in asst_msgs):
                refusals += 1

            if user_msgs:
                q_len_list.append(len(user_msgs[0]["content"]))
            if asst_msgs:
                last_a = asst_msgs[-1]["content"]
                a_len_list.append(len(last_a))
                if "<reasoning>" in last_a:
                    has_reasoning += 1

            if sys_msgs:
                p = _extract_perspective(sys_msgs[0]["content"])
                perspectives[p] = perspectives.get(p, 0) + 1

    def _pct(n: int) -> float:
        return round(100.0 * n / total, 1) if total else 0.0

    multi_turn_count = sum(1 for t in turns_list if t > 1)

    card: dict = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "format": "ChatML JSONL (multi-turn)",
        "language": "Polish (pl)",
        "domain": "ESG / EU Directives (CSRD, SFDR, EU Taxonomy, CSDDD)",
        "license": "Proprietary — B2B licence",
        # ── Volume ───────────────────────────────────────────────────────────
        "total_records": total,
        "refusal_count": refusals,
        "refusal_pct": _pct(refusals),
        "cot_reasoning_pct": _pct(has_reasoning),
        # ── Conversation structure ────────────────────────────────────────────
        "turns": {
            "distribution": _dist(turns_list),
            "single_turn_count": total - multi_turn_count,
            "multi_turn_count": multi_turn_count,
            "multi_turn_pct": _pct(multi_turn_count),
        },
        # ── Perspective distribution ──────────────────────────────────────────
        "perspective_distribution": {
            k: {"count": v, "pct": _pct(v)}
            for k, v in sorted(perspectives.items(), key=lambda x: -x[1])
        },
        # ── Length statistics ─────────────────────────────────────────────────
        "question_length_chars": _dist(q_len_list),
        "answer_length_chars":   _dist(a_len_list),
    }

    if extra_meta:
        card.update(extra_meta)

    # Write sidecar
    out_path = path.with_suffix(".datacard.json")
    out_path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Datacard written → %s  (%d records, %.1f%% refusals, %.1f%% multi-turn)",
        out_path.name, total, _pct(refusals), _pct(multi_turn_count),
    )
    return card
