"""
training/datacard.py — Generuje datacard.json ze statystykami datasetu.

Datacard jest pakowany do klienta w ZIP razem z modelem,
aby klient widział skąd pochodzi wiedza modelu i jak dobra jest jakość.
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_datacard(
    jsonl_path: str,
    dpo_path: Optional[str] = None,
    output_path: Optional[str] = None,
    domain_label: str = "ESG / Prawo korporacyjne UE",
    base_model: str = "",
    client_id: str = "",
) -> str:
    """
    Analyse the SFT JSONL dataset and write a datacard JSON file.

    Args:
        jsonl_path:   Path to SFT JSONL dataset.
        dpo_path:     Path to DPO JSONL (optional).
        output_path:  Where to write datacard JSON. Defaults to <jsonl>.datacard.json.
        domain_label: Human-readable domain name.
        base_model:   Base model name used for training.
        client_id:    Client identifier (optional).

    Returns:
        Path to the written datacard JSON file.
    """
    jsonl_file = Path(jsonl_path)
    if output_path is None:
        output_path = str(jsonl_file.with_suffix(".datacard.json"))

    card: dict = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "domain": domain_label,
        "base_model": base_model,
        "client_id": client_id,
        "dataset": {},
        "quality": {},
        "perspectives": {},
        "difficulties": {},
        "dpo": {},
    }

    # -----------------------------------------------------------------------
    # Analyse SFT dataset
    # -----------------------------------------------------------------------
    records: list[dict] = []
    if jsonl_file.exists():
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        logger.warning("JSONL file not found: %s", jsonl_path)

    n_records = len(records)
    card["dataset"] = {
        "n_records": n_records,
        "jsonl_path": str(jsonl_file.name),
    }

    if records:
        # Quality scores
        scores = [
            float(r.get("metadata", {}).get("quality_score", 0))
            for r in records
            if r.get("metadata", {}).get("quality_score") is not None
        ]
        if scores:
            card["quality"] = {
                "avg_score": round(statistics.mean(scores), 3),
                "median_score": round(statistics.median(scores), 3),
                "min_score": round(min(scores), 3),
                "max_score": round(max(scores), 3),
                "pass_rate_088": round(
                    sum(1 for s in scores if s >= 0.88) / len(scores) * 100, 1
                ),
                "fail_rate_070": round(
                    sum(1 for s in scores if s < 0.70) / len(scores) * 100, 1
                ),
                "n_scored": len(scores),
            }

        # Perspectives breakdown
        perspectives: dict[str, int] = {}
        for r in records:
            p = r.get("metadata", {}).get("perspective") or r.get("perspective", "unknown")
            perspectives[p] = perspectives.get(p, 0) + 1
        card["perspectives"] = {
            k: {"count": v, "pct": round(v / n_records * 100, 1)}
            for k, v in sorted(perspectives.items(), key=lambda x: -x[1])
        }

        # Difficulties breakdown
        difficulties: dict[str, int] = {}
        for r in records:
            d = r.get("metadata", {}).get("difficulty") or r.get("difficulty", "unknown")
            difficulties[d] = difficulties.get(d, 0) + 1
        card["difficulties"] = {
            k: {"count": v, "pct": round(v / n_records * 100, 1)}
            for k, v in sorted(difficulties.items(), key=lambda x: -x[1])
        }

        # Multi-turn stats
        multi_turn = sum(
            1 for r in records
            if len([m for m in r.get("messages", []) if m.get("role") == "user"]) > 1
        )
        card["dataset"]["multi_turn_count"] = multi_turn
        card["dataset"]["multi_turn_pct"] = round(multi_turn / n_records * 100, 1)

        # Avg response length
        assistant_lengths = []
        for r in records:
            for msg in r.get("messages", []):
                if msg.get("role") == "assistant":
                    assistant_lengths.append(len(msg.get("content", "")))
                    break
        if assistant_lengths:
            card["quality"]["avg_response_len"] = round(
                statistics.mean(assistant_lengths), 0
            )

    # -----------------------------------------------------------------------
    # Analyse DPO dataset
    # -----------------------------------------------------------------------
    dpo_records: list[dict] = []
    if dpo_path:
        dpo_file = Path(dpo_path)
        if dpo_file.exists():
            with dpo_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            dpo_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    card["dpo"] = {
        "n_pairs": len(dpo_records),
        "dpo_path": Path(dpo_path).name if dpo_path else None,
    }

    # -----------------------------------------------------------------------
    # Write datacard
    # -----------------------------------------------------------------------
    Path(output_path).write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Datacard written: %s (%d records)", output_path, n_records)
    return output_path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Generate dataset datacard JSON")
    p.add_argument("--jsonl", required=True, help="Path to SFT JSONL")
    p.add_argument("--dpo", default="", help="Path to DPO JSONL")
    p.add_argument("--output", default="", help="Output path (default: <jsonl>.datacard.json)")
    p.add_argument("--domain", default="ESG / Prawo korporacyjne UE")
    p.add_argument("--base-model", default="")
    p.add_argument("--client-id", default="")
    args = p.parse_args()

    out = generate_datacard(
        jsonl_path=args.jsonl,
        dpo_path=args.dpo or None,
        output_path=args.output or None,
        domain_label=args.domain,
        base_model=args.base_model,
        client_id=args.client_id,
    )
    print(f"Datacard saved: {out}")
