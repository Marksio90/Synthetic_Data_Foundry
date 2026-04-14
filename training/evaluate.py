"""
training/evaluate.py — Automatyczna ewaluacja wytrenowanego modelu.

Strategia:
  1. Weź val set (20% datasetu JSONL — próbki z record_index % 5 == 0)
  2. Dla każdego pytania: odpytaj model przez Ollama API
  3. Oceń odpowiedzi przez judge (gpt-4o-mini — ten sam co w pipeline)
  4. Wygeneruj raport eval: avg_score, refusal_rate, response_length, examples

Nie wymaga GPU — model musi być wcześniej załadowany do Ollama.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
from pathlib import Path
from typing import Optional

import openai

from config.settings import settings

logger = logging.getLogger(__name__)

_REFUSAL_PHRASE = "Brak danych w dyrektywie"


def _ask_model_ollama(
    model_name: str,
    system_prompt: str,
    question: str,
    ollama_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> str:
    """Query the model through Ollama's OpenAI-compatible API."""
    client = openai.OpenAI(
        base_url=f"{ollama_url}/v1",
        api_key="ollama",  # Ollama doesn't need a real key
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=512,
            timeout=timeout,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Ollama call failed: %s", exc)
        return ""


def _judge_answer(question: str, answer: str, context: str = "") -> float:
    """Quick judge call — returns score 0.0–1.0."""
    from agents.judge import _call_judge_model, _parse_judge_response, _safe_score, _JUDGE_SYSTEM, _JUDGE_USER_TEMPLATE

    if not answer or len(answer) < 20:
        return 0.0

    user_content = _JUDGE_USER_TEMPLATE.format(
        context=context or "Brak kontekstu (eval mode).",
        question=question,
        answer=answer,
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = _call_judge_model(settings.openai_primary_model, messages)
        result = _parse_judge_response(raw)
        return _safe_score(result.get("score", 0.0))
    except Exception as exc:
        logger.warning("Judge failed in eval: %s", exc)
        return 0.0


def evaluate_model(
    model_name: str,
    jsonl_path: str,
    n_samples: int = 50,
    ollama_url: str = "http://localhost:11434",
    seed: int = 42,
) -> dict:
    """
    Evaluate a model loaded in Ollama against a holdout from the JSONL dataset.

    Args:
        model_name:   Ollama model name (e.g., "foundry-esg-3b")
        jsonl_path:   Path to the SFT JSONL dataset
        n_samples:    Number of samples to evaluate (default 50)
        ollama_url:   Ollama API endpoint
        seed:         Random seed for reproducible sampling

    Returns:
        dict with evaluation results and sample examples.
    """
    rng = random.Random(seed)
    path = Path(jsonl_path)
    if not path.exists():
        logger.error("Dataset not found: %s", jsonl_path)
        return {"error": f"Dataset not found: {jsonl_path}"}

    # Load all records
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return {"error": "Empty dataset"}

    # Sample
    sample = rng.sample(records, min(n_samples, len(records)))
    logger.info("Evaluating model '%s' on %d samples...", model_name, len(sample))

    scores: list[float] = []
    refusals: int = 0
    response_lengths: list[int] = []
    examples: list[dict] = []

    for i, rec in enumerate(sample):
        messages = rec.get("messages", [])
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        ref_answer = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        if not user_msg:
            continue

        model_answer = _ask_model_ollama(model_name, sys_msg, user_msg, ollama_url)
        score = _judge_answer(user_msg, model_answer)

        scores.append(score)
        response_lengths.append(len(model_answer))
        if _REFUSAL_PHRASE in model_answer:
            refusals += 1

        if len(examples) < 5 and (score < 0.7 or score > 0.9):
            examples.append({
                "question": user_msg[:200],
                "model_answer": model_answer[:300],
                "ref_answer": ref_answer[:300],
                "score": round(score, 3),
            })

        if (i + 1) % 10 == 0:
            logger.info("  Eval progress: %d/%d, avg_score=%.3f", i + 1, len(sample),
                       statistics.mean(scores) if scores else 0)

    if not scores:
        return {"error": "No valid samples evaluated"}

    result = {
        "model_name": model_name,
        "n_evaluated": len(scores),
        "avg_score": round(statistics.mean(scores), 3),
        "median_score": round(statistics.median(scores), 3),
        "p25_score": round(sorted(scores)[len(scores) // 4], 3),
        "p75_score": round(sorted(scores)[len(scores) * 3 // 4], 3),
        "refusal_pct": round(refusals / len(scores) * 100, 1),
        "avg_response_len": round(statistics.mean(response_lengths), 0) if response_lengths else 0,
        "pass_rate_088": round(sum(1 for s in scores if s >= 0.88) / len(scores) * 100, 1),
        "fail_rate_070": round(sum(1 for s in scores if s < 0.70) / len(scores) * 100, 1),
        "low_score_examples": examples,
    }

    logger.info(
        "Eval complete: avg=%.3f median=%.3f refusal=%.1f%% pass>=0.88: %.1f%%",
        result["avg_score"], result["median_score"],
        result["refusal_pct"], result["pass_rate_088"],
    )

    # Save eval report
    report_path = Path(jsonl_path).parent / f"eval_{model_name.replace('/', '_')}.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Eval report saved: %s", report_path)

    return result
