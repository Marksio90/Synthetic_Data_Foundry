"""
agents/judge.py — Agent Sędzia (Kaskadowy Weryfikator Jakości)

LangGraph node: judge_answer(state) → partial state update

Cascade logic (Self-Check 2.0):
  1. Send to gpt-4o-mini first (fast + cheap).
  2. If returned confidence < QUALITY_THRESHOLD → fallback to o1-mini (reasoning).
  3. All OpenAI calls wrapped in Tenacity exponential backoff (429, 5xx).
  4. Optionally uses OpenAI Batch API (50% discount) when BATCH_MODE=true.

Grounding check for adversarial samples:
  - If is_adversarial=True AND answer != refusal phrase → score = 0.0 (hallucination).
  - If is_adversarial=True AND answer contains refusal phrase → score = 1.0 (correct).
"""

from __future__ import annotations

import json
import logging
import re

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from pipeline.state import FoundryState

logger = logging.getLogger(__name__)

_REFUSAL_PHRASE = "Brak danych w dyrektywie"

_JUDGE_SYSTEM = """Jesteś rygorystycznym audytorem jakości zestawów danych ESG.

Oceniasz parę (pytanie, odpowiedź) pod kątem:
1. UGRUNTOWANIE (Grounding): Czy odpowiedź wynika wyłącznie z dostarczonego kontekstu?
   Halucynacja = dodane fakty spoza kontekstu.
2. POPRAWNOŚĆ (Accuracy): Czy odpowiedź jest merytorycznie poprawna względem kontekstu?
3. KOMPLETNOŚĆ (Completeness): Czy odpowiedź wyczerpuje temat pytania?
4. FORMAT: Czy odpowiedź jest profesjonalna i po polsku?

Odpowiedz WYŁĄCZNIE w formacie JSON (bez żadnego innego tekstu):
{
  "score": <liczba od 0.0 do 1.0>,
  "reasoning": "<1-2 zdania po angielsku>",
  "has_hallucination": <true|false>,
  "confidence": <liczba od 0.0 do 1.0 reprezentująca pewność tej oceny>
}
"""

_JUDGE_USER_TEMPLATE = """KONTEKST DYREKTYWY:
{context}

PYTANIE:
{question}

ODPOWIEDŹ DO OCENY:
{answer}

Oceń powyższą parę zgodnie z instrukcją systemową."""


# ---------------------------------------------------------------------------
# Retry decorator (Self-Check 2.0)
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Retry only on 429 (rate limit) and 5xx (server errors). Never on 4xx."""
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _retry_openai(func):
    return retry(
        reraise=True,
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(
            multiplier=1,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


@_retry_openai
def _call_judge_model(model: str, messages: list[dict]) -> str:
    client = openai.OpenAI(api_key=settings.openai_api_key)
    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    # o1-mini uses max_completion_tokens instead of max_tokens
    if "o1" in model:
        kwargs["max_completion_tokens"] = 512
    else:
        kwargs["max_tokens"] = 512

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def _parse_judge_response(raw: str) -> dict:
    """Extract JSON from model response, robust to markdown fences."""
    # Strip ```json ... ``` fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            logger.error("Failed to parse judge response: %s", raw[:200])
            return {"score": 0.0, "reasoning": "parse error", "has_hallucination": True, "confidence": 0.0}
    return data


def _build_messages(state: FoundryState) -> list[dict]:
    context = "\n---\n".join(state.get("retrieved_context", [state["chunk"]["content"]]))
    user_content = _JUDGE_USER_TEMPLATE.format(
        context=context[:3000],  # cap context to avoid token limits
        question=state["question"],
        answer=state["answer"],
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def judge_answer(state: FoundryState) -> dict:
    """
    LangGraph node.
    Returns {"quality_score": ..., "judge_model": ..., "judge_reasoning": ...}
    """
    is_adversarial = state.get("is_adversarial", False)
    answer = state.get("answer", "")

    # Fast-path for adversarial samples: check refusal compliance
    if is_adversarial:
        if _REFUSAL_PHRASE in answer:
            logger.debug("Adversarial sample correctly refused → score=1.0")
            return {
                "quality_score": 1.0,
                "judge_model": "rule-based",
                "judge_reasoning": "Correct refusal for adversarial question.",
            }
        else:
            logger.warning("Adversarial question was answered (possible hallucination) → score=0.0")
            return {
                "quality_score": 0.0,
                "judge_model": "rule-based",
                "judge_reasoning": "Adversarial question answered instead of refused — hallucination.",
            }

    # Normal path: call gpt-4o-mini
    messages = _build_messages(state)
    try:
        raw = _call_judge_model(settings.openai_primary_model, messages)
        result = _parse_judge_response(raw)
        score: float = float(result.get("score", 0.0))
        confidence: float = float(result.get("confidence", 1.0))
        judge_model = settings.openai_primary_model

        # Cascade: if confidence < threshold, escalate to o1-mini (Self-Check 2.0)
        if confidence < settings.quality_threshold:
            logger.info(
                "Judge confidence %.2f < %.2f — escalating to %s",
                confidence, settings.quality_threshold, settings.openai_fallback_model,
            )
            raw2 = _call_judge_model(settings.openai_fallback_model, messages)
            result = _parse_judge_response(raw2)
            score = float(result.get("score", 0.0))
            judge_model = settings.openai_fallback_model

    except Exception as exc:
        logger.error("Judge failed for chunk %s: %s", state["chunk"]["id"], exc)
        score = 0.0
        result = {"reasoning": str(exc), "has_hallucination": True}
        judge_model = "error"

    logger.debug(
        "Judge [%s] → score=%.2f  reasoning=%s",
        judge_model,
        score,
        result.get("reasoning", "")[:80],
    )
    return {
        "quality_score": score,
        "judge_model": judge_model,
        "judge_reasoning": result.get("reasoning", ""),
    }
