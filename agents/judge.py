"""
agents/judge.py — Agent Sędzia (Wielowymiarowy Weryfikator Jakości)

LangGraph node: judge_answer(state) → partial state update

Wielowymiarowa ocena (Gods Finger v2):
  - grounding_score:    czy odpowiedź wynika z kontekstu? (0-1)
  - citation_score:     czy artykuły/ustępy są cytowane? (0-1)
  - completeness_score: czy odpowiedź wyczerpuje pytanie? (0-1)
  - language_score:     jakość języka polskiego? (0-1)
  - hallucination_flag: czy wykryto fakty spoza kontekstu? (bool)
  - overall_score:      ważona średnia (finalna ocena)
  - confidence:         pewność sędziego (→ eskalacja do gpt-4o)

Cascade logic:
  1. gpt-4o-mini (fast + cheap, ~95% przypadków)
  2. gpt-4o (gdy confidence < judge_confidence_threshold, ~5% przypadków)

Adversarial fast-path:
  - is_adversarial + refusal → score=1.0 (reguła, bez API call)
  - is_adversarial + answer  → score=0.0 (hallucynacja)
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

_REFUSAL_VARIANTS = (
    "brak danych w dyrektywie",
    "brak informacji w dyrektywie",
    "nie ma danych w tekście",
    "poza zakresem dyrektywy",
    "brak danych w tekście",
)


def _is_refusal(text: str) -> bool:
    lower = text.lower().strip()
    return any(v in lower for v in _REFUSAL_VARIANTS)


_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>\s*", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    return _REASONING_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Wielowymiarowy system prompt sędziego
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """Jesteś rygorystycznym audytorem jakości zestawów danych ESG.

Oceniasz parę (pytanie, odpowiedź) pod kątem 5 wymiarów:

1. UGRUNTOWANIE (grounding_score 0-1):
   Czy każde twierdzenie w odpowiedzi wynika z dostarczonego kontekstu?
   1.0 = w pełni ugruntowana | 0.0 = wymyślona/poza kontekstem

2. CYTOWANIE (citation_score 0-1):
   Czy odpowiedź cytuje numery artykułów/ustępów jako podstawę prawną?
   1.0 = każde twierdzenie z artykułem | 0.0 = brak cytowań

3. KOMPLETNOŚĆ (completeness_score 0-1):
   Czy odpowiedź adresuje wszystkie aspekty pytania?
   1.0 = pełna | 0.5 = częściowa | 0.0 = nie odpowiada na pytanie

4. JAKOŚĆ JĘZYKOWA (language_score 0-1):
   Czy odpowiedź jest po polsku, profesjonalna, 2-8 zdań?
   1.0 = wzorowa | 0.5 = akceptowalna | 0.0 = nieczytelna/nie po polsku

5. HALUCYNACJA (has_hallucination bool):
   Czy wykryłeś fakty/liczby/daty spoza kontekstu?
   true = tak (naruszenie) | false = czysta odpowiedź

Oblicz overall_score jako: 0.40*grounding + 0.25*citation + 0.20*completeness + 0.15*language
Jeśli has_hallucination=true, odejmij 0.3 od overall_score (min 0.0).

Odpowiedz WYŁĄCZNIE w formacie JSON (bez żadnego innego tekstu):
{
  "grounding_score": <0.0-1.0>,
  "citation_score": <0.0-1.0>,
  "completeness_score": <0.0-1.0>,
  "language_score": <0.0-1.0>,
  "has_hallucination": <true|false>,
  "hallucination_detail": "<co konkretnie jest halucynacją lub 'none'>",
  "overall_score": <0.0-1.0>,
  "reasoning": "<1-2 zdania po angielsku — główne uzasadnienie>",
  "confidence": <0.0-1.0>
}
"""

_JUDGE_USER_TEMPLATE = """KONTEKST DYREKTYWY:
{context}

PYTANIE:
{question}

ODPOWIEDŹ DO OCENY:
{answer}

Oceń zgodnie z instrukcją systemową."""


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
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
    kwargs: dict = dict(model=model, messages=messages, temperature=0.0)
    if "o1" in model:
        kwargs["max_completion_tokens"] = 512
    else:
        kwargs["max_tokens"] = 512
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def _parse_judge_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse judge response: %s", raw[:200])
                return _fallback_parse_result()
        else:
            logger.error("No JSON found in judge response: %s", raw[:200])
            return _fallback_parse_result()
    return data


def _fallback_parse_result() -> dict:
    return {
        "grounding_score": 0.0,
        "citation_score": 0.0,
        "completeness_score": 0.0,
        "language_score": 0.0,
        "has_hallucination": True,
        "hallucination_detail": "parse error",
        "overall_score": 0.0,
        "reasoning": "parse error",
        "confidence": 0.0,
    }


def _safe_score(raw_score: object) -> float:
    try:
        s = float(raw_score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, s))


_MIN_ANSWER_CHARS = 30


def _build_messages(state: FoundryState) -> list[dict]:
    context_parts = state.get("retrieved_context", [state["chunk"]["content"]])
    context_raw = "\n---\n".join(context_parts)
    if len(context_raw) > 8000:
        logger.warning("Judge context TRUNCATED from %d to 8000 chars", len(context_raw))
        context_raw = context_raw[:8000]
    answer_for_eval = _strip_reasoning(state["answer"])
    user_content = _JUDGE_USER_TEMPLATE.format(
        context=context_raw,
        question=state["question"],
        answer=answer_for_eval,
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def judge_answer(state: FoundryState) -> dict:
    """
    LangGraph node.
    Returns {
      "quality_score": float,
      "judge_model": str,
      "judge_reasoning": str,
      "judge_details": dict  ← nowe: grounding/citation/completeness/language/hallucination
    }
    """
    is_adversarial = state.get("is_adversarial", False)
    answer = state.get("answer", "")
    answer_eval = _strip_reasoning(answer)

    # Fast-path: trivially short non-adversarial answers
    if not is_adversarial and len(answer_eval) < _MIN_ANSWER_CHARS:
        logger.warning("Answer too short (%d chars) — rejecting", len(answer_eval))
        return {
            "quality_score": 0.0,
            "judge_model": "rule-based",
            "judge_reasoning": f"Answer too short ({len(answer_eval)} chars < {_MIN_ANSWER_CHARS}).",
            "judge_details": {},
        }

    # Fast-path: adversarial samples
    if is_adversarial:
        if _is_refusal(answer_eval):
            logger.debug("Adversarial: correct refusal → score=1.0")
            return {
                "quality_score": 1.0,
                "judge_model": "rule-based",
                "judge_reasoning": "Correct refusal for adversarial question.",
                "judge_details": {"adversarial": True, "refused": True},
            }
        else:
            logger.warning("Adversarial: answered instead of refused → score=0.0 (hallucination)")
            return {
                "quality_score": 0.0,
                "judge_model": "rule-based",
                "judge_reasoning": "Adversarial question answered instead of refused — hallucination.",
                "judge_details": {"adversarial": True, "refused": False},
            }

    # Normal path: wywołaj gpt-4o-mini z wielowymiarowym promptem
    messages = _build_messages(state)
    try:
        raw = _call_judge_model(settings.openai_primary_model, messages)
        result = _parse_judge_response(raw)
        score: float = _safe_score(result.get("overall_score", 0.0))
        confidence: float = _safe_score(result.get("confidence", 1.0))
        judge_model = settings.openai_primary_model

        # Cascade: eskaluj gdy pewność sędziego niska
        if confidence < settings.judge_confidence_threshold:
            logger.info(
                "Judge confidence %.2f < %.2f — escalating to %s",
                confidence, settings.judge_confidence_threshold, settings.openai_fallback_model,
            )
            try:
                raw2 = _call_judge_model(settings.openai_fallback_model, messages)
                result = _parse_judge_response(raw2)
                score = _safe_score(result.get("overall_score", 0.0))
                judge_model = settings.openai_fallback_model
            except Exception as cascade_exc:
                logger.warning(
                    "Fallback judge (%s) failed: %s — keeping primary result",
                    settings.openai_fallback_model, cascade_exc,
                )

    except Exception as exc:
        logger.error("Judge failed for chunk %s: %s", state["chunk"]["id"], exc)
        score = 0.0
        result = _fallback_parse_result()
        result["reasoning"] = str(exc)
        judge_model = "error"

    # Structured log z wszystkimi wymiarami
    logger.info(
        "JUDGE chunk=%s turn=%d perspective=%s model=%s "
        "score=%.3f grnd=%.2f cite=%.2f comp=%.2f lang=%.2f halluc=%s",
        state["chunk"]["id"][:8],
        state.get("turn_count", 0),
        state.get("perspective", "?"),
        judge_model,
        score,
        _safe_score(result.get("grounding_score", 0)),
        _safe_score(result.get("citation_score", 0)),
        _safe_score(result.get("completeness_score", 0)),
        _safe_score(result.get("language_score", 0)),
        result.get("has_hallucination", False),
    )

    judge_details = {
        "grounding_score": _safe_score(result.get("grounding_score", 0)),
        "citation_score": _safe_score(result.get("citation_score", 0)),
        "completeness_score": _safe_score(result.get("completeness_score", 0)),
        "language_score": _safe_score(result.get("language_score", 0)),
        "has_hallucination": bool(result.get("has_hallucination", False)),
        "hallucination_detail": result.get("hallucination_detail", ""),
    }

    return {
        "quality_score": score,
        "judge_model": judge_model,
        "judge_reasoning": result.get("reasoning", ""),
        "judge_details": judge_details,
    }
