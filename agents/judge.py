"""
agents/judge.py — Wieloagentowy System Ewaluacji Jakości (Sędzia) - ENTERPRISE EDITION

LangGraph node: judge_answer(state) → partial state update

Ten plik implementuje zaawansowaną logikę oceny LLM-as-a-Judge z wykorzystaniem:
1. Pydantic Structured Outputs (Gwarancja formatu JSON Schema).
2. Pydantic Validators (Wymuszanie logiki biznesowej, np. kary za halucynacje).
3. Tiktoken Context Management (Precyzyjne zarządzanie oknem kontekstowym).
4. Token Economics (Dokładne wyliczanie kosztów per próbka w centach).
5. Advanced Fallback Cascade (Przejście na GPT-4o przy niskiej pewności z użyciem Deep Think).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Any, Tuple, Optional

import openai
import tiktoken
from pydantic import BaseModel, Field, model_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from config.settings import settings
from pipeline.state import FoundryState

logger = logging.getLogger("foundry.agents.judge")

# ---------------------------------------------------------------------------
# Inicjalizacja Klienta i Optymalizacja Zapytań
# ---------------------------------------------------------------------------
# Globalny klient HTTP dla OpenAI - współdzielenie puli połączeń
client = openai.OpenAI(api_key=settings.openai_api_key)

# Ekonomia: Ceny za 1M tokenów (Input / Output) w USD na rok 2024/2025
_COSTS_PER_1M = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (5.00, 15.00),
    "o1-mini": (3.00, 12.00),
}


# ---------------------------------------------------------------------------
# Pydantic Schemas & Biznesowa Walidacja Wyników
# ---------------------------------------------------------------------------
class JudgeDetails(BaseModel):
    """
    Ścisła struktura odpowiedzi Sędziego. Wykorzystuje JSON Schema do wymuszenia 
    formatu wyjściowego od API OpenAI.
    """
    grounding_score: float = Field(ge=0.0, le=1.0, description="Ocena ugruntowania odpowiedzi w kontekście źródłowym (1.0 = pełne pokrycie, 0.0 = brak).")
    citation_score: float = Field(ge=0.0, le=1.0, description="Ocena dokładności cytowania numerów artykułów/ustępów w odpowiedzi.")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Miara wyczerpania wszystkich wątków zadanych w pytaniu.")
    language_score: float = Field(ge=0.0, le=1.0, description="Ocena naturalności, profesjonalizmu i czytelności języka polskiego.")
    has_hallucination: bool = Field(description="Flaga krytyczna: Czy wykryto zmyślenia, fakty spoza dyrektywy lub błędne daty/kwoty?")
    hallucination_detail: str = Field(description="Zwięzły opis zidentyfikowanej halucynacji lub słowo 'brak'.")
    overall_score: float = Field(ge=0.0, le=1.0, description="Finalna ważona ocena matematyczna.")
    reasoning: str = Field(description="Szczegółowe uzasadnienie werdyktu sędziowskiego (1-3 zdania po angielsku dla lepszego rozumowania LLM).")
    confidence: float = Field(ge=0.0, le=1.0, description="Pewność modelu co do wystawionej oceny (używane do wyzwalania kaskady GPT-4o).")

    @model_validator(mode='after')
    def enforce_hallucination_penalty(self) -> 'JudgeDetails':
        """
        Walidator na poziomie Pythona. Model LLM może się 'pomylić' w matematyce,
        ale Python nie wybacza. Jeśli model przyznał flagę halucynacji, ale 
        zapomniał odjąć punktów - robimy to siłowo.
        """
        # Teoretyczna waga: (0.40 * grnd) + (0.25 * cite) + (0.20 * comp) + (0.15 * lang)
        calculated_score = (
            (0.40 * self.grounding_score) +
            (0.25 * self.citation_score) +
            (0.20 * self.completeness_score) +
            (0.15 * self.language_score)
        )
        
        if self.has_hallucination:
            # Bezwzględna kara za halucynację
            self.overall_score = min(calculated_score - 0.3, 0.3)
            logger.debug(f"Pydantic Validator: Aplikacja kary za halucynację. Score obniżony do {self.overall_score:.2f}")
        else:
            # Wygładzenie ewentualnych błędów zaokrągleń modelu
            self.overall_score = min(max(calculated_score, 0.0), 1.0)
            
        return self


# ---------------------------------------------------------------------------
# Zarządzanie Kontekstem (Tiktoken Token Window Manager)
# ---------------------------------------------------------------------------
def _truncate_context_by_tokens(context: str, max_tokens: int = 8000, model: str = "gpt-4o-mini") -> str:
    """
    Używa oficjalnego enkodera OpenAI, aby matematycznie idealnie przyciąć 
    kontekst dyrektywy, zapobiegając błędom 'context_length_exceeded'.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base") # Domyślny enkoder dla o1 i gpt-4o

    tokens = encoding.encode(context)
    if len(tokens) <= max_tokens:
        return context

    logger.warning(f"Zbyt długi kontekst ({len(tokens)} tokenów). Przycinam precyzyjnie do {max_tokens} tokenów.")
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


# ---------------------------------------------------------------------------
# Prompty i Analiza Tekstu
# ---------------------------------------------------------------------------
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

_JUDGE_SYSTEM = """Jesteś rygorystycznym audytorem jakości syntetycznych zestawów danych prawnych (ESG).
Twoim zadaniem jest bezwzględna ocena odpowiedzi asystenta AI na podstawie twardego kontekstu z dyrektyw UE.

KRYTERIA PUNKTACJI (Użyj pełnej skali 0.0 do 1.0 dla każdego):
1. UGRUNTOWANIE: 1.0 = 100% twierdzeń ma pokrycie w tekście. 0.0 = totalne zmyślenie.
2. CYTOWANIE: 1.0 = idealnie przywołane jednostki (art., ust.). 0.0 = brak przypisów prawnych.
3. KOMPLETNOŚĆ: 1.0 = wyczerpano pytania. 0.5 = brakuje istotnego wątku.
4. JĘZYK: 1.0 = styl prawniczy/korporacyjny, doskonała polszczyzna.

HALUCYNACJE (has_hallucination): 
Zaznacz True TYLKO w przypadku, gdy model podaje konkretne zmyślone fakty (np. fałszywe progi finansowe, zmyślone daty wejścia w życie dyrektywy, których nie ma w tekście).

Wypełnij precyzyjnie obiekt JSON. Wymagam bezstronnej i surowej oceny.
"""

_JUDGE_USER_TEMPLATE = """[KONTEKST ZRODŁOWY - DYREKTYWY UE]
{context}

[ZAPYTANIE UŻYTKOWNIKA]
{question}

[ODPOWIEDŹ ASYSTENTA DO OCENY]
{answer}
"""


# ---------------------------------------------------------------------------
# System Odporności Sieciowej (Resilience & Retry)
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False

def _retry_openai(func):
    return retry(
        reraise=True,
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(
            multiplier=1.5,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


# ---------------------------------------------------------------------------
# Wykonanie API i Ekonomia (Koszty)
# ---------------------------------------------------------------------------
@_retry_openai
def _call_structured_judge(model: str, messages: list[dict], enforce_deep_think: bool = False) -> Tuple[JudgeDetails, float]:
    """
    Wywołuje API OpenAI wymuszając zwrot w Pydantic (Structured Outputs).
    Oblicza precyzyjnie koszt wykorzystania tokenów.
    """
    # Tryb "Deep Think" dla eskalacji (GPT-4o z wyższą temperaturą by przełamać blokadę myślową)
    temperature = 0.3 if enforce_deep_think else 0.0
    
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=JudgeDetails,
            # Zabezpieczenie przed ucięciem wyjścia przez LLM
            max_completion_tokens=1024,
        )
    except Exception as api_err:
        # Przechwycenie LengthFinishReasonError i innych błędów parsowania OpenAI
        logger.error(f"Structured Output API Error w modelu {model}: {api_err}")
        raise

    # Telemetria: Obliczanie kosztów zapytania
    usage = response.usage
    cost_cents = 0.0
    if usage and model in _COSTS_PER_1M:
        cost_in, cost_out = _COSTS_PER_1M[model]
        cost_cents = ((usage.prompt_tokens / 1_000_000) * cost_in * 100) + \
                     ((usage.completion_tokens / 1_000_000) * cost_out * 100)

    parsed_result = response.choices[0].message.parsed
    if not parsed_result:
        raise ValueError("Model odmówił wygenerowania struktury JSON (Refusal przy użyciu Structured Outputs).")
        
    return parsed_result, cost_cents


def _fallback_parse_result(reasoning: str) -> JudgeDetails:
    """Tworzy awaryjny, pusty obiekt w przypadku absolutnej klęski API, zapobiegając usterce całego grafu."""
    return JudgeDetails(
        grounding_score=0.0, citation_score=0.0, completeness_score=0.0,
        language_score=0.0, has_hallucination=True, hallucination_detail="FATAL API ERROR",
        overall_score=0.0, reasoning=f"System Fallback: {reasoning}", confidence=0.0
    )


# ---------------------------------------------------------------------------
# Główny Węzeł Orkiestratora LangGraph
# ---------------------------------------------------------------------------
_MIN_ANSWER_CHARS = 30

def judge_answer(state: FoundryState) -> Dict[str, Any]:
    """
    Główny punkt wejścia LangGraph.
    Ocenia odpowiedź przy użyciu modeli AI, wdrażając ścieżki Fast-Path dla prostych zapytań 
    oraz głęboką analizę (kaskadę) dla skomplikowanych zagadnień prawnych.
    """
    is_adversarial = state.get("is_adversarial", False)
    answer = state.get("answer", "")
    answer_eval = _strip_reasoning(answer)
    chunk_id_short = state.get("chunk", {}).get("id", "UNKNOWN")[:8]
    turn_count = state.get("turn_count", 0)

    # =========================================================
    # FAZA 1: FILTRY ZASADOWE (RULE-BASED FAST-PATHS)
    # =========================================================
    
    if not is_adversarial and len(answer_eval) < _MIN_ANSWER_CHARS:
        logger.warning(f"[JUDGE:{chunk_id_short}] Odpowiedź za krótka ({len(answer_eval)} znaków). Fast-Reject.")
        return {
            "quality_score": 0.0,
            "judge_model": "rule-based-fastpath",
            "judge_reasoning": f"Niewystarczająca długość odpowiedzi ({len(answer_eval)} znaków < wymaganych {_MIN_ANSWER_CHARS}).",
            "judge_details": {"fast_path": True, "reason": "too_short"},
        }

    if is_adversarial:
        if _is_refusal(answer_eval):
            logger.debug(f"[JUDGE:{chunk_id_short}] Adversarial Prawidłowo Zablokowany -> score 1.0")
            return {
                "quality_score": 1.0,
                "judge_model": "rule-based-adversarial",
                "judge_reasoning": "Model poprawnie zastosował odmowę ('Brak danych w dyrektywie') dla zapytania adwersaryjnego.",
                "judge_details": {"adversarial": True, "refused": True},
            }
        else:
            logger.warning(f"[JUDGE:{chunk_id_short}] Adversarial ZŁAMANY (Halucynacja na pułapce) -> score 0.0")
            return {
                "quality_score": 0.0,
                "judge_model": "rule-based-adversarial",
                "judge_reasoning": "Model złamał wytyczne i udzielił odpowiedzi na pytanie celowo wprowadzające w błąd.",
                "judge_details": {"adversarial": True, "refused": False, "has_hallucination": True},
            }

    # =========================================================
    # FAZA 2: PRZYGOTOWANIE KONTEKSTU (TIKTOKEN OPTIMIZATION)
    # =========================================================
    context_parts = state.get("retrieved_context", [state.get("chunk", {}).get("content", "")])
    raw_context = "\n---\n".join(context_parts)
    
    # Ucinanie Tokenowe, a nie znakowe! Gwarancja braku błędu przepełnienia
    safe_context = _truncate_context_by_tokens(raw_context, max_tokens=10000, model=settings.openai_primary_model)

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(
            context=safe_context, question=state.get("question", ""), answer=answer_eval
        )},
    ]

    # =========================================================
    # FAZA 3: INTELIGENTNA EWALUACJA (LLM CASCADE)
    # =========================================================
    total_cost_cents = 0.0
    judge_model = settings.openai_primary_model
    
    try:
        # Krok 1: Próba tanim modelem (gpt-4o-mini)
        result_obj, cost = _call_structured_judge(judge_model, messages)
        total_cost_cents += cost

        # Krok 2: Kaskada w przypadku niskiej pewności Sędziego
        if result_obj.confidence < settings.judge_confidence_threshold:
            fallback_model = settings.openai_fallback_model
            logger.info(
                f"[JUDGE:{chunk_id_short}] Sędzia {judge_model} niepewny (Confidence: {result_obj.confidence:.2f}). "
                f"Wyzwalam kaskadę Deep Think na model {fallback_model}."
            )
            try:
                # Wymuszamy głębsze myślenie (temperature 0.3) przy re-ewaluacji
                result_obj_fallback, cost_fallback = _call_structured_judge(fallback_model, messages, enforce_deep_think=True)
                total_cost_cents += cost_fallback
                
                # Tylko jeśli nowy Sędzia jest pewniejszy, nadpisujemy wynik
                if result_obj_fallback.confidence >= result_obj.confidence:
                    result_obj = result_obj_fallback
                    judge_model = fallback_model
                else:
                    logger.debug(f"[JUDGE:{chunk_id_short}] Fallback był jeszcze mniej pewny. Zostaję przy pierwszej ocenie.")
                    
            except Exception as cascade_exc:
                logger.warning(f"[JUDGE:{chunk_id_short}] Błąd Kaskady Fallback ({fallback_model}): {cascade_exc}. Użyto oceny pierwotnej.")

    except Exception as exc:
        logger.error(f"[JUDGE:{chunk_id_short}] ZAWALENIE SYSTEMU SĘDZIOWSKIEGO: {exc}", exc_info=True)
        result_obj = _fallback_parse_result(reasoning=str(exc))
        judge_model = "FATAL_ERROR"

    # =========================================================
    # FAZA 4: TELEMETRIA I AKTUALIZACJA STANU
    # =========================================================
    judge_details_dict = result_obj.model_dump()
    judge_details_dict["eval_cost_cents"] = round(total_cost_cents, 4)

    logger.info(
        f"⚖️ [VERDICT] {chunk_id_short} | Turn:{turn_count} | Mdl:{judge_model} | "
        f"Score:{result_obj.overall_score:.2f} (G:{result_obj.grounding_score:.2f}, "
        f"C:{result_obj.citation_score:.2f}, L:{result_obj.language_score:.2f}) | "
        f"Koszt: {total_cost_cents:.4f}¢ | Halluc:{result_obj.has_hallucination}"
    )

    return {
        "quality_score": result_obj.overall_score,
        "judge_model": judge_model,
        "judge_reasoning": result_obj.reasoning,
        "judge_details": judge_details_dict,
    }
