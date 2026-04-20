"""
agents/constitutional.py — Constitutional AI (Self-Critique & Revision) - ENTERPRISE EDITION

Implementuje zaawansowany wzorzec Constitutional AI (Anthropic, 2022) oparty na 
Structured Outputs (Pydantic) i precyzyjnej kontroli tokenów.

Fazy działania:
  1. Critique (Structured): LLM ocenia odpowiedź względem Konstytucji i zwraca obiekt JSON.
  2. Revision (Generative): LLM poprawia błędy, zachowując strukturę rozumowania (CoT).

Korzyści PRO:
  - Gwarantowana deterministyczna ocena naruszeń (Pydantic).
  - Precyzyjne liczenie kosztów (FinOps) fazy audytu i poprawek.
  - Zabezpieczenie tagów <reasoning> przed zniszczeniem.
  - Automatyczne darmowe pary DPO (Chosen vs Rejected) gotowe do treningu.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, Any, Tuple, Optional

import openai
import tiktoken
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

# Importujemy router generatywny z eksperta do fazy Rewizji (aby wspierał Local/Cloud)
from agents.expert import _call_vllm 
from config.settings import settings
from pipeline.state import FoundryState

logger = logging.getLogger("foundry.agents.constitutional")

# Globalny klient dla Structured Outputs (Krytyka)
client = openai.OpenAI(api_key=settings.openai_api_key, max_retries=0)

# Ceny dla telemetrii
_COSTS_PER_1M = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (5.00, 15.00),
}


# ---------------------------------------------------------------------------
# Pydantic Schemas (Deterministyczna Krytyka)
# ---------------------------------------------------------------------------
class CritiqueResult(BaseModel):
    """Ścisła struktura audytu konstytucyjnego."""
    has_violations: bool = Field(
        description="Czy wykryto jakiekolwiek naruszenia Konstytucji w odpowiedzi?"
    )
    violated_rules: list[int] = Field(
        default_factory=list,
        description="Lista numerów naruszonych zasad (1-7). Pusta, jeśli brak naruszeń."
    )
    critique_reasoning: str = Field(
        description="Precyzyjne, 2-3 zdaniowe uzasadnienie naruszeń. Instrukcja naprawy dla Rewidenta."
    )


# ---------------------------------------------------------------------------
# Konstytucja (Zasady Jakości)
# ---------------------------------------------------------------------------
_CONSTITUTION = """
ZASADY ODPOWIEDZI ESG (Konstytucja AI — wersja 3.0 PRO):

1. UGRUNTOWANIE: Każde twierdzenie MUSI bezwzględnie wynikać z podanego kontekstu.
   (Naruszenie: dodanie faktów, liczb, dat lub definicji spoza tekstu).
2. CYTOWANIE: Każde twierdzenie prawne musi być poparte numerem artykułu/ustępu.
   (Naruszenie: ogólne stwierdzenia bez wskazania twardej podstawy).
3. KOMPLETNOŚĆ: Odpowiedź musi adresować wszystkie aspekty zadanego pytania.
   (Naruszenie: pominięcie kluczowej części pytania).
4. PRECYZJA PRAWNA: Należy używać dokładnych terminów z dokumentu.
   (Naruszenie: używanie własnych, potocznych parafraz zamiast terminologii prawnej).
5. OSTROŻNOŚĆ EPISTEMICZNA: Gdy kontekst jest niejednoznaczny — zaznacz to wyraźnie.
   (Naruszenie: kategoryczne twierdzenia przy niepewnych danych).
6. FORMAT I JĘZYK: Odpowiedź w języku polskim, styl profesjonalny (2-8 zdań).
   (Naruszenie: zbyt krótka lub rozlazła odpowiedź, potoczny styl).
7. ODMOWA: Jeśli pytanie jest całkowicie poza kontekstem, jedyna dopuszczalna treść to: "Brak danych w dyrektywie."
   (Naruszenie: próba częściowej odpowiedzi lub zgadywania przy braku danych).
"""

_CRITIQUE_SYSTEM = (
    "Jesteś bezwzględnym audytorem jakości systemu AI (Constitutional AI).\n\n"
    "Twoje zadanie: Ocenić odpowiedź asystenta zgodnie z poniższą KONSTYTUCJĄ.\n\n"
    f"{_CONSTITUTION}\n\n"
    "Zidentyfikuj wszelkie naruszenia i zwróć ścisły obiekt JSON określający błędy i zasady, które złamano."
)

_REVISION_SYSTEM = (
    "Jesteś Senior Ekspertem ESG. Twoim zadaniem jest poprawa odpowiedzi AI na podstawie twardego audytu jakości.\n\n"
    "ZASADY REWIZJI:\n"
    "- Przeanalizuj uwagi z audytu i bezwzględnie je wdróż.\n"
    "- Zachowaj poprawną część merytoryczną oryginału.\n"
    "- Usuń halucynacje, dodaj brakujące artykuły.\n"
    "- Pisz po polsku, 2-8 zdań, chłodnym stylem prawniczym.\n"
    "- ZACHOWAJ FORMAT: Zawsze rozpocznij od bloku <reasoning>...</reasoning>, w którym poprawiasz "
    "swój tok myślenia, a następnie podaj nową, poprawną odpowiedź.\n"
    "- NIE dodawaj wiedzy spoza dostarczonego kontekstu."
)


# ---------------------------------------------------------------------------
# Zarządzanie Tokenami (Tiktoken Manager)
# ---------------------------------------------------------------------------
def _truncate_by_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.warning(f"[Constitutional] Ucinam kontekst z {len(tokens)} do {max_tokens} tokenów.")
    return enc.decode(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Odporność Sieciowa dla Audytu
# ---------------------------------------------------------------------------
def _is_retryable_openai(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False

@retry(
    reraise=True,
    retry=retry_if_exception(_is_retryable_openai),
    wait=wait_random_exponential(multiplier=1.5, max=15),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def _call_structured_critique(model: str, messages: list[dict]) -> Tuple[CritiqueResult, float]:
    """Wykonuje audyt zwracając twardy model Pydantic i koszt w centach."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0, # Zero kreatywności, czysty audyt
        response_format=CritiqueResult,
    )
    
    cost_cents = 0.0
    if response.usage and model in _COSTS_PER_1M:
        c_in, c_out = _COSTS_PER_1M[model]
        cost_cents = ((response.usage.prompt_tokens / 1_000_000) * c_in * 100) + \
                     ((response.usage.completion_tokens / 1_000_000) * c_out * 100)
                     
    parsed = response.choices[0].message.parsed
    if not parsed:
        raise ValueError("Model odmówił audytu (Refusal).")
        
    return parsed, cost_cents


# ---------------------------------------------------------------------------
# WĘZEŁ GŁÓWNY (LangGraph Node)
# ---------------------------------------------------------------------------
def constitutional_revision(state: FoundryState) -> Dict[str, Any]:
    """
    Węzeł LangGraph: Constitutional AI.
    Przeprowadza auto-audyt wygenerowanej odpowiedzi. Jeśli wykryje naruszenia Konstytucji,
    wymusza rewizję (przepisanie) i generuje parę DPO.
    """
    if not getattr(settings, "use_constitutional_ai", True):
        return {}

    answer = state.get("answer", "")
    question = state.get("question", "")
    is_adversarial = state.get("is_adversarial", False)
    chunk_id_short = state.get("chunk", {}).get("id", "UNKNOWN")[:8]

    # Fast-Path: Nie rewizuj krótkich odmów
    if is_adversarial or not answer or len(answer.strip()) < 30:
        return {}

    # Ochrona Okna Kontekstowego
    context_parts = state.get("retrieved_context", [state.get("chunk", {}).get("content", "")])
    raw_context = "\n---\n".join(context_parts)
    safe_context = _truncate_by_tokens(raw_context, max_tokens=6000)

    # =========================================================================
    # KROK 1: KRYTYKA (AUDYT STRUKTURALNY)
    # =========================================================================
    critique_prompt = (
        f"[KONTEKST ŹRÓDŁOWY]\n{safe_context}\n\n"
        f"[PYTANIE]\n{question}\n\n"
        f"[ODPOWIEDŹ DO AUDYTU]\n{answer}\n\n"
    )
    
    messages = [
        {"role": "system", "content": _CRITIQUE_SYSTEM},
        {"role": "user", "content": critique_prompt}
    ]

    try:
        start_time = time.perf_counter()
        critique_model = settings.openai_primary_model
        critique_result, critique_cost = _call_structured_critique(critique_model, messages)
        elapsed_critique = time.perf_counter() - start_time
    except Exception as exc:
        logger.error(f"[Constitutional:{chunk_id_short}] Błąd API podczas audytu: {exc}")
        return {}

    # Analiza wyników
    if not critique_result.has_violations:
        logger.debug(f"[Constitutional:{chunk_id_short}] ✓ Czysta odpowiedź. Brak naruszeń (Koszt: {critique_cost:.4f}¢)")
        return {"constitutional_critique": "Brak naruszeń."}

    rules_str = ", ".join(map(str, critique_result.violated_rules))
    logger.info(
        f"⚠️ [Constitutional:{chunk_id_short}] Wykryto naruszenia zasad [{rules_str}]. "
        f"Uruchamiam Rewizję... (Koszt audytu: {critique_cost:.4f}¢)"
    )

    # =========================================================================
    # KROK 2: REWIZJA (NAPRAWA BŁĘDÓW)
    # =========================================================================
    revision_user_prompt = (
        f"[KONTEKST ŹRÓDŁOWY]\n{safe_context}\n\n"
        f"[PYTANIE]\n{question}\n\n"
        f"[ORYGINALNA ODPOWIEDŹ Z NARUSZENIAMI]\n{answer}\n\n"
        f"[WYNIK AUDYTU KONSTYTUCYJNEGO]\n"
        f"Złamane zasady: {rules_str}\n"
        f"Uzasadnienie audytora: {critique_result.critique_reasoning}\n\n"
        f"Działaj. Wygeneruj poprawioną odpowiedź zaczynając od bloku <reasoning>."
    )

    try:
        start_time = time.perf_counter()
        # Wykorzystujemy standardowy router z expert.py, który wspiera Local/Cloud
        revised_answer = _call_vllm(_REVISION_SYSTEM, revision_user_prompt)
        elapsed_revision = time.perf_counter() - start_time
    except Exception as exc:
        logger.error(f"[Constitutional:{chunk_id_short}] Błąd API podczas rewizji: {exc}")
        return {"constitutional_critique": critique_result.critique_reasoning}

    # =========================================================================
    # KROK 3: SANITY CHECKS (Weryfikacja Sensowności Rewizji)
    # =========================================================================
    clean_revised = revised_answer.strip()
    
    # 1. Czy model nie usunął wszystkiego?
    if len(clean_revised) < 20:
        logger.warning(f"[Constitutional:{chunk_id_short}] Rewizja za krótka. Odrzucam zmiany.")
        return {"constitutional_critique": critique_result.critique_reasoning}

    # 2. Czy w ogóle coś zmienił?
    # Usuwamy białe znaki do porównania
    if re.sub(r'\s+', '', clean_revised) == re.sub(r'\s+', '', answer):
        logger.warning(f"[Constitutional:{chunk_id_short}] Rewizja jest identyczna z oryginałem. Odrzucam zmiany.")
        return {"constitutional_critique": critique_result.critique_reasoning}

    logger.info(
        f"✨ [Constitutional:{chunk_id_short}] ✓ Sukces. Zrewidowano odpowiedź "
        f"({len(answer)} → {len(revised_answer)} znaków). Czas: {elapsed_revision:.2f}s."
    )

    # Konstruujemy zwrotny stan do Grafu LangGraph (Zapisujemy cenną parę DPO!)
    return {
        "answer": revised_answer,
        "constitutional_critique": f"Złamane zasady: {rules_str}. Uzasadnienie: {critique_result.critique_reasoning}",
        # Najważniejsze dla treningu: Stara odpowiedź staje się 'rejected' (Chosen vs Rejected)
        "rejected_answer": state.get("rejected_answer") or answer,
    }
