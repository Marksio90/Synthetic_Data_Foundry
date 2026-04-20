"""
agents/expert.py — Agent Ekspert (Odpowiadacz + RAG) - ENTERPRISE EDITION

LangGraph node: expert_answer(state) → partial state update

Rozszerzenia PRO:
1. Tokenomics & Precise Windowing: Zastąpienie limitów znakowych precyzyjnym liczeniem tokenów (tiktoken).
2. Global Connection Pools: Współdzieleni klienci OpenAI/Ollama (redukcja narzutu TCP/TLS).
3. XML Context Binding: Precyzyjne oddzielanie wiedzy w tagach <document>, ograniczające halucynacje.
4. FinOps Telemetry: Śledzenie kosztów osadzeń (embeddings) i generacji w czasie rzeczywistym.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Tuple

import openai
import tiktoken
from sqlalchemy.orm import Session
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from config.settings import settings
from db import repository as repo
from pipeline.state import FoundryState
from utils.classifier import classify_question

logger = logging.getLogger("foundry.agents.expert")

_REFUSAL_PHRASE = "Brak danych w dyrektywie."

# Ekonomia: Ceny za 1M tokenów w USD
_COSTS_PER_1M = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (5.00, 15.00),
    "text-embedding-3-small": (0.020, 0.0),
}

# ---------------------------------------------------------------------------
# Globalne Pule Połączeń (Connection Pools)
# ---------------------------------------------------------------------------
_OPENAI_CLIENT = openai.OpenAI(api_key=settings.openai_api_key, max_retries=0)

def _get_ollama_client() -> openai.OpenAI | None:
    if getattr(settings, "ollama_url", None):
        base = settings.ollama_url.rstrip("/")
        # Używamy wyższego timeoutu dla lokalnych modeli
        return openai.OpenAI(api_key="ollama", base_url=f"{base}/v1", max_retries=0, timeout=180.0)
    return None

_OLLAMA_CLIENT = _get_ollama_client()

# ---------------------------------------------------------------------------
# Prompty i Formatowanie
# ---------------------------------------------------------------------------
_COT_FORMAT = (
    "\n\nFORMAT ODPOWIEDZI:\n"
    "Przed udzieleniem odpowiedzi przeprowadź krótkie rozumowanie w tagach:\n"
    "<reasoning>\n"
    "Pytanie dotyczy: [kluczowy element pytania]\n"
    "Relevantne artykuły: [które artykuły/ustępy w kontekście są istotne]\n"
    "Wniosek: [co wynika bezpośrednio z kontekstu]\n"
    "</reasoning>\n\n"
    "Następnie podaj właściwą odpowiedź po polsku (2–8 zdań)."
)

_EXPERT_SYSTEM_BY_PERSPECTIVE: Dict[str, str] = {
    "cfo": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy CFO.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: zakresie podmiotowym, terminach, progach kwalifikacyjnych, wymogach ujawnień.\n"
        "5. Odpowiadaj po polsku, zwięźle i precyzyjnie (2–8 zdań)."
        + _COT_FORMAT
    ),
    "prawnik": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy radcy prawnego.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj podstawę prawną: artykuł, ustęp, punkt.\n"
        "4. Interpretuj literalnie: zakres podmiotowy, przedmiotowy, wyjątki, definicje legalne.\n"
        "5. Odpowiadaj po polsku z precyzją języka prawniczego (2–8 zdań)."
        + _COT_FORMAT
    ),
    "audytor": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy biegłego rewidenta ESG.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: wymogach ujawnień, kryteriach kwalifikacji, wskaźnikach ESG, metodach pomiaru.\n"
        "5. Odpowiadaj po polsku, technicznie i precyzyjnie (2–8 zdań)."
        + _COT_FORMAT
    ),
    "analityk": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy analityka finansowego.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: progach ilościowych, terminach wdrożenia, porównaniu wymogów, wpływie na wycenę.\n"
        "5. Odpowiadaj po polsku, analitycznie i precyzyjnie (2–8 zdań)."
        + _COT_FORMAT
    ),
    "regulator": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy regulatora/nadzorcy.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: obowiązkach regulacyjnych, sankcjach, zakresie nadzoru, wyjątkach podmiotowych.\n"
        "5. Odpowiadaj po polsku, autorytatywnie i precyzyjnie (2–8 zdań)."
        + _COT_FORMAT
    ),
    "akademik": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy badacza akademickiego.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: interpretacji przepisów, kontekście systemowym, spójności z innymi regulacjami.\n"
        "5. Odpowiadaj po polsku, analitycznie i z dystansem naukowym (2–8 zdań)."
        + _COT_FORMAT
    ),
    "dziennikarz": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, wyjaśniającym przepisy przystępnym językiem.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: praktycznym znaczeniu przepisów, kto jest dotknięty, od kiedy, jakie konsekwencje.\n"
        "5. Odpowiadaj po polsku, przystępnie ale precyzyjnie (2–8 zdań)."
        + _COT_FORMAT
    ),
    "inwestor": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy inwestora instytucjonalnego.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w dokumencie.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: obowiązkach ujawnień, ryzyku compliance, due diligence, harmonogramie wymogów.\n"
        "5. Odpowiadaj po polsku, z perspektywy ryzyka i wartości dla portfela (2–8 zdań)."
        + _COT_FORMAT
    ),
}

# Precyzyjne zarządzanie oknem (zastąpienie ułomnego liczenia znaków)
_MAX_CONTEXT_TOKENS = 3000

def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _normalize_weights(vector_weight: float, bm25_weight: float) -> tuple[float, float]:
    """
    Normalizuje wagi hybrydowe do zakresu [0, 1] i sumy 1.0.
    Zabezpiecza przed wartościami ujemnymi, NaN i zerową sumą.
    """
    try:
        vw = max(0.0, float(vector_weight))
    except (TypeError, ValueError):
        vw = 0.0
    try:
        bw = max(0.0, float(bm25_weight))
    except (TypeError, ValueError):
        bw = 0.0

    weight_sum = vw + bw
    if weight_sum <= 0.0:
        return 0.5, 0.5
    return vw / weight_sum, bw / weight_sum


# ---------------------------------------------------------------------------
# Odporność sieciowa (Resilience)
# ---------------------------------------------------------------------------
def _is_retryable_openai(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return False

def _retry_api(func):
    return retry(
        reraise=True,
        retry=retry_if_exception(_is_retryable_openai),
        # Zastosowano Jitter do rozproszenia zapytań
        wait=wait_random_exponential(
            multiplier=1.5,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


# ---------------------------------------------------------------------------
# Moduł Embeddings
# ---------------------------------------------------------------------------
@_retry_api
def _embed_query(query: str) -> Tuple[List[float], float]:
    """
    Generuje wektor dla pytania.
    Zwraca: (Wektor, Koszt_USD_cents)
    """
    cost_cents = 0.0
    
    if settings.use_local_embeddings and _OLLAMA_CLIENT and settings.ollama_embed_model:
        try:
            resp = _OLLAMA_CLIENT.embeddings.create(
                model=settings.ollama_embed_model,
                input=[query],
            )
            return resp.data[0].embedding, cost_cents
        except Exception as e:
            logger.warning(f"Ollama embed niedostępny ({e}), używam chmury OpenAI.")

    # Fallback / Default: OpenAI
    model = getattr(settings, "openai_embedding_model", "text-embedding-3-small")
    resp = _OPENAI_CLIENT.embeddings.create(
        model=model,
        input=[query],
        dimensions=settings.openai_embedding_dims,
    )
    
    usage = resp.usage
    if usage and model in _COSTS_PER_1M:
        cost_in = _COSTS_PER_1M[model][0]
        cost_cents = (usage.prompt_tokens / 1_000_000) * cost_in * 100
        
    return resp.data[0].embedding, cost_cents


# ---------------------------------------------------------------------------
# Wywołanie Modeli (Local -> Cloud Cascade)
# ---------------------------------------------------------------------------
def _call_provider(messages: list[dict]) -> Tuple[str, float]:
    """
    Routing 2-poziomowy ze śledzeniem kosztów:
      1. Ollama LOCAL  (darmowy)
      2. OpenAI        (fallback gdy lokalny niedostępny)
    Zwraca: (Odpowiedź_Tekstowa, Koszt_Centów)
    """
    max_tok = settings.generation_max_tokens
    cost_cents = 0.0

    # ── 1. Ollama LOCAL ──────────────────────────────────────────────
    if settings.ollama_model and _OLLAMA_CLIENT:
        try:
            start_time = time.perf_counter()
            resp = _OLLAMA_CLIENT.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.generation_temperature,
                max_tokens=max_tok,
            )
            elapsed = time.perf_counter() - start_time
            logger.debug(f"[LLM-Local] Wygenerowano odpowiedź w {elapsed:.2f}s (Koszt: 0.0¢).")
            return resp.choices[0].message.content.strip(), cost_cents
        except Exception as e:
            logger.warning(f"[Circuit Breaker] Model Ollama niedostępny ({e}), przełączam na OpenAI.")

    # ── 2. OpenAI fallback ────────────────────────────────────────────
    start_time = time.perf_counter()
    model = settings.openai_primary_model
    resp = _OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=settings.generation_temperature,
        max_tokens=max_tok,
    )
    
    if resp.usage and model in _COSTS_PER_1M:
        u = resp.usage
        c_in, c_out = _COSTS_PER_1M[model]
        cost_cents = ((u.prompt_tokens / 1_000_000) * c_in * 100) + \
                     ((u.completion_tokens / 1_000_000) * c_out * 100)

    elapsed = time.perf_counter() - start_time
    logger.debug(f"[LLM-Cloud] Wygenerowano odpowiedź modelu {model} w {elapsed:.2f}s (Koszt: {cost_cents:.4f}¢).")
    
    return resp.choices[0].message.content.strip(), cost_cents


@_retry_api
def _call_vllm_messages(messages: list[dict]) -> str:
    """Wrapper z retry dla multi-turn context. Odpakowuje sam tekst."""
    answer, _ = _call_provider(messages)
    return answer


@_retry_api
def _call_vllm(system: str, user: str) -> str:
    """Wrapper na system/user prompt."""
    return _call_vllm_messages([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


# =============================================================================
# WĘZEŁ 1: RETRIEVAL (Hybrydowy RAG z Adaptacyjnym Rozmiarem)
# =============================================================================
def retrieve_context(state: FoundryState, session: Session) -> dict:
    """
    Phase 1: Run hybrid search.
    Zwraca ustrukturyzowany XML jako ochronę przed halucynacją.
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    conversation_history = state.get("conversation_history", [])

    is_retry = retry_count > 0
    top_k = 12 if is_retry else 8

    # Adaptive token-based resizing
    history_text = " ".join([m["content"] for m in conversation_history])
    history_tokens = _count_tokens(history_text)
    
    if history_tokens >= _MAX_CONTEXT_TOKENS:
        top_k = 3
    elif _MAX_CONTEXT_TOKENS - history_tokens < 500:
        top_k = max(3, top_k // 2)

    try:
        vector_weight = settings.hybrid_vector_weight
        bm25_weight = settings.hybrid_bm25_weight
        question_type, _difficulty = classify_question(question)
        if settings.adaptive_hybrid_weights:
            if question_type in {"scope", "compliance"}:
                bm25_weight = min(1.0, bm25_weight + settings.adaptive_weight_scope_bonus)
            elif question_type in {"comparative", "process"}:
                vector_weight = min(1.0, vector_weight + settings.adaptive_weight_comparative_bonus)

        # Normalizacja do sumy = 1.0 (ochrona przed przekroczeniem wag)
        vector_weight, bm25_weight = _normalize_weights(vector_weight, bm25_weight)

        query_embedding, emb_cost = _embed_query(question)
        chunks = repo.hybrid_search(
            session,
            query_embedding=query_embedding,
            query_text=question,
            top_k=top_k,
            diversify_by_section=is_retry,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        
        # Wprowadzenie XML Tagging dla silniejszego trzymania kontekstu
        context_texts = []
        for idx, c in enumerate(chunks, 1):
            source_id = str(c.id)[:8]
            context_texts.append(f"<document index=\"{idx}\" ref=\"{source_id}\">\n{c.content.strip()}\n</document>")
            
        context_ids = [str(c.id) for c in chunks]
        logger.debug(
            "[RAG] Pobrano %d fragmentów. typ=%s vec_w=%.2f bm25_w=%.2f koszt_embed=%.4f¢",
            len(chunks),
            question_type,
            vector_weight,
            bm25_weight,
            emb_cost,
        )
        
    except Exception as exc:
        logger.error(f"RAG retrieval failed: {exc}", exc_info=True)
        # Fallback do bazowego chunka
        context_texts = [f"<document index=\"0\" ref=\"base\">\n{state['chunk']['content']}\n</document>"]
        context_ids = [state["chunk"]["id"]]

    return {"retrieved_context": context_texts, "retrieved_ids": context_ids}


# =============================================================================
# WĘZEŁ 2: GENERATION (Ekspert Odpowiada)
# =============================================================================
def generate_answer(state: FoundryState) -> dict:
    """
    Phase 2: Generate grounded answer using Ollama (primary) or OpenAI (fallback).
    Implementuje ścisłą kontrolę tokenów dla uniknięcia błędów przepełnienia.
    """
    question = state["question"]
    is_adversarial = state.get("is_adversarial", False)
    chunk = state["chunk"]
    retrieved = state.get("retrieved_context", [])
    conversation_history = state.get("conversation_history", [])
    turn_count = state.get("turn_count", 0)
    perspective = state.get("perspective", "cfo")

    system_prompt = _EXPERT_SYSTEM_BY_PERSPECTIVE.get(
        perspective, _EXPERT_SYSTEM_BY_PERSPECTIVE["cfo"]
    )

    # Złożenie bloków w formacie XML
    context_parts = [f"<document index=\"0\" ref=\"główny_wątek\">\n{chunk['content']}\n</document>"]
    # Zwiększony limit referencji (6) dzięki lepszemu parsowaniu XML
    for ctx in retrieved[:6]:   
        if ctx not in context_parts:
            context_parts.append(ctx)
            
    context_block = "\n\n".join(context_parts)

    # Dynamic truncation by Tokens
    history_text = " ".join([m["content"] for m in conversation_history])
    history_tokens = _count_tokens(history_text)
    available_tokens = _MAX_CONTEXT_TOKENS - history_tokens
    
    context_tokens = _count_tokens(context_block)
    if context_tokens > available_tokens:
        logger.debug(f"Przycinam tokeny kontekstu: z {context_tokens} na {available_tokens} (Historia zajmuje {history_tokens}t)")
        # Przycinamy tekst, orientacyjnie 1 token = ~4 znaki
        safe_chars = max(500, available_tokens * 4) 
        context_block = context_block[:safe_chars] + "\n...[ucięto ze względu na limity]"

    adversarial_hint = (
        "\n\nUWAGA SYSTEMOWA: To pytanie może celowo wykraczać poza zakres dostarczonych dokumentów. "
        "Jeśli w podanych fragmentach nie ma twardych dowodów na odpowiedź, musisz odpowiedzieć "
        "DOKŁADNIE w ten sposób: \"Brak danych w dyrektywie.\""
        if is_adversarial else ""
    )

    try:
        if turn_count > 0 and conversation_history:
            system_with_ctx = (
                f"{system_prompt}\n\nBAZA WIEDZY (obowiązuje przez całą rozmowę):\n"
                f"{context_block}"
            )
            messages = (
                [{"role": "system", "content": system_with_ctx}]
                + list(conversation_history)
                + [{"role": "user", "content": f"PYTANIE: {question}{adversarial_hint}\n\nTwoja analiza i odpowiedź:"}]
            )
            answer = _call_vllm_messages(messages)
        else:
            user_prompt = (
                f"BAZA WIEDZY:\n{context_block}"
                f"{adversarial_hint}"
                f"\n\nPYTANIE: {question}"
                f"\n\nTwoja analiza i odpowiedź:"
            )
            answer = _call_vllm(system_prompt, user_prompt)
            
    except Exception as exc:
        logger.error(f"Ekspert Generacji zawiódł kaskadowo: {exc}")
        answer = _REFUSAL_PHRASE

    logger.debug(f"Expert [{perspective.upper()}] wygenerował odpowiedź ({len(answer)} znaków).")
    return {"answer": answer}
