"""
agents/expert.py — Agent Ekspert (Odpowiadacz + RAG)

LangGraph node: expert_answer(state) → partial state update

Two-phase operation:
  Phase 1 (retrieve): hybrid search (vector cosine + BM25 ts_rank) in
    PostgreSQL, filtered WHERE is_superseded = FALSE (Self-Check 3.0).
  Phase 2 (generate): prompt LLM with retrieved context + original chunk.
    If question is adversarial, the grounding prompt forces the refusal
    phrase "Brak danych w dyrektywie".

Provider routing (priority order):
  1. Ollama LOCAL  (darmowy, ~4.7 GB RAM dla llama3.1:8b)
  2. LLaMA API     (Groq/Together — tani cloud)
  3. OpenAI / vLLM (fallback — zawsze działa)

Embeddings routing:
  USE_LOCAL_EMBEDDINGS=true  → Ollama nomic-embed-text (darmowy, 0.3 GB RAM)
  USE_LOCAL_EMBEDDINGS=false → OpenAI text-embedding-3-small (domyślny)
"""

from __future__ import annotations

import logging
import re as _re

import openai
from sqlalchemy.orm import Session
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from config.settings import settings
from db import repository as repo
from pipeline.state import FoundryState

logger = logging.getLogger(__name__)

_REFUSAL_PHRASE = "Brak danych w dyrektywie."

# Chain-of-Thought format appended to every perspective prompt.
# Teaches the model to reason before answering; <reasoning> tags are stripped
# by the Judge for quality evaluation but kept in the JSONL for fine-tuning.
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

# Perspective-aware expert system prompts — improve answer precision per role
_EXPERT_SYSTEM_BY_PERSPECTIVE: dict[str, str] = {
    "cfo": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy CFO.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w KONTEKST.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: zakresie podmiotowym, terminach, progach kwalifikacyjnych, wymogach ujawnień.\n"
        "5. Odpowiadaj po polsku, zwięźle i precyzyjnie (2–8 zdań)."
        + _COT_FORMAT
    ),
    "prawnik": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy radcy prawnego.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w KONTEKST.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj podstawę prawną: artykuł, ustęp, punkt.\n"
        "4. Interpretuj literalnie: zakres podmiotowy, przedmiotowy, wyjątki, definicje legalne.\n"
        "5. Odpowiadaj po polsku z precyzją języka prawniczego (2–8 zdań)."
        + _COT_FORMAT
    ),
    "audytor": (
        "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE, odpowiadającym z perspektywy biegłego rewidenta ESG.\n\n"
        "ZASADY:\n"
        "1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w KONTEKST.\n"
        "2. Jeśli pytanie wykracza poza kontekst, odpowiedz DOKŁADNIE: \"Brak danych w dyrektywie.\"\n"
        "3. Cytuj numery artykułów i ustępów podane w tekście.\n"
        "4. Skup się na: wymogach ujawnień, kryteriach kwalifikacji, wskaźnikach ESG, metodach pomiaru.\n"
        "5. Odpowiadaj po polsku, technicznie i precyzyjnie (2–8 zdań)."
        + _COT_FORMAT
    ),
}

# Max context chars — increased for local Ollama (no token cost) and larger context windows.
# Ollama llama3.1:8b supports 128k context; Groq/OpenAI gpt-4o-mini supports 128k.
# We cap at 6000 chars (~1500 tokens) to leave room for question + history + answer.
_MAX_CONTEXT_CHARS = 6000


# ---------------------------------------------------------------------------
# Retry decorator for Groq/vLLM calls (handles 429 and 5xx)
# ---------------------------------------------------------------------------

def _is_tpd_limit(exc: BaseException) -> bool:
    """Zwraca True gdy Groq zwraca dzienny limit TPD (nie minutowy TPM)."""
    return isinstance(exc, openai.RateLimitError) and "tokens per day" in str(exc).lower()


def _is_retryable_vllm(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        # TPD (daily limit) — nie ma sensu retry, poczekać trzeba 20+ minut
        if _is_tpd_limit(exc):
            return False
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _parse_groq_retry_after(exc: BaseException) -> float:
    """
    Parsuje czas oczekiwania z komunikatu Groq 429.
    Obsługuje formaty:
      - 'try again in 8.69s'      → 8.69s
      - 'try again in 21m17.856s' → 1277.856s
    Dodaje 2s buforu.
    """
    body = str(exc)
    # Format minuty + sekundy: "21m17.856s"
    m = _re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", body, _re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 2.0
    # Format same sekundy: "8.69s"
    m = _re.search(r"try again in (\d+(?:\.\d+)?)s", body, _re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0
    return settings.tenacity_initial_wait


def _groq_aware_wait(retry_state) -> float:  # type: ignore[return]
    """
    Respects Groq's 'retry-after' hint from 429 body.
    Falls back to exponential backoff for other errors.
    """
    exc = retry_state.outcome.exception()
    if exc and isinstance(exc, openai.RateLimitError):
        return _parse_groq_retry_after(exc)
    # Exponential for 5xx
    return min(
        settings.tenacity_initial_wait * (2 ** max(0, retry_state.attempt_number - 1)),
        settings.tenacity_max_wait,
    )


def _retry_vllm(func):
    return retry(
        reraise=True,
        retry=retry_if_exception(_is_retryable_vllm),
        wait=_groq_aware_wait,
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


def _make_ollama_client() -> openai.OpenAI:
    """Klient Ollama przez OpenAI-compatible API."""
    base = settings.ollama_url.rstrip("/")
    return openai.OpenAI(api_key="ollama", base_url=f"{base}/v1", max_retries=0)


def _call_provider(messages: list[dict]) -> str:
    """
    Routing 3-poziomowy dla odpowiedzi (długich — max vllm_max_tokens):
      1. Ollama LOCAL  (darmowy, brak limitu)
      2. LLaMA API     (Groq/Together — tani cloud)
      3. OpenAI/vLLM   (fallback — zawsze działa)
    """
    max_tok = settings.vllm_max_tokens

    # ── 1. Ollama LOCAL ──────────────────────────────────────────────
    if settings.ollama_model:
        try:
            client = _make_ollama_client()
            resp = client.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.vllm_temperature,
                max_tokens=max_tok,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Ollama niedostępny (%s), przełączam na LLaMA API / OpenAI", e)

    # ── 2. LLaMA API (Groq / Together) ───────────────────────────────
    if settings.groq_api_key:
        client = openai.OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            max_retries=0,
        )
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=settings.vllm_temperature,
            max_tokens=max_tok,
        )
        return resp.choices[0].message.content.strip()

    # ── 3. OpenAI / vLLM fallback ────────────────────────────────────
    is_openai = "openai.com" in settings.vllm_base_url
    client = openai.OpenAI(
        api_key=settings.openai_api_key if is_openai else (settings.vllm_api_key or "not-needed"),
        base_url=None if is_openai else settings.vllm_base_url,
        max_retries=0,
    )
    resp = client.chat.completions.create(
        model=settings.openai_primary_model if is_openai else settings.vllm_model,
        messages=messages,
        temperature=settings.vllm_temperature,
        max_tokens=max_tok,
    )
    return resp.choices[0].message.content.strip()


@_retry_vllm
def _call_vllm(system: str, user: str) -> str:
    """Wrapper z retry — buduje messages i wywołuje _call_provider."""
    return _call_provider([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


@_retry_vllm
def _call_vllm_messages(messages: list[dict]) -> str:
    """Wrapper z retry dla multi-turn context."""
    return _call_provider(messages)


def _embed_query(query: str) -> list[float]:
    """
    Embeddingi z routingiem:
      USE_LOCAL_EMBEDDINGS=true  → Ollama nomic-embed-text (darmowy, ~1536 dims)
      USE_LOCAL_EMBEDDINGS=false → OpenAI text-embedding-3-small (domyślny)
    """
    if settings.use_local_embeddings and settings.ollama_embed_model:
        try:
            base = settings.ollama_url.rstrip("/")
            client = openai.OpenAI(api_key="ollama", base_url=f"{base}/v1", max_retries=0)
            resp = client.embeddings.create(
                model=settings.ollama_embed_model,
                input=[query],
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.warning("Ollama embed niedostępny (%s), używam OpenAI embeddings", e)

    # OpenAI embeddings (default)
    client = openai.OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=[query],
        dimensions=settings.openai_embedding_dims,
    )
    return resp.data[0].embedding


def retrieve_context(state: FoundryState, session: Session) -> dict:
    """
    Phase 1: Run hybrid search.
    Returns {"retrieved_context": [...], "retrieved_ids": [...]}.
    """
    question = state["question"]
    try:
        query_embedding = _embed_query(question)
        chunks = repo.hybrid_search(
            session,
            query_embedding=query_embedding,
            query_text=question,
            top_k=8,  # zwiększone z 5 → bogatszy kontekst RAG
        )
        context_texts = [c.content for c in chunks]
        context_ids = [str(c.id) for c in chunks]
    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)
        # Fallback: use the chunk itself as context
        context_texts = [state["chunk"]["content"]]
        context_ids = [state["chunk"]["id"]]

    return {"retrieved_context": context_texts, "retrieved_ids": context_ids}


def generate_answer(state: FoundryState) -> dict:
    """
    Phase 2: Generate grounded answer using Groq/vLLM.
    Uses perspective-aware system prompt and conversation history for follow-up turns.
    Returns {"answer": ...}.
    """
    question = state["question"]
    is_adversarial = state.get("is_adversarial", False)
    chunk = state["chunk"]
    retrieved = state.get("retrieved_context", [])
    conversation_history = state.get("conversation_history", [])
    turn_count = state.get("turn_count", 0)
    perspective = state.get("perspective", "cfo")

    # Pick perspective-aware system prompt
    system_prompt = _EXPERT_SYSTEM_BY_PERSPECTIVE.get(
        perspective, _EXPERT_SYSTEM_BY_PERSPECTIVE["cfo"]
    )

    # Build context block — dynamically cap to avoid token overflow
    context_parts = [f"[Główny fragment]\n{chunk['content']}"]
    for i, ctx in enumerate(retrieved[:6], start=1):   # zwiększone z 4 → 6 fragmentów
        context_parts.append(f"[Powiązany fragment {i}]\n{ctx}")
    context_block = "\n\n---\n\n".join(context_parts)

    # Dynamic truncation: leave room for conversation history + question + answer
    history_chars = sum(len(m["content"]) for m in conversation_history)
    available = _MAX_CONTEXT_CHARS - history_chars
    if len(context_block) > available:
        context_block = context_block[:max(available, 500)]
        logger.debug("Context truncated to %d chars (history=%d)", len(context_block), history_chars)

    adversarial_hint = (
        "\n\nUWAGA: To pytanie może wykraczać poza zakres dostarczonych fragmentów. "
        "Jeśli tak jest, musisz odpowiedzieć DOKŁADNIE: \"Brak danych w dyrektywie.\""
        if is_adversarial else ""
    )

    try:
        if turn_count > 0 and conversation_history:
            # Multi-turn: inject context only in system, conversation history in messages
            # Avoids duplicating context in every turn
            system_with_ctx = (
                f"{system_prompt}\n\nKONTEKST DYREKTYWY (obowiązuje przez całą rozmowę):\n"
                f"{context_block}"
            )
            messages = (
                [{"role": "system", "content": system_with_ctx}]
                + list(conversation_history)
                + [{"role": "user", "content": f"PYTANIE: {question}{adversarial_hint}\n\nODPOWIEDŹ:"}]
            )
            answer = _call_vllm_messages(messages)
        else:
            # First turn: simple system + user format
            user_prompt = (
                f"KONTEKST:\n{context_block}"
                f"{adversarial_hint}"
                f"\n\nPYTANIE: {question}"
                f"\n\nODPOWIEDŹ:"
            )
            answer = _call_vllm(system_prompt, user_prompt)
    except Exception as exc:
        logger.error("Expert generation failed: %s", exc)
        answer = _REFUSAL_PHRASE

    logger.debug("Expert [%s] answer (%d chars): %s", perspective, len(answer), answer[:80])
    return {"answer": answer}
