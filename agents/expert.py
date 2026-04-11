"""
agents/expert.py — Agent Ekspert (Odpowiadacz + RAG)

LangGraph node: expert_answer(state) → partial state update

Two-phase operation:
  Phase 1 (retrieve): hybrid search (vector cosine + BM25 ts_rank) in
    PostgreSQL, filtered WHERE is_superseded = FALSE (Self-Check 3.0).
  Phase 2 (generate): prompt Llama 3 (local vLLM) with retrieved context
    + original chunk.  If question is adversarial, the grounding prompt
    forces the refusal phrase "Brak danych w dyrektywie".

Groq routing (primary when GROQ_API_KEY is set):
  Uses Llama 3.3 70B via Groq API for answer generation.
  Falls back to VLLM_BASE_URL / OpenAI when GROQ_API_KEY is empty.
"""

from __future__ import annotations

import logging

import openai
from sqlalchemy.orm import Session
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from db import repository as repo
from pipeline.state import FoundryState

logger = logging.getLogger(__name__)

_REFUSAL_PHRASE = "Brak danych w dyrektywie."

_EXPERT_SYSTEM = """Jesteś ekspertem ds. ESG i prawa korporacyjnego UE.

ZASADY ODPOWIEDZI:
1. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów dyrektyw podanych w sekcji KONTEKST.
2. Nie wolno Ci podawać informacji spoza dostarczonego kontekstu.
3. Jeśli pytanie dotyczy czegoś, czego nie ma w kontekście, odpowiedz dokładnie:
   "Brak danych w dyrektywie."
4. Cytuj artykuły i ustępy, jeśli są podane w tekście.
5. Odpowiadaj po polsku, w sposób profesjonalny i precyzyjny.
"""


# ---------------------------------------------------------------------------
# Retry decorator for Groq/vLLM calls (handles 429 and 5xx)
# ---------------------------------------------------------------------------

def _is_retryable_vllm(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _retry_vllm(func):
    return retry(
        reraise=True,
        retry=retry_if_exception(_is_retryable_vllm),
        wait=wait_exponential(
            multiplier=1,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


@_retry_vllm
def _call_vllm(system: str, user: str) -> str:
    """Prefer Groq (Llama 3.3) when GROQ_API_KEY is set, fall back to VLLM/OpenAI."""
    if settings.groq_api_key:
        client = openai.OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        model = settings.groq_model
    else:
        is_openai = "openai.com" in settings.vllm_base_url
        client = openai.OpenAI(
            api_key=settings.openai_api_key if is_openai else (settings.vllm_api_key or "not-needed"),
            base_url=None if is_openai else settings.vllm_base_url,
        )
        model = settings.vllm_model
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=settings.vllm_max_tokens,
    )
    return response.choices[0].message.content.strip()


@_retry_vllm
def _call_vllm_messages(messages: list[dict]) -> str:
    """Same Groq/vLLM routing but accepts full messages list (for multi-turn context)."""
    if settings.groq_api_key:
        client = openai.OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
        model = settings.groq_model
    else:
        is_openai = "openai.com" in settings.vllm_base_url
        client = openai.OpenAI(
            api_key=settings.openai_api_key if is_openai else (settings.vllm_api_key or "not-needed"),
            base_url=None if is_openai else settings.vllm_base_url,
        )
        model = settings.vllm_model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=settings.vllm_max_tokens,
    )
    return response.choices[0].message.content.strip()


def _embed_query(query: str) -> list[float]:
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
            top_k=5,
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
    Uses conversation history for follow-up turns (turn_count > 0).
    Returns {"answer": ...}.
    """
    question = state["question"]
    is_adversarial = state.get("is_adversarial", False)
    chunk = state["chunk"]
    retrieved = state.get("retrieved_context", [])
    conversation_history = state.get("conversation_history", [])
    turn_count = state.get("turn_count", 0)

    # Build context block
    context_parts = [f"[Główny fragment]\n{chunk['content']}"]
    for i, ctx in enumerate(retrieved[:4], start=1):
        context_parts.append(f"[Powiązany fragment {i}]\n{ctx}")
    context_block = "\n\n---\n\n".join(context_parts)

    adversarial_hint = ""
    if is_adversarial:
        adversarial_hint = (
            "\n\nUWAGA: To pytanie może wykraczać poza zakres dostarczonych fragmentów. "
            "Jeśli tak jest, musisz odpowiedzieć dokładnie: \"Brak danych w dyrektywie.\""
        )

    try:
        if turn_count > 0 and conversation_history:
            # Multi-turn: build full messages list with history for context
            system_msg = {"role": "system", "content": _EXPERT_SYSTEM}
            context_user_msg = {
                "role": "user",
                "content": f"KONTEKST:\n{context_block}{adversarial_hint}",
            }
            context_assistant_msg = {
                "role": "assistant",
                "content": "Rozumiem. Jestem gotowy odpowiadać na pytania na podstawie dostarczonego kontekstu.",
            }
            # Build: system + context exchange + conversation history + new question
            messages = (
                [system_msg, context_user_msg, context_assistant_msg]
                + list(conversation_history)
                + [{"role": "user", "content": f"PYTANIE: {question}\n\nODPOWIEDŹ:"}]
            )
            answer = _call_vllm_messages(messages)
        else:
            user_prompt = (
                f"KONTEKST:\n{context_block}"
                f"{adversarial_hint}"
                f"\n\nPYTANIE: {question}"
                f"\n\nODPOWIEDŹ:"
            )
            answer = _call_vllm(_EXPERT_SYSTEM, user_prompt)
    except Exception as exc:
        logger.error("Expert vLLM generation failed: %s", exc)
        answer = _REFUSAL_PHRASE

    # Sanity check: adversarial questions that slip through should be caught
    # here — if the model hallucinated something, the Judge will catch it.
    logger.debug("Expert answer (%d chars): %s", len(answer), answer[:80])
    return {"answer": answer}
