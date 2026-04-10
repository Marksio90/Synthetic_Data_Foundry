"""
agents/expert.py — Agent Ekspert (Odpowiadacz + RAG)

LangGraph node: expert_answer(state) → partial state update

Two-phase operation:
  Phase 1 (retrieve): hybrid search (vector cosine + BM25 ts_rank) in
    PostgreSQL, filtered WHERE is_superseded = FALSE (Self-Check 3.0).
  Phase 2 (generate): prompt Llama 3 (local vLLM) with retrieved context
    + original chunk.  If question is adversarial, the grounding prompt
    forces the refusal phrase "Brak danych w dyrektywie".
"""

from __future__ import annotations

import logging

import openai
from sqlalchemy.orm import Session

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


def _call_vllm(system: str, user: str) -> str:
    client = openai.OpenAI(
        api_key=settings.vllm_api_key or settings.openai_api_key,
        base_url=settings.vllm_base_url,
    )
    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,  # low temperature → factual, grounded
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
    Phase 2: Generate grounded answer using Llama 3.
    Returns {"answer": ...}.
    """
    question = state["question"]
    is_adversarial = state.get("is_adversarial", False)
    chunk = state["chunk"]
    retrieved = state.get("retrieved_context", [])

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

    user_prompt = (
        f"KONTEKST:\n{context_block}"
        f"{adversarial_hint}"
        f"\n\nPYTANIE: {question}"
        f"\n\nODPOWIEDŹ:"
    )

    try:
        answer = _call_vllm(_EXPERT_SYSTEM, user_prompt)
    except Exception as exc:
        logger.error("Expert vLLM generation failed: %s", exc)
        answer = _REFUSAL_PHRASE

    # Sanity check: adversarial questions that slip through should be caught
    # here — if the model hallucinated something, the Judge will catch it.
    logger.debug("Expert answer (%d chars): %s", len(answer), answer[:80])
    return {"answer": answer}
