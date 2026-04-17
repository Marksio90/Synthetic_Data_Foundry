"""
agents/constitutional.py — Constitutional AI self-critique & revision.

Implementuje wzorzec Constitutional AI (Anthropic, 2022):
  1. Critique:  LLM ocenia własną odpowiedź względem zestawu zasad (Konstytucja).
  2. Revision:  LLM poprawia odpowiedź na podstawie krytyki.

Korzyści:
  - Drastyczna poprawa jakości bez ludzkich etykiet
  - Redukcja halucynacji przez self-correction
  - DARMOWE pary DPO: oryginalna odpowiedź (rejected) vs poprawiona (chosen)
  - Używa tego samego providera co generacja (Cerebras FREE) — zero dodatkowego kosztu

Integracja z pipeline:
  generate_answer → constitutional_revision → judge_answer

Gdy rewizja jest aktywna:
  - state["answer"]          = poprawiona odpowiedź
  - state["rejected_answer"] = oryginalna odpowiedź (do pary DPO/ORPO/KTO)
  - state["constitutional_critique"] = treść krytyki (do analizy)
"""

from __future__ import annotations

import logging

from agents.expert import _call_vllm
from config.settings import settings
from pipeline.state import FoundryState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstytucja — zasady jakości odpowiedzi ESG
# ---------------------------------------------------------------------------

_CONSTITUTION = """
ZASADY ODPOWIEDZI ESG (Konstytucja AI — wersja 2.0):

1. UGRUNTOWANIE: Każde twierdzenie MUSI wynikać z podanego kontekstu dyrektywy.
   Naruszenie: dodanie faktów, liczb, dat lub definicji spoza kontekstu.

2. CYTOWANIE: Każde twierdzenie prawne musi być poparte numerem artykułu/ustępu.
   Naruszenie: ogólne stwierdzenia bez wskazania podstawy prawnej.

3. KOMPLETNOŚĆ: Odpowiedź musi adresować wszystkie aspekty pytania.
   Naruszenie: pominięcie kluczowej części pytania bez uzasadnienia.

4. PRECYZJA PRAWNA: Używaj dokładnych terminów z dyrektywy, nie parafrazuj prawa.
   Naruszenie: używanie własnych interpretacji zamiast terminologii dyrektywy.

5. OSTROŻNOŚĆ EPISTEMICZNA: Gdy kontekst jest niejednoznaczny — zaznacz to.
   Naruszenie: kategoryczne twierdzenia przy niepewnych danych.

6. FORMAT I JĘZYK: Odpowiedź po polsku, profesjonalnym językiem, 2-8 zdań.
   Naruszenie: zbyt krótka (<2 zdania) lub zbyt długa (>8 zdań) odpowiedź.

7. ODMOWA: Pytania poza kontekstem → DOKŁADNIE: "Brak danych w dyrektywie."
   Naruszenie: próba odpowiedzi na pytanie spoza zakresu kontekstu.
"""

_CRITIQUE_SYSTEM = (
    "Jesteś audytorem jakości systemu AI do analizy dyrektyw UE.\n\n"
    "Twoje zadanie: ocenić odpowiedź AI zgodnie z poniższą KONSTYTUCJĄ.\n\n"
    + _CONSTITUTION
    + "\n\nZidentyfikuj KONKRETNE naruszenia zasad (jeśli istnieją). "
    "Bądź precyzyjny: wskaż której zasady dotyczy naruszenie i na czym polega.\n"
    "Odpowiedź w 1-3 zdaniach po polsku.\n"
    "Jeśli odpowiedź jest w pełni zgodna z Konstytucją, napisz DOKŁADNIE: 'Brak naruszeń.'"
)

_REVISION_SYSTEM = (
    "Jesteś ekspertem ESG poprawiającym odpowiedź AI na podstawie audytu jakości.\n\n"
    "KONSTYTUCJA:\n" + _CONSTITUTION
    + "\n\nZasady rewizji:\n"
    "- Zachowaj merytoryczną treść, która jest poprawna\n"
    "- Usuń lub popraw elementy wskazane w krytyce\n"
    "- Dodaj numery artykułów tam gdzie ich brakuje\n"
    "- Odpowiedź po polsku, 2-8 zdań, profesjonalnym językiem\n"
    "- NIE dodawaj wiedzy spoza dostarczonego kontekstu\n\n"
    "Napisz WYŁĄCZNIE poprawioną odpowiedź — bez komentarzy meta."
)


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def constitutional_revision(state: FoundryState) -> dict:
    """
    LangGraph node: Constitutional AI self-critique + revision.

    Wejście:  state["answer"], state["question"], state["retrieved_context"]
    Wyjście:  {"answer": revised, "constitutional_critique": critique,
               "rejected_answer": original}  ← darmowa para DPO

    Jeśli rewizja się nie powiedzie lub nie wniesie poprawy — zwraca oryginał.
    Aktywacja: settings.use_constitutional_ai = True
    """
    if not settings.use_constitutional_ai:
        return {}

    answer = state.get("answer", "")
    question = state.get("question", "")
    is_adversarial = state.get("is_adversarial", False)

    # Nie rewizuj odmów — są poprawne z definicji
    if is_adversarial or not answer or len(answer.strip()) < 30:
        return {}

    # Zbuduj kontekst do krytyki (ogranicz do 4000 znaków)
    context_parts = state.get("retrieved_context", [state["chunk"]["content"]])
    context = "\n---\n".join(context_parts[:4])[:4000]

    # ── Step 1: Critique ──────────────────────────────────────────────────────
    critique_prompt = (
        f"KONTEKST DYREKTYWY:\n{context}\n\n"
        f"PYTANIE: {question}\n\n"
        f"ODPOWIEDŹ DO AUDYTU:\n{answer}\n\n"
        "Oceń odpowiedź zgodnie z Konstytucją:"
    )

    try:
        critique = _call_vllm(_CRITIQUE_SYSTEM, critique_prompt)
    except Exception as exc:
        logger.debug("Constitutional critique failed: %s", exc)
        return {}

    # Brak naruszeń — nie ma co poprawiać
    if "brak naruszeń" in critique.lower():
        logger.debug("Constitutional AI: ✓ no violations (chunk=%s)", state["chunk"]["id"][:8])
        return {"constitutional_critique": critique}

    logger.debug(
        "Constitutional AI: violations found — revising (chunk=%s critique=%s…)",
        state["chunk"]["id"][:8],
        critique[:80],
    )

    # ── Step 2: Revision ──────────────────────────────────────────────────────
    revision_prompt = (
        f"KONTEKST DYREKTYWY:\n{context}\n\n"
        f"PYTANIE: {question}\n\n"
        f"ORYGINALNA ODPOWIEDŹ (z naruszeniami):\n{answer}\n\n"
        f"WYNIKI AUDYTU:\n{critique}\n\n"
        "POPRAWIONA ODPOWIEDŹ:"
    )

    try:
        revised = _call_vllm(_REVISION_SYSTEM, revision_prompt)
    except Exception as exc:
        logger.debug("Constitutional revision failed: %s", exc)
        return {"constitutional_critique": critique}

    # Sanity check: rewizja musi być sensowna
    if len(revised.strip()) < 20 or revised.strip() == answer.strip():
        logger.debug("Constitutional AI: revision identical or too short — keeping original")
        return {"constitutional_critique": critique}

    logger.info(
        "Constitutional AI: ✓ revised (chunk=%s len=%d→%d)",
        state["chunk"]["id"][:8],
        len(answer),
        len(revised),
    )

    return {
        "answer": revised,
        "constitutional_critique": critique,
        # Oryginalna odpowiedź staje się "rejected" w parze DPO/ORPO/KTO
        "rejected_answer": state.get("rejected_answer") or answer,
    }
