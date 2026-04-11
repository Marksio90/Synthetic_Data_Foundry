"""
agents/simulator.py — Agent Symulator (Generator Pytań)

LangGraph node: simulate_question(state) → partial state update

Adversarial prompting (Self-Check spec):
  - 90% of the time: generate a realistic, answerable business question
    grounded in the chunk text.
  - 10% of the time: generate a "trick" question about something NOT present
    in the text.  The Expert agent must then respond with the refusal phrase.
"""

from __future__ import annotations

import logging
import random

import openai

from config.settings import settings
from pipeline.state import FoundryState

logger = logging.getLogger(__name__)

# System prompts — trzy perspektywy + adversarial dla każdej
_NORMAL_PROMPTS: dict[str, str] = {
    "cfo": (
        "Jesteś CFO dużej spółki notowanej na giełdzie.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj, co KONKRETNIE zawiera ten fragment: "
        "jakie obowiązki raportowe, jakie progi lub zakresy podmiotowe, "
        "jakie terminy, jakie wymogi ujawnień finansowych lub definicje.\n"
        "Krok 3: Zadaj jedno pytanie dotyczące TEGO, co jest wprost opisane "
        "w fragmencie — np. kto jest objęty obowiązkiem, co trzeba ujawnić, "
        "w jakim terminie, jaki próg kwalifikuje podmiot.\n\n"
        "WAŻNE: Nie pytaj o koszty wdrożenia ani prognozy finansowe "
        "— dyrektywy nie zawierają szacunków kosztów.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "prawnik": (
        "Jesteś radcą prawnym specjalizującym się w prawie korporacyjnym UE.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy.\n"
        "Krok 2: Zidentyfikuj, co KONKRETNIE zawiera ten fragment: "
        "definicje prawne, zakres podmiotowy lub przedmiotowy obowiązku, "
        "wyjątki, procedury, terminy lub odesłania do innych przepisów.\n"
        "Krok 3: Zadaj jedno pytanie interpretacyjne dotyczące TEGO, "
        "co jest wprost opisane w fragmencie — np. jak definiuje się dany termin, "
        "kto jest wyłączony z obowiązku, jaki jest zakres stosowania.\n\n"
        "WAŻNE: Nie pytaj o sankcje ani przepisy nieobecne w tym fragmencie.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "audytor": (
        "Jesteś biegłym rewidentem przeprowadzającym audit ESG.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj, co KONKRETNIE zawiera ten fragment: "
        "wymogi ujawnień, kryteria kwalifikacji działalności, "
        "definicje wskaźników lub obowiązki dokumentacyjne.\n"
        "Krok 3: Zadaj jedno pytanie weryfikacyjne dotyczące TEGO, "
        "co jest wprost opisane w fragmencie — np. jakie kryteria muszą być spełnione, "
        "co musi być ujawnione, jak definiuje się dany wskaźnik ESG.\n\n"
        "WAŻNE: Nie pytaj o metody weryfikacji ani dokumenty "
        "nieobecne w tym fragmencie.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
}

_ADVERSARIAL_PROMPTS: dict[str, str] = {
    "cfo": (
        "Jesteś CFO testującym odporność systemu AI na halucynacje. "
        "Przeczytaj fragment dyrektywy i zadaj pytanie o obowiązki finansowe "
        "lub progi liczbowe, których NIE MA w tym fragmencie "
        "(mogą istnieć w innych przepisach, ale nie tutaj). "
        "Pytanie musi brzmieć realistycznie i profesjonalnie. "
        "Odpowiedz tylko pytaniem."
    ),
    "prawnik": (
        "Jesteś prawnikiem testującym odporność systemu AI na halucynacje. "
        "Przeczytaj fragment dyrektywy i zadaj pytanie o konkretny przepis, "
        "wyjątek lub definicję, który NIE ISTNIEJE w tym fragmencie "
        "(może istnieć gdzie indziej, ale nie tutaj). "
        "Pytanie musi brzmieć profesjonalnie i technicznie. "
        "Odpowiedz tylko pytaniem."
    ),
    "audytor": (
        "Jesteś audytorem testującym odporność systemu AI na halucynacje. "
        "Przeczytaj fragment dyrektywy i zadaj pytanie o konkretny wymóg "
        "dokumentacyjny lub metodę weryfikacji, która NIE JEST opisana "
        "w tym fragmencie (może istnieć w innych częściach dyrektywy). "
        "Pytanie musi brzmieć technicznie i wiarygodnie. "
        "Odpowiedz tylko pytaniem."
    ),
}


def _call_vllm(system_prompt: str, user_text: str) -> str:
    """Call question-generation LLM — local vLLM or OpenAI depending on VLLM_BASE_URL."""
    is_openai = "openai.com" in settings.vllm_base_url
    client = openai.OpenAI(
        api_key=settings.openai_api_key if is_openai else (settings.vllm_api_key or "not-needed"),
        base_url=None if is_openai else settings.vllm_base_url,
    )
    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=settings.vllm_temperature,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


_PERSPECTIVES = ["cfo", "prawnik", "audytor"]


def simulate_question(state: FoundryState) -> dict:
    """
    LangGraph node.
    Reads state["chunk"] + state["perspective"] →
    returns {"question": ..., "is_adversarial": ...}
    """
    chunk = state["chunk"]
    perspective = state.get("perspective", "cfo")
    is_adversarial = random.random() < settings.adversarial_ratio

    prompts = _ADVERSARIAL_PROMPTS if is_adversarial else _NORMAL_PROMPTS
    system = prompts.get(perspective, prompts["cfo"])

    prompt = (
        f"Fragment dyrektywy:\n\n{chunk['content']}\n\n"
        f"Sekcja: {chunk.get('section_heading', 'N/A')}"
    )

    try:
        question = _call_vllm(system, prompt)
    except Exception as exc:
        logger.error("Simulator vLLM call failed: %s", exc)
        question = "Jakie są główne wymogi określone w tej sekcji dyrektywy?"
        is_adversarial = False

    logger.debug(
        "Simulator [%s] → %s: %s",
        perspective, "ADVERSARIAL" if is_adversarial else "NORMAL", question[:80]
    )
    return {"question": question, "is_adversarial": is_adversarial}
