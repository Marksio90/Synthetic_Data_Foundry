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

# System prompts
_NORMAL_SYSTEM = (
    "Jesteś konsultantem ESG. Przeczytaj poniższy fragment dyrektywy UE i zadaj "
    "jedno konkretne, realistyczne pytanie biznesowe, które mógłby zadać CFO lub "
    "dyrektor ds. zrównoważonego rozwoju. Pytanie musi dotyczyć wyłącznie treści "
    "tego fragmentu. Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
)

_ADVERSARIAL_SYSTEM = (
    "Jesteś konsultantem ESG próbującym wychwycić luki w systemie AI. "
    "Przeczytaj poniższy fragment dyrektywy UE i zadaj pytanie, na które NIE MA "
    "odpowiedzi w tym fragmencie — pytaj o coś, co wykracza poza jego zakres. "
    "Pytanie powinno brzmieć realistycznie i profesjonalnie. "
    "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
)


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


def simulate_question(state: FoundryState) -> dict:
    """
    LangGraph node.
    Reads state["chunk"] → returns {"question": ..., "is_adversarial": ...}
    """
    chunk = state["chunk"]
    is_adversarial = random.random() < settings.adversarial_ratio

    system = _ADVERSARIAL_SYSTEM if is_adversarial else _NORMAL_SYSTEM
    prompt = (
        f"Fragment dyrektywy:\n\n{chunk['content']}\n\n"
        f"Sekcja: {chunk.get('section_heading', 'N/A')}"
    )

    try:
        question = _call_vllm(system, prompt)
    except Exception as exc:
        logger.error("Simulator vLLM call failed: %s", exc)
        # Generate a safe fallback question so the pipeline doesn't stall
        question = f"Jakie są główne wymogi określone w tej sekcji dyrektywy?"
        is_adversarial = False

    logger.debug(
        "Simulator → %s question: %s", "ADVERSARIAL" if is_adversarial else "NORMAL", question[:80]
    )
    return {"question": question, "is_adversarial": is_adversarial}
