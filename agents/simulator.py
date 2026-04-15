"""
agents/simulator.py — Agent Symulator (Generator Pytań)

LangGraph node: simulate_question(state) → partial state update

Adversarial prompting (Self-Check spec):
  - 90% of the time: generate a realistic, answerable business question
    grounded in the chunk text.
  - 10% of the time: generate a "trick" question about something NOT present
    in the text.  The Expert agent must then respond with the refusal phrase.

Provider routing (priority order):
  1. Ollama (LOCAL, darmowy) — gdy OLLAMA_MODEL ustawiony i serwer dostępny
  2. LLaMA API / Groq (CLOUD, tani) — gdy GROQ_API_KEY ustawiony
  3. OpenAI / vLLM (FALLBACK) — zawsze dostępny przez OPENAI_API_KEY
"""

from __future__ import annotations

import logging
import random

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

# Follow-up prompts — deepen/clarify based on conversation history
_FOLLOWUP_PROMPTS: dict[str, str] = {
    "cfo": (
        "Jesteś CFO dużej spółki notowanej na giełdzie, prowadzącym rozmowę z ekspertem ESG.\n\n"
        "Na podstawie dotychczasowej rozmowy (pytania i odpowiedzi), zadaj jedno pogłębiające "
        "pytanie uzupełniające, które:\n"
        "- Doprecyzowuje lub rozwija kwestię poruszoną w ostatniej odpowiedzi ekserta\n"
        "- Dotyczy praktycznych implikacji finansowych lub operacyjnych dla spółki\n"
        "- Jest logiczną kontynuacją dialogu\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań. Pytaj o szczegóły, wyjątki lub konsekwencje "
        "tego, co zostało już wyjaśnione.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "prawnik": (
        "Jesteś radcą prawnym specjalizującym się w prawie korporacyjnym UE, "
        "prowadzącym rozmowę z ekspertem ESG.\n\n"
        "Na podstawie dotychczasowej rozmowy, zadaj jedno pogłębiające pytanie interpretacyjne, "
        "które:\n"
        "- Analizuje wyjątki, definicje lub odesłania wspomniane w ostatniej odpowiedzi\n"
        "- Doprecyzowuje zakres podmiotowy lub przedmiotowy omawianych przepisów\n"
        "- Jest logiczną kontynuacją dialogu prawniczego\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań. Skoncentruj się na niuansach prawnych.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "audytor": (
        "Jesteś biegłym rewidentem przeprowadzającym audit ESG, "
        "prowadzącym rozmowę z ekspertem ESG.\n\n"
        "Na podstawie dotychczasowej rozmowy, zadaj jedno pogłębiające pytanie weryfikacyjne, "
        "które:\n"
        "- Dotyczy dokumentacji, dowodów lub metod pomiaru wspomnianych w ostatniej odpowiedzi\n"
        "- Precyzuje kryteria zgodności lub wskaźniki ESG\n"
        "- Jest logiczną kontynuacją dialogu audytorskiego\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań. Skupiaj się na weryfikowalności twierdzeń.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
}


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


def _make_ollama_client() -> openai.OpenAI:
    """Klient Ollama przez OpenAI-compatible API."""
    base = settings.ollama_url.rstrip("/")
    return openai.OpenAI(api_key="ollama", base_url=f"{base}/v1", max_retries=0)


def _call_provider(system_prompt: str, user_text: str, max_tokens: int = 256) -> str:
    """
    Routing 3-poziomowy:
      1. Ollama LOCAL  (darmowy, brak limitu)
      2. LLaMA API     (Groq/Together — tani cloud)
      3. OpenAI/vLLM   (fallback — zawsze działa)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    # ── 1. Ollama LOCAL ──────────────────────────────────────────────
    if settings.ollama_model:
        try:
            client = _make_ollama_client()
            resp = client.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.vllm_temperature,
                max_tokens=max_tokens,
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
            max_tokens=max_tokens,
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
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


@_retry_vllm
def _call_vllm(system_prompt: str, user_text: str) -> str:
    """Wrapper z retry dla _call_provider (pytania — krótkie, max 256 tokenów)."""
    return _call_provider(system_prompt, user_text, max_tokens=256)


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
        # Preserve adversarial intent — use a simple fallback question of the right type
        if is_adversarial:
            question = "Jakie sankcje przewiduje ta dyrektywa za naruszenie obowiązków raportowych?"
        else:
            question = "Jakie są główne wymogi określone w tej sekcji dyrektywy?"

    logger.debug(
        "Simulator [%s] → %s: %s",
        perspective, "ADVERSARIAL" if is_adversarial else "NORMAL", question[:80]
    )
    return {"question": question, "is_adversarial": is_adversarial}


def simulate_followup(state: FoundryState) -> dict:
    """
    LangGraph node — generates a follow-up question based on conversation history.
    Reads state["conversation_history"], state["chunk"], state["perspective"] →
    returns {"question": ..., "is_adversarial": False}
    """
    chunk = state["chunk"]
    perspective = state.get("perspective", "cfo")
    conversation_history = state.get("conversation_history", [])

    system = _FOLLOWUP_PROMPTS.get(perspective, _FOLLOWUP_PROMPTS["cfo"])

    # Format conversation history as readable text
    history_lines = []
    for msg in conversation_history:
        role_label = "Pytanie" if msg["role"] == "user" else "Odpowiedź"
        history_lines.append(f"{role_label}: {msg['content']}")
    history_text = "\n\n".join(history_lines)

    prompt = (
        f"Fragment dyrektywy (kontekst):\n\n{chunk['content']}\n\n"
        f"Sekcja: {chunk.get('section_heading', 'N/A')}\n\n"
        f"Dotychczasowa rozmowa:\n\n{history_text}"
    )

    try:
        question = _call_vllm(system, prompt)
    except Exception as exc:
        logger.error("Simulator follow-up vLLM call failed: %s", exc)
        question = "Proszę o doprecyzowanie poprzedniej odpowiedzi."

    logger.debug(
        "Simulator follow-up [%s] turn=%d: %s",
        perspective, state.get("turn_count", 0), question[:80]
    )
    return {"question": question, "is_adversarial": False}
