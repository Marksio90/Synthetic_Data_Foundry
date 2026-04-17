"""
agents/simulator.py — Agent Symulator (Generator Pytań)

LangGraph node: simulate_question(state) → partial state update

8 perspektyw eksperckich (zamiast 3):
  cfo         — CFO spółki giełdowej (finanse, progi, terminy)
  prawnik     — radca prawny (definicje, zakres, wyjątki)
  audytor     — biegły rewident ESG (wymogi, wskaźniki, dokumentacja)
  analityk    — analityk finansowy buy-side (ryzyko, implikacje rynkowe)
  regulator   — urzędnik KNF/ESMA (nadzór, compliance, wdrożenie)
  akademik    — badacz ESG (metodologia, spójność, porównanie z literaturą)
  dziennikarz — dziennikarz finansowy (wyjaśnienie, kontekst, implikacje)
  inwestor    — inwestor instytucjonalny (ESG scoring, alokacja kapitału)

Adversarial prompting (Self-Check):
  - 90% normalnych pytań ugruntowanych w tekście
  - 10% pytań-pułapek o rzeczy NIEOBECNE w tekście → model musi odmówić

Provider routing (priorytet):
  1. Ollama LOCAL (darmowy — qwen2.5:14b)
  2. OpenAI (fallback gdy Ollama niedostępny)
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

# ---------------------------------------------------------------------------
# Prompty systemowe — 8 perspektyw × normalny / adversarial / follow-up
# ---------------------------------------------------------------------------

_NORMAL_PROMPTS: dict[str, str] = {
    "cfo": (
        "Jesteś CFO dużej spółki notowanej na giełdzie.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj konkretne obowiązki finansowe, progi liczbowe, "
        "terminy wdrożenia lub wymogi ujawnień finansowych.\n"
        "Krok 3: Zadaj JEDNO pytanie dotyczące tego, co jest wprost opisane w fragmencie.\n\n"
        "WAŻNE: Nie pytaj o koszty wdrożenia ani prognozy nieobecne w tekście.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "prawnik": (
        "Jesteś radcą prawnym specjalizującym się w prawie korporacyjnym UE.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy.\n"
        "Krok 2: Zidentyfikuj definicje prawne, zakres podmiotowy/przedmiotowy, "
        "wyjątki, procedury lub odesłania do innych przepisów.\n"
        "Krok 3: Zadaj JEDNO pytanie interpretacyjne o to, co jest wprost w fragmencie.\n\n"
        "WAŻNE: Nie pytaj o sankcje ani przepisy nieobecne w tym fragmencie.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "audytor": (
        "Jesteś biegłym rewidentem przeprowadzającym audit ESG.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj wymogi ujawnień, kryteria kwalifikacji działalności, "
        "definicje wskaźników lub obowiązki dokumentacyjne.\n"
        "Krok 3: Zadaj JEDNO pytanie weryfikacyjne dotyczące wymogów z fragmentu.\n\n"
        "WAŻNE: Nie pytaj o metody weryfikacji nieobecne w tekście.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "analityk": (
        "Jesteś analitykiem finansowym buy-side specjalizującym się w ESG.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj wymogi raportowe, zakresy podmiotowe lub kryteria "
        "klasyfikacji istotne dla oceny ryzyka i wyceny spółek.\n"
        "Krok 3: Zadaj JEDNO pytanie analityczne o implikacje z tego fragmentu.\n\n"
        "WAŻNE: Pytaj o fakty z tekstu, nie o prognozy rynkowe.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "regulator": (
        "Jesteś urzędnikiem KNF/ESMA odpowiedzialnym za nadzór nad wdrożeniem dyrektyw ESG.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj obowiązki podmiotów nadzorowanych, terminy, "
        "zakres stosowania lub mechanizmy compliance.\n"
        "Krok 3: Zadaj JEDNO pytanie nadzorcze dotyczące wymogów z fragmentu.\n\n"
        "WAŻNE: Skup się na tym, co reguluje ten konkretny fragment.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "akademik": (
        "Jesteś badaczem ESG publikującym w recenzowanych czasopismach naukowych.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj definicje, metodologię, zakres lub kryteria klasyfikacji "
        "istotne dla badań naukowych nad ESG.\n"
        "Krok 3: Zadaj JEDNO pytanie metodologiczne lub analityczne o treść fragmentu.\n\n"
        "WAŻNE: Pytaj o to, co jest wprost w dyrektywie.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "dziennikarz": (
        "Jesteś dziennikarzem finansowym piszącym dla szerokiego grona biznesowego.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj kto jest objęty przepisami, co musi zrobić i kiedy — "
        "konkretne, praktyczne fakty z tekstu.\n"
        "Krok 3: Zadaj JEDNO proste, konkretne pytanie o fakty z fragmentu.\n\n"
        "WAŻNE: Pytania muszą dotyczyć faktów z tekstu, nie opinii.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
    "inwestor": (
        "Jesteś inwestorem instytucjonalnym (fundusz emerytalny) stosującym strategię ESG.\n\n"
        "Krok 1: Przeczytaj uważnie poniższy fragment dyrektywy UE.\n"
        "Krok 2: Zidentyfikuj wymogi ujawnień, kryteria klasyfikacji lub obowiązki "
        "raportowe wpływające na alokację kapitału i ocenę ESG spółek.\n"
        "Krok 3: Zadaj JEDNO pytanie inwestycyjne o to, co konkretnie wynika z fragmentu.\n\n"
        "WAŻNE: Pytaj o fakty prawne z tekstu, nie o strategie inwestycyjne.\n"
        "Odpowiedz tylko pytaniem — bez wstępu ani komentarza."
    ),
}

_ADVERSARIAL_PROMPTS: dict[str, str] = {
    "cfo": (
        "Jesteś CFO testującym AI na halucynacje. "
        "Przeczytaj fragment i zadaj pytanie o obowiązki finansowe lub progi "
        "których NIE MA w tym fragmencie. Pytanie musi brzmieć realistycznie. "
        "Odpowiedz tylko pytaniem."
    ),
    "prawnik": (
        "Jesteś prawnikiem testującym AI na halucynacje. "
        "Przeczytaj fragment i zadaj pytanie o przepis, wyjątek lub definicję "
        "który NIE ISTNIEJE w tym fragmencie. "
        "Pytanie musi brzmieć profesjonalnie. Odpowiedz tylko pytaniem."
    ),
    "audytor": (
        "Jesteś audytorem testującym AI na halucynacje. "
        "Przeczytaj fragment i zadaj pytanie o wymóg dokumentacyjny lub metodę "
        "która NIE JEST opisana w tym fragmencie. "
        "Pytanie musi brzmieć technicznie. Odpowiedz tylko pytaniem."
    ),
    "analityk": (
        "Jesteś analitykiem testującym AI na halucynacje. "
        "Zapytaj o konkretny wskaźnik finansowy lub próg liczbowy "
        "który NIE POJAWIA SIĘ w tym fragmencie. "
        "Pytanie musi brzmieć analitycznie. Odpowiedz tylko pytaniem."
    ),
    "regulator": (
        "Jesteś regulatorem testującym AI na halucynacje. "
        "Zapytaj o konkretny mechanizm nadzorczy lub procedurę "
        "która NIE JEST opisana w tym fragmencie. "
        "Pytanie musi brzmieć oficjalnie. Odpowiedz tylko pytaniem."
    ),
    "akademik": (
        "Jesteś badaczem testującym AI na halucynacje. "
        "Zapytaj o konkretną metodologię lub kryterium klasyfikacji "
        "które NIE POJAWIA SIĘ w tym fragmencie. "
        "Pytanie musi brzmieć naukowo. Odpowiedz tylko pytaniem."
    ),
    "dziennikarz": (
        "Jesteś dziennikarzem testującym AI na halucynacje. "
        "Zapytaj o konkretny fakt, datę lub wymóg "
        "którego NIE MA w tym fragmencie. "
        "Pytanie musi brzmieć naturalnie. Odpowiedz tylko pytaniem."
    ),
    "inwestor": (
        "Jesteś inwestorem testującym AI na halucynacje. "
        "Zapytaj o konkretny wymóg ujawnień lub kryterium ESG "
        "który NIE POJAWIA SIĘ w tym fragmencie. "
        "Pytanie musi brzmieć inwestycyjnie. Odpowiedz tylko pytaniem."
    ),
}

_FOLLOWUP_PROMPTS: dict[str, str] = {
    "cfo": (
        "Jesteś CFO prowadzącym rozmowę z ekspertem ESG.\n\n"
        "Na podstawie dotychczasowej rozmowy zadaj JEDNO pogłębiające pytanie o:\n"
        "- Praktyczne implikacje finansowe lub operacyjne\n"
        "- Szczegóły z ostatniej odpowiedzi\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
    "prawnik": (
        "Jesteś radcą prawnym prowadzącym rozmowę z ekspertem ESG.\n\n"
        "Na podstawie dotychczasowej rozmowy zadaj JEDNO pogłębiające pytanie o:\n"
        "- Wyjątki, definicje lub odesłania z ostatniej odpowiedzi\n"
        "- Zakres podmiotowy lub przedmiotowy\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
    "audytor": (
        "Jesteś biegłym rewidentem ESG prowadzącym rozmowę z ekspertem.\n\n"
        "Na podstawie dotychczasowej rozmowy zadaj JEDNO pogłębiające pytanie o:\n"
        "- Dokumentację, dowody lub metody pomiaru\n"
        "- Kryteria zgodności lub wskaźniki ESG\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
    "analityk": (
        "Jesteś analitykiem ESG prowadzącym rozmowę z ekspertem.\n\n"
        "Na podstawie rozmowy zadaj JEDNO pogłębiające pytanie o:\n"
        "- Implikacje dla oceny ryzyka ESG lub wyceny\n"
        "- Szczegóły metodologiczne wymogów raportowych\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
    "regulator": (
        "Jesteś urzędnikiem regulacyjnym prowadzącym rozmowę z ekspertem.\n\n"
        "Na podstawie rozmowy zadaj JEDNO pogłębiające pytanie o:\n"
        "- Szczegóły mechanizmów nadzorczych lub procedur\n"
        "- Harmonogram wdrożenia i etapy\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
    "akademik": (
        "Jesteś badaczem ESG prowadzącym rozmowę z ekspertem.\n\n"
        "Na podstawie rozmowy zadaj JEDNO pogłębiające pytanie o:\n"
        "- Metodologię klasyfikacji lub kryteria jakościowe\n"
        "- Spójność definicji z innymi regulacjami UE\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
    "dziennikarz": (
        "Jesteś dziennikarzem finansowym prowadzącym rozmowę z ekspertem.\n\n"
        "Na podstawie rozmowy zadaj JEDNO proste pogłębiające pytanie o:\n"
        "- Praktyczne konsekwencje dla firm lub inwestorów\n"
        "- Kto konkretnie jest objęty wymogiem i od kiedy\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
    "inwestor": (
        "Jesteś inwestorem instytucjonalnym prowadzącym rozmowę z ekspertem.\n\n"
        "Na podstawie rozmowy zadaj JEDNO pogłębiające pytanie o:\n"
        "- Wpływ na ESG scoring i alokację kapitału\n"
        "- Terminy wdrożenia ważne dla due diligence\n\n"
        "WAŻNE: Nie powtarzaj poprzednich pytań.\n"
        "Odpowiedz tylko pytaniem."
    ),
}

# Wszystkie 8 perspektyw — eksportowane do main.py
ALL_PERSPECTIVES = list(_NORMAL_PROMPTS.keys())


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def _is_retryable_openai(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _retry_api(func):
    return retry(
        reraise=True,
        retry=retry_if_exception(_is_retryable_openai),
        wait=wait_exponential(
            multiplier=1,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


def _make_ollama_client() -> openai.OpenAI:
    base = settings.ollama_url.rstrip("/")
    return openai.OpenAI(api_key="ollama", base_url=f"{base}/v1", max_retries=0)


def _call_provider(system_prompt: str, user_text: str, max_tokens: int = 256) -> str:
    """2-poziomowy routing: Ollama → OpenAI."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    if settings.ollama_model:
        try:
            client = _make_ollama_client()
            resp = client.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.generation_temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Ollama niedostępny (%s), przełączam na OpenAI", e)

    client = openai.OpenAI(api_key=settings.openai_api_key, max_retries=0)
    resp = client.chat.completions.create(
        model=settings.openai_primary_model,
        messages=messages,
        temperature=settings.generation_temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


@_retry_api
def _call_vllm(system_prompt: str, user_text: str) -> str:
    return _call_provider(system_prompt, user_text, max_tokens=256)


# ---------------------------------------------------------------------------
# Similarity check (duplicate follow-up prevention)
# ---------------------------------------------------------------------------

def _jaccard_similarity(a: str, b: str) -> float:
    def bigrams(text: str) -> set[str]:
        words = text.lower().split()
        return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)} if len(words) >= 2 else {text.lower()}

    set_a, set_b = bigrams(a), bigrams(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Public nodes
# ---------------------------------------------------------------------------

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
        question = (
            "Jakie sankcje przewiduje ta dyrektywa za naruszenie obowiązków raportowych?"
            if is_adversarial
            else "Jakie są główne wymogi określone w tej sekcji dyrektywy?"
        )

    logger.debug(
        "Simulator [%s/%s] → %s: %s",
        perspective,
        state["chunk"]["id"][:6],
        "ADV" if is_adversarial else "NRM",
        question[:80],
    )
    return {"question": question, "is_adversarial": is_adversarial}


def simulate_followup(state: FoundryState) -> dict:
    """
    LangGraph node — generuje pytanie uzupełniające.
    Sprawdza podobieństwo Jaccard (≥0.75) — zapobiega powtórkom.
    """
    chunk = state["chunk"]
    perspective = state.get("perspective", "cfo")
    conversation_history = state.get("conversation_history", [])

    system = _FOLLOWUP_PROMPTS.get(perspective, _FOLLOWUP_PROMPTS["cfo"])

    history_lines = []
    for msg in conversation_history:
        label = "Pytanie" if msg["role"] == "user" else "Odpowiedź"
        history_lines.append(f"{label}: {msg['content']}")

    prompt = (
        f"Fragment dyrektywy (kontekst):\n\n{chunk['content']}\n\n"
        f"Sekcja: {chunk.get('section_heading', 'N/A')}\n\n"
        "Dotychczasowa rozmowa:\n\n"
        + "\n\n".join(history_lines)
    )

    prev_questions = [m["content"] for m in conversation_history if m["role"] == "user"]

    try:
        question = _call_vllm(system, prompt)
    except Exception as exc:
        logger.error("Simulator follow-up call failed: %s", exc)
        question = "Proszę o doprecyzowanie poprzedniej odpowiedzi."
    else:
        for prev in prev_questions:
            if _jaccard_similarity(question, prev) >= 0.75:
                logger.debug("Follow-up duplicate (sim≥0.75) — fallback")
                question = "Proszę o doprecyzowanie poprzedniej odpowiedzi."
                break

    logger.debug(
        "Simulator follow-up [%s] turn=%d: %s",
        perspective,
        state.get("turn_count", 0),
        question[:80],
    )
    return {"question": question, "is_adversarial": False}
