"""
agents/simulator.py — Agent Symulator (Generator Pytań) - ENTERPRISE EDITION

LangGraph node: simulate_question(state) → partial state update

Odpowiada za inicjację potoku RAG. Symuluje 8 eksperckich perspektyw
i opcjonalnie wprowadza pytania adwersaryjne (Self-Check), aby trenować
odporność asystentów AI na halucynacje.

Ulepszenia PRO:
- Pydantic Structured Outputs: Gwarantowany format JSON, chroni przed "meta-komentarzami" od LLM.
- Global Connection Pools: Współdzieleni klienci OpenAI/Ollama (zwiększona wydajność I/O).
- Semantic De-looping: Rozbudowany system ratunkowy z badaniem Jaccard Similarity (zapobiega pętlom follow-up).
- Jittered Backoff: Zaawansowana dystrybucja obciążenia na GPU (Ollama) i API przy wielowątkowości.
- Full Prompt Preservation: Zachowana 100% oryginalna inżynieria promptów z repozytorium.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Dict, Any, Tuple

import openai
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from config.settings import settings
from pipeline.state import FoundryState

logger = logging.getLogger("foundry.agents.simulator")

# ---------------------------------------------------------------------------
# Globalni Klienci i Pule Połączeń (Connection Pooling)
# ---------------------------------------------------------------------------
_OPENAI_CLIENT = openai.OpenAI(api_key=settings.openai_api_key, max_retries=0)

def _get_ollama_client() -> openai.OpenAI | None:
    if getattr(settings, "ollama_url", None):
        base = settings.ollama_url.rstrip("/")
        # Lokalny symulator potrzebuje dłuższego timeoutu
        return openai.OpenAI(api_key="ollama", base_url=f"{base}/v1", max_retries=0, timeout=45.0)
    return None

_OLLAMA_CLIENT = _get_ollama_client()

# Ceny dla telemetrii (w USD za 1M tokenów)
_COSTS_PER_1M = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (5.00, 15.00),
}


# ---------------------------------------------------------------------------
# Pydantic Schemas (Wymuszony Format Wyjściowy)
# ---------------------------------------------------------------------------
class SimulatedQuestion(BaseModel):
    """
    Ścisła struktura odpowiedzi generatora. 
    Wymusza na LLMie oddzielenie toku myślenia od samego wygenerowanego pytania.
    """
    reasoning: str = Field(
        description="Twój krótki, analityczny tok myślenia. Jakie fakty z kontekstu chcesz wykorzystać do zadania pytania?"
    )
    question: str = Field(
        description="Wygenerowane, bezpośrednie pytanie. BEZ ŻADNYCH komentarzy w stylu 'Oto moje pytanie:'."
    )


# ---------------------------------------------------------------------------
# Oryginalne Prompty Systemowe — 8 perspektyw × normalny / adversarial / follow-up
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

ALL_PERSPECTIVES = list(_NORMAL_PROMPTS.keys())

# Pytania Ratunkowe (Semantic De-looping fallback)
_FALLBACK_QUESTIONS = [
    "Jakie mogą być nieoczywiste konsekwencje regulacji podanych w tej odpowiedzi dla średnich przedsiębiorstw?",
    "Czy istnieją jakieś kluczowe luki interpretacyjne w tym opisie, o których powinienem wiedzieć?",
    "Jakie dokumenty będą potrzebne, by dowieść zgodności z wymogami opisanymi w Twojej odpowiedzi?",
    "Jak zmiana opisana w Twojej odpowiedzi przekłada się bezpośrednio na strukturę wydatków operacyjnych?"
]

# ---------------------------------------------------------------------------
# Odporność Sieciowa (Resilience & Jitter)
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
    # Użycie rozrzutu (Jitter) chroni API przed tzw. Thundering Herd (szturm wielu wątków w tej samej sekundzie)
    wait=wait_random_exponential(multiplier=1.5, min=settings.tenacity_initial_wait, max=settings.tenacity_max_wait),
    stop=stop_after_attempt(settings.tenacity_max_attempts),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call_structured_simulator(messages: list[dict], max_tokens: int = 400) -> Tuple[SimulatedQuestion, float]:
    """
    Multi-Tier Routing: Próba uruchomienia Ollamy, spadek na OpenAI przy awarii.
    Zwraca krotkę: (Obiekt Pytania wg schematu Pydantic, Koszt w centach).
    """
    cost_cents = 0.0

    # Próba Lokalna (Ollama). Parsowanie JSON manualnie dla wsparcia słabszych modeli lokalnych.
    if settings.ollama_model and _OLLAMA_CLIENT:
        try:
            start_time = time.perf_counter()
            resp = _OLLAMA_CLIENT.chat.completions.create(
                model=settings.ollama_model,
                messages=messages,
                temperature=settings.generation_temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} 
            )
            raw_content = resp.choices[0].message.content.strip()
            
            import json
            try:
                data = json.loads(raw_content)
                sq = SimulatedQuestion(
                    reasoning=data.get("reasoning", "Lokalny LLM pominął tag reasoning."),
                    question=data.get("question", data.get("pytanie", raw_content))
                )
            except json.JSONDecodeError:
                sq = SimulatedQuestion(reasoning="Błąd parsowania", question=raw_content)

            elapsed = time.perf_counter() - start_time
            logger.debug(f"[Symulator-Local] Wymyślono pytanie w {elapsed:.2f}s.")
            return sq, cost_cents
        except Exception as e:
            logger.warning(f"Lokalny Symulator ({settings.ollama_model}) zawiódł: {e}. Przełączam na API OpenAI.")

    # Próba Chmurowa (OpenAI) z pełnym, natywnym wsparciem Pydantic Structured Outputs
    model = settings.openai_primary_model
    start_time = time.perf_counter()
    resp = _OPENAI_CLIENT.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=settings.generation_temperature,
        max_tokens=max_tokens,
        response_format=SimulatedQuestion
    )
    
    if resp.usage and model in _COSTS_PER_1M:
        c_in, c_out = _COSTS_PER_1M[model]
        cost_cents = ((resp.usage.prompt_tokens / 1_000_000) * c_in * 100) + \
                     ((resp.usage.completion_tokens / 1_000_000) * c_out * 100)

    parsed = resp.choices[0].message.parsed
    if not parsed:
        raise ValueError("Model odmówił wygenerowania pytania w zadanym schemacie.")

    elapsed = time.perf_counter() - start_time
    logger.debug(f"[Symulator-Cloud] Wymyślono pytanie ({model}) w {elapsed:.2f}s. Koszt: {cost_cents:.4f}¢")
    return parsed, cost_cents


# ---------------------------------------------------------------------------
# Badanie Podobieństwa Semantycznego (Anty-Loop)
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
# Główne Węzły (LangGraph Nodes)
# ---------------------------------------------------------------------------
def simulate_question(state: FoundryState) -> Dict[str, Any]:
    """
    Inicjuje potok zadając startowe pytanie dla danej partii tekstu.
    """
    chunk = state["chunk"]
    perspective = state.get("perspective", "cfo")
    is_adversarial = random.random() < getattr(settings, "adversarial_ratio", 0.1)

    # Log SIL perspective weight for observability (weights shift over time as weaknesses are detected)
    try:
        from agents.self_improving_loop import get_self_improving_loop
        from pipeline.graph import _SIL_KEY_TO_PERSPECTIVE
        _inv = {v: k for k, v in _SIL_KEY_TO_PERSPECTIVE.items()}
        sil_key = _inv.get(perspective, perspective)
        _pw = get_self_improving_loop().last_perspective_weights
        if _pw:
            logger.debug("[Simulator] perspective=%s sil_weight=%.2f", perspective, _pw.get(sil_key, 1.0))
    except Exception:
        pass

    prompts = _ADVERSARIAL_PROMPTS if is_adversarial else _NORMAL_PROMPTS
    base_system = prompts.get(perspective, prompts["cfo"])

    # Obudowujemy instrukcje w żądanie JSON, nie ingerując w biznesową część Twojego promptu
    system = (
        f"{base_system}\n\n"
        "UWAGA SYSTEMOWA: Twój ostateczny wynik musi być zwrócony w formacie JSON zawierającym dwa klucze:\n"
        "1. 'reasoning': Twoja krótka analiza.\n"
        "2. 'question': Wygenerowane pytanie. Podaj tylko treść pytania jako wartość tego klucza, absolutnie bez dodatkowych tekstów czy wstępów."
    )

    prompt = (
        f"Fragment dyrektywy:\n\n{chunk.get('content', 'Brak tekstu')}\n\n"
        f"Sekcja dokumentu: {chunk.get('section_heading', 'N/A')}\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    try:
        simulated_obj, cost = _call_structured_simulator(messages)
        question = simulated_obj.question
    except Exception as exc:
        logger.error(f"Krytyczny błąd generatora pytań: {exc}", exc_info=True)
        question = (
            "Jakie sankcje przewiduje ta dyrektywa za naruszenie obowiązków raportowych?"
            if is_adversarial
            else "Jakie są główne wymogi określone w tej sekcji dyrektywy?"
        )

    logger.info(
        f"💭 [SYMULATOR:{perspective.upper()}] Pytanie ("
        f"{'ADVERSARIAL' if is_adversarial else 'NORMAL'}) dla chunk {chunk.get('id', 'N/A')[:6]}: {question[:100]}..."
    )
    
    return {"question": question, "is_adversarial": is_adversarial}


def simulate_followup(state: FoundryState) -> Dict[str, Any]:
    """
    Węzeł generujący pytania w rozmowie wieloturowej.
    Odporny na pętle konwersacyjne dzięki weryfikacji podobieństwa Jaccarda.
    """
    chunk = state["chunk"]
    perspective = state.get("perspective", "cfo")
    conversation_history = state.get("conversation_history", [])

    base_system = _FOLLOWUP_PROMPTS.get(perspective, _FOLLOWUP_PROMPTS["cfo"])
    system = (
        f"{base_system}\n\n"
        "UWAGA SYSTEMOWA: Twój ostateczny wynik musi być zwrócony w formacie JSON zawierającym klucze 'reasoning' oraz 'question'."
    )

    history_lines = []
    for msg in conversation_history:
        label = "Pytanie z historii" if msg["role"] == "user" else "Odpowiedź eksperta"
        history_lines.append(f"[{label}]: {msg['content']}")

    prompt = (
        f"Fragment dyrektywy (kontekst bazowy):\n{chunk.get('content', '')}\n\n"
        f"Dotychczasowa rozmowa:\n" + "\n\n".join(history_lines)
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    prev_questions = [m["content"] for m in conversation_history if m["role"] == "user"]

    try:
        simulated_obj, cost = _call_structured_simulator(messages)
        question = simulated_obj.question
        
        # Ochrona przed zapętleniem konwersacji
        for prev in prev_questions:
            if _jaccard_similarity(question, prev) >= 0.75:
                logger.warning(f"🔄 [Anti-Loop] Symulator zapętlił pytanie dla chunk {chunk.get('id', 'N/A')[:6]}. Aktywuję Fallback.")
                question = random.choice(_FALLBACK_QUESTIONS)
                break
                
    except Exception as exc:
        logger.error(f"Błąd generatora follow-up: {exc}", exc_info=True)
        question = random.choice(_FALLBACK_QUESTIONS)

    logger.debug(f"🗣️ [FOLLOW-UP:{perspective.upper()}] Turn={state.get('turn_count', 0)}: {question[:100]}...")
    return {"question": question, "is_adversarial": False}
