"""
agents/translator.py — Automatyczne tłumaczenie dokumentów na język polski.

Strategia wyboru backendu (w kolejności priorytetu):
  1. DeepL API — jeśli DEEPL_API_KEY ustawiony (najwyższa jakość tłumaczenia)
  2. OpenAI (gpt-4o-mini) — fallback gdy brak DeepL

Tłumaczenie odbywa się na poziomie chunków (nie całych PDF-ów), więc
można je wpiąć do pipeline'u PRZED generowaniem Q&A bez zmiany struktury kodu.

Główna funkcja publiczna:
    translate_text(text, source_lang) → str
    translate_chunks(chunks, source_lang) → list[str]
"""

from __future__ import annotations

import logging

import httpx
import openai
from sqlalchemy import update
from sqlalchemy.orm import Session
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)

# Maksymalna długość tekstu wysyłanego w jednym zapytaniu tłumaczącym
_MAX_CHUNK_CHARS = 3000

# Prompt systemowy dla OpenAI — tłumaczenie prawno-regulacyjne
_TRANSLATE_SYSTEM = (
    "Jesteś profesjonalnym tłumaczem specjalizującym się w prawie korporacyjnym UE "
    "i dokumentach regulacyjnych.\n\n"
    "ZASADY:\n"
    "1. Tłumacz na język POLSKI, zachowując precyzję języka prawniczego.\n"
    "2. Numery artykułów, ustępów i punktów zachowaj bez zmian.\n"
    "3. Nazwy własne aktów prawnych (CSRD, SFDR, DORA itp.) zachowaj w oryginale lub "
    "użyj oficjalnego polskiego odpowiednika.\n"
    "4. Nie dodawaj komentarzy, przypisów ani wyjaśnień — tylko tłumaczenie.\n"
    "5. Zachowaj strukturę (nagłówki, listy, numerację)."
)


# ---------------------------------------------------------------------------
# Retry decorator — obsługa rate limitów (DeepL / OpenAI)
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (429, 503):
        return True
    return False


def _retry(func):
    return retry(
        reraise=True,
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(
            multiplier=1,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


# ---------------------------------------------------------------------------
# Backend 1: DeepL API (najwyższa jakość)
# ---------------------------------------------------------------------------

@_retry
def _translate_deepl(text: str, source_lang: str) -> str:
    """Tłumaczy tekst przez DeepL API (bezpłatna lub płatna wersja)."""
    # DeepL Free API endpoint; płatna → api.deepl.com
    url = "https://api-free.deepl.com/v2/translate"
    # Mapowanie kodów języków na format DeepL
    deepl_lang_map = {
        "en": "EN", "de": "DE", "fr": "FR",
        "es": "ES", "it": "IT", "pt": "PT",
        "nl": "NL", "pl": "PL",
    }
    src = deepl_lang_map.get(source_lang.lower(), source_lang.upper())
    response = httpx.post(
        url,
        data={
            "auth_key": settings.deepl_api_key,
            "text": text,
            "source_lang": src,
            "target_lang": "PL",
            "formality": "prefer_more",  # formalny styl prawniczy
        },
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    return result["translations"][0]["text"]


# ---------------------------------------------------------------------------
# Backend 2: OpenAI
# ---------------------------------------------------------------------------

@_retry
def _translate_openai(text: str, source_lang: str) -> str:
    """Tłumaczy przez OpenAI gpt-4o-mini."""
    lang_names = {
        "en": "angielskiego", "de": "niemieckiego",
        "fr": "francuskiego", "es": "hiszpańskiego",
        "it": "włoskiego", "pt": "portugalskiego",
    }
    lang_name = lang_names.get(source_lang.lower(), source_lang)
    user_prompt = (
        f"Przetłumacz poniższy fragment dokumentu prawnego z języka {lang_name} na polski:\n\n"
        f"{text}"
    )

    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_primary_model,
        messages=[
            {"role": "system", "content": _TRANSLATE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=settings.generation_max_tokens,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_text(text: str, source_lang: str = "en") -> str:
    """
    Przetłumacz tekst na język polski.

    Args:
        text:        Tekst do przetłumaczenia.
        source_lang: Kod języka źródłowego ('en', 'de', 'fr', itp.).

    Returns:
        Przetłumaczony tekst po polsku.
        Jeśli język źródłowy to 'pl' — zwraca oryginał bez zmian.
    """
    if not text or not text.strip():
        return text

    source_lang = source_lang.lower()
    if source_lang == "pl":
        return text  # Nic do tłumaczenia

    # Przycinaj do max długości (zbyt długie teksty → błędy API)
    if len(text) > _MAX_CHUNK_CHARS:
        logger.debug(
            "Translator: truncating text from %d to %d chars", len(text), _MAX_CHUNK_CHARS
        )
        text = text[:_MAX_CHUNK_CHARS]

    try:
        if settings.deepl_api_key:
            try:
                translated = _translate_deepl(text, source_lang)
                logger.debug("Translated via DeepL (%d → %d chars)", len(text), len(translated))
                return translated
            except Exception as deepl_exc:
                logger.warning("DeepL failed (%s) — falling back to OpenAI", deepl_exc)
        translated = _translate_openai(text, source_lang)
        logger.debug("Translated via OpenAI (%d → %d chars)", len(text), len(translated))
        return translated
    except Exception as exc:
        logger.error("Translation failed: %s — returning original text", exc)
        return text  # Bezpieczny fallback: nie blokuj pipeline'u


def translate_chunks_in_db(
    session: Session,
    chunk_ids: list[str],
    source_lang: str = "en",
) -> int:
    """
    Tłumaczy zawartość chunków bezpośrednio w bazie danych (in-place).

    Args:
        session:     Aktywna sesja SQLAlchemy.
        chunk_ids:   Lista UUIDs chunków do przetłumaczenia.
        source_lang: Kod języka źródłowego.

    Returns:
        Liczba przetłumaczonych chunków.
    """
    from db.models import DirectiveChunk
    from sqlalchemy import select
    import uuid

    if not chunk_ids or source_lang == "pl":
        return 0

    uuids = [uuid.UUID(cid) for cid in chunk_ids]
    chunks = list(session.scalars(
        select(DirectiveChunk).where(DirectiveChunk.id.in_(uuids))
    ))

    translated_count = 0
    for chunk in chunks:
        if not chunk.content:
            continue
        translated = translate_text(chunk.content, source_lang)
        chunk.content = translated
        if chunk.content_md and chunk.content_md != chunk.content:
            chunk.content_md = translate_text(chunk.content_md, source_lang)
        else:
            chunk.content_md = translated
        translated_count += 1

    if translated_count:
        session.flush()
        logger.info(
            "DB translator: %d/%d chunks translated (%s → pl) for %d chunk IDs",
            translated_count, len(chunk_ids), source_lang, len(chunk_ids),
        )
    return translated_count


def translate_chunks(
    chunks: list[dict],
    source_lang: str = "en",
) -> list[dict]:
    """
    Przetłumacz zawartość chunków na język polski.

    Modyfikuje pola 'content' i 'content_md' każdego chunku in-place.
    Zwraca zmodyfikowaną listę chunków (dla wygody).
    """
    if source_lang == "pl":
        return chunks

    translated_count = 0
    for chunk in chunks:
        original = chunk.get("content", "")
        if original:
            chunk["content"] = translate_text(original, source_lang)
            translated_count += 1
        # Tłumacz też wersję markdown jeśli istnieje i różni się od content
        md = chunk.get("content_md", "")
        if md and md != original:
            chunk["content_md"] = translate_text(md, source_lang)

    logger.info(
        "Translator: %d/%d chunks translated (%s → pl)",
        translated_count, len(chunks), source_lang,
    )
    return chunks
