"""
agents/chunker.py — Agent Chunker (Fragmentator)

Semantic chunking with overlap:
  - Splits Markdown on H1/H2/H3 headings first, then on paragraph breaks.
  - Appends the last OVERLAP_CHARS characters of the previous chunk to the
    beginning of the current one (context window bridge).
  - Generates embeddings (OpenAI text-embedding-3-small) and stores vectors
    in pgvector for hybrid RAG retrieval.
  - Sets is_superseded and valid_from_date inherited from the parent document.
"""

from __future__ import annotations

import logging
import re
import uuid

import openai
from sqlalchemy.orm import Session

from config.settings import settings
from db import repository as repo

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_MIN_CHUNK_CHARS = 250   # zwiększone: dokumenty prawne mają min. sensowne sekcje
_MAX_CHUNK_CHARS = 2200  # zmniejszone: zostawia ~800 tokenów na odpowiedź LLM

# Detekcja artykułów prawnych jako naturalnych granic chunków.
# Obsługuje formaty: "Article 5", "Art. 5", "Artykuł 5", "§ 3", "Section 12"
_LEGAL_ARTICLE_RE = re.compile(
    r"^(?:Article|Art(?:ykuł)?\.?|§|Section)\s+\d+\b.*$",
    re.MULTILINE | re.IGNORECASE,
)


def _split_by_legal_articles(text: str) -> list[tuple[str, str]]:
    """
    Dzieli tekst na sekcje wg nagłówków artykułów prawnych.
    Zwraca pary (nagłówek_artykułu, treść).
    Używane gdy tekst nie ma nagłówków Markdown, ale zawiera artykuły prawne.
    """
    matches = list(_LEGAL_ARTICLE_RE.finditer(text))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """
    Zwraca listę par (nagłówek, treść).
    Priorytet: 1) nagłówki Markdown, 2) artykuły prawne, 3) cały tekst jako jedna sekcja.
    """
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(markdown))

    if matches:
        for i, m in enumerate(matches):
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            body = markdown[start:end].strip()
            if body:
                sections.append((heading, body))
        return sections

    # Brak nagłówków Markdown — spróbuj artykułów prawnych
    legal_sections = _split_by_legal_articles(markdown)
    if legal_sections:
        return legal_sections

    return [("", markdown.strip())]


_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ\"\(\[])")


def _split_by_sentences(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries, respecting max_chars. Avoids cutting mid-sentence."""
    sentences = _SENTENCE_END_RE.split(text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip() if current else sent.strip()
        else:
            if current:
                chunks.append(current)
            # Sentence itself longer than max → hard split at word boundary
            if len(sent) > max_chars:
                words = sent.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        current = (current + " " + word).strip() if current else word
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = sent.strip()
    if current:
        chunks.append(current)
    return chunks or [text]


def _split_long_section(body: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split body on paragraph breaks, then sentence boundaries if still too long."""
    if len(body) <= max_chars:
        return [body]
    paragraphs = re.split(r"\n{2,}", body)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Paragraph itself too long → split by sentences
            if len(para) > max_chars:
                chunks.extend(_split_by_sentences(para, max_chars))
                current = ""
            else:
                current = para.strip()
    if current:
        chunks.append(current)
    return chunks or [body]


def _merge_small_chunks(raw: list[tuple[str, str]], min_chars: int) -> list[tuple[str, str]]:
    """
    Merge consecutive chunks that are too short (< min_chars) with the next chunk.
    Prevents tiny orphan chunks that confuse the LLM.
    """
    if not raw:
        return raw
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(raw):
        heading, body = raw[i]
        # Merge forward while current chunk is too small and next exists
        while len(body) < min_chars and i + 1 < len(raw):
            i += 1
            next_heading, next_body = raw[i]
            sep = f"\n\n## {next_heading}\n\n" if next_heading else "\n\n"
            body = body + sep + next_body
        merged.append((heading, body))
        i += 1
    return merged


def _add_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """
    Prepend the last *overlap_chars* of the previous chunk to each chunk.
    Uses a clear separator to preserve semantic boundaries (not a bare space).
    """
    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap_chars:].strip()
        # Separator makes the overlap boundary explicit for the model
        result.append(f"[...{tail}]\n\n{chunks[i]}")
    return result


def _embed(texts: list[str]) -> list[list[float]]:
    """Call OpenAI text-embedding-3-small for a batch of texts."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
        dimensions=settings.openai_embedding_dims,
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def chunk_document(
    session: Session,
    source_doc_id: str,
    markdown: str,
    valid_from_date=None,
    is_superseded: bool = False,
    overlap_chars: int = settings.chunk_overlap_chars,
) -> list[str]:
    """
    Split *markdown* into overlapping semantic chunks, embed them, and
    persist to directive_chunks.  Returns list of chunk UUIDs.
    """
    sections = _split_by_headings(markdown)
    raw_chunks: list[tuple[str, str]] = []  # (heading, body)

    for heading, body in sections:
        for part in _split_long_section(body):
            if len(part.strip()) >= _MIN_CHUNK_CHARS // 2:  # looser pre-merge filter
                raw_chunks.append((heading, part.strip()))

    # Merge orphan short chunks before embedding
    raw_chunks = _merge_small_chunks(raw_chunks, _MIN_CHUNK_CHARS)
    # Final filter after merge
    raw_chunks = [(h, b) for h, b in raw_chunks if len(b) >= _MIN_CHUNK_CHARS]

    if not raw_chunks:
        logger.warning("No usable chunks extracted from doc %s", source_doc_id)
        return []

    bodies = [body for _, body in raw_chunks]
    bodies_with_overlap = _add_overlap(bodies, overlap_chars)

    # Embed in batches of 100 (API limit)
    all_embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(bodies_with_overlap), batch_size):
        batch = bodies_with_overlap[i : i + batch_size]
        all_embeddings.extend(_embed(batch))

    chunk_ids: list[str] = []
    for idx, ((heading, _), content, embedding) in enumerate(
        zip(raw_chunks, bodies_with_overlap, all_embeddings)
    ):
        chunk = repo.insert_chunk(
            session,
            source_doc_id=uuid.UUID(source_doc_id),
            chunk_index=idx,
            content=content,
            content_md=content,  # Markdown preserved
            embedding=embedding,
            valid_from_date=valid_from_date,
            is_superseded=is_superseded,
            section_heading=heading,
            status="new",
        )
        chunk_ids.append(str(chunk.id))

    session.commit()

    # Log chunk size distribution for quality monitoring
    lengths = [len(c) for _, c in raw_chunks]
    if lengths:
        logger.info(
            "Chunked doc %s → %d chunks (overlap=%d) | len: min=%d p50=%d max=%d",
            source_doc_id, len(chunk_ids), overlap_chars,
            min(lengths),
            sorted(lengths)[len(lengths) // 2],
            max(lengths),
        )
    return chunk_ids
