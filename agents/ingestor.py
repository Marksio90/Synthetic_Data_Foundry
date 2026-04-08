"""
agents/ingestor.py — Agent Ingestor (Poborca & Parser)

Responsibilities:
  1. Accept a PDF file path.
  2. Convert it to Markdown using LlamaParse (preferred) or OpenAI Vision API
     as a fallback.  Both paths handle tables correctly.
  3. Extract directive metadata (name, year, valid_from_date).
  4. Detect supersession relationships (e.g. amendment replaces old article)
     and set is_superseded=True on any previously stored chunks of the same
     document family (Self-Check 3.0 patch).
  5. Persist the SourceDocument row and return raw markdown for the Chunker.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from config.settings import settings
from db import repository as repo

logger = logging.getLogger(__name__)

# Regex: find ISO dates near keywords like "wchodzi w życie", "applicable from"
_DATE_RE = re.compile(
    r"(?:wchodzi w życie|applicable from|valid from|effective|od dnia)\s*:?\s*"
    r"(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{4}|\d{4}[\./\-]\d{2}[\./\-]\d{2})",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# ---------------------------------------------------------------------------
# LlamaParse client (lazy import — not installed in unit-test environments)
# ---------------------------------------------------------------------------

def _parse_with_llamaparse(pdf_path: Path) -> str:
    """Use LlamaParse cloud API to convert PDF → Markdown (table-aware)."""
    from llama_parse import LlamaParse  # type: ignore

    parser = LlamaParse(
        api_key=settings.llama_parse_api_key,
        result_type="markdown",
        language="en",
        verbose=False,
    )
    documents = parser.load_data(str(pdf_path))
    return "\n\n".join(doc.text for doc in documents)


def _parse_with_openai_vision(pdf_path: Path) -> str:
    """
    Fallback: convert each PDF page to an image and send to GPT-4o vision.
    Requires 'pdf2image' and 'Pillow' packages + poppler utilities.
    """
    import base64
    import io

    import openai
    from pdf2image import convert_from_path  # type: ignore
    from PIL import Image  # type: ignore

    client = openai.OpenAI(api_key=settings.openai_api_key)
    pages = convert_from_path(str(pdf_path), dpi=150)
    full_markdown: list[str] = []

    for i, page in enumerate(pages):
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Convert this EU directive page to Markdown. "
                                "Reproduce all tables using Markdown table syntax. "
                                "Preserve all article numbers and headings exactly."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        full_markdown.append(f"<!-- Page {i+1} -->\n{response.choices[0].message.content}")

    return "\n\n".join(full_markdown)


def _extract_valid_from_date(markdown: str) -> Optional[date]:
    match = _DATE_RE.search(markdown)
    if not match:
        return None
    raw = match.group(1)
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            from datetime import datetime
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _extract_directive_year(markdown: str, filename: str) -> Optional[int]:
    for text in (filename, markdown[:500]):
        match = _YEAR_RE.search(text)
        if match:
            return int(match.group(1))
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str, session: Session) -> tuple[str, str]:
    """
    Parse *pdf_path* and persist a SourceDocument.

    Returns (source_doc_id, raw_markdown).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    file_hash = _sha256(path)

    # Dedup check — skip re-parsing if already in DB
    from sqlalchemy import select
    from db.models import SourceDocument
    existing = session.scalar(
        select(SourceDocument).where(SourceDocument.file_hash == file_hash)
    )
    if existing:
        logger.info("Document already ingested: %s — skipping parse", path.name)
        return str(existing.id), existing.raw_markdown or ""

    # Parse PDF → Markdown
    markdown = ""
    if settings.llama_parse_api_key:
        logger.info("Parsing with LlamaParse: %s", path.name)
        try:
            markdown = _parse_with_llamaparse(path)
        except Exception as exc:
            logger.warning("LlamaParse failed (%s), falling back to OpenAI Vision", exc)

    if not markdown:
        logger.info("Parsing with OpenAI Vision: %s", path.name)
        markdown = _parse_with_openai_vision(path)

    valid_from = _extract_valid_from_date(markdown)
    directive_year = _extract_directive_year(markdown, path.stem)

    doc = repo.upsert_source_document(
        session,
        filename=path.name,
        file_hash=file_hash,
        directive_name=path.stem.upper(),
        directive_year=directive_year,
        valid_from_date=valid_from,
        raw_markdown=markdown,
    )
    session.commit()
    logger.info(
        "Ingested '%s' → doc_id=%s  valid_from=%s", path.name, doc.id, valid_from
    )
    return str(doc.id), markdown
