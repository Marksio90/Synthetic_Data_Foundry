"""
pipeline/state.py — LangGraph shared state schema.

All agents read from and write to this TypedDict.
LangGraph passes the entire state dict between nodes; each node
returns a partial dict with only the keys it modified.
"""

from __future__ import annotations

import uuid
from typing import Optional
from typing_extensions import TypedDict


class ChunkMeta(TypedDict):
    """Minimal descriptor of the directive chunk being processed."""
    id: str                  # UUID string
    content: str             # plain-text chunk body
    content_md: str          # markdown version (with tables)
    source_document: str     # filename
    chunk_index: int
    section_heading: str
    valid_from_date: str     # ISO date string or ""


class FoundryState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    chunk: ChunkMeta

    # ── Simulator output ───────────────────────────────────────────────────
    question: str
    is_adversarial: bool     # True → question asks about something NOT in text
    perspective: str         # "cfo" | "prawnik" | "audytor"

    # ── Expert RAG phase ───────────────────────────────────────────────────
    retrieved_context: list[str]   # list of relevant chunk snippets
    retrieved_ids: list[str]       # UUIDs of retrieved chunks (for audit)

    # ── Expert generation phase ─────────────────────────────────────────────
    answer: str

    # ── Judge output ───────────────────────────────────────────────────────
    quality_score: float           # 0.0 – 1.0
    judge_model: str               # which model was used
    judge_reasoning: str

    # ── Control flow ───────────────────────────────────────────────────────
    retry_count: int               # how many times this chunk was retried
    status: str                    # new | in_progress | ready | unresolvable
    error_message: Optional[str]

    # ── Output tracking ────────────────────────────────────────────────────
    sample_id: Optional[str]       # UUID of GeneratedSample row
    batch_id: str                  # logical batch identifier
    record_index: int              # position in output JSONL (global counter)
