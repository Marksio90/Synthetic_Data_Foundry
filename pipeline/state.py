"""
pipeline/state.py — LangGraph shared state schema.

All agents read from and write to this TypedDict.
LangGraph passes the entire state dict between nodes; each node
returns a partial dict with only the keys it modified.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class ChunkMeta(TypedDict):
    """Minimal descriptor of the directive chunk being processed."""
    id: str
    content: str
    content_md: str
    source_document: str
    chunk_index: int
    section_heading: str
    valid_from_date: str


class FoundryState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    chunk: ChunkMeta

    # ── Simulator output ───────────────────────────────────────────────────
    question: str
    is_adversarial: bool
    perspective: str        # cfo|prawnik|audytor|analityk|regulator|akademik|dziennikarz|inwestor

    # ── Expert RAG phase ───────────────────────────────────────────────────
    retrieved_context: list[str]
    retrieved_ids: list[str]

    # ── Expert generation phase ─────────────────────────────────────────────
    answer: str

    # ── Constitutional AI (Gods Finger v2) ────────────────────────────────
    constitutional_critique: Optional[str]  # wyniki audytu Constitutional AI

    # ── Judge output ───────────────────────────────────────────────────────
    quality_score: float           # 0.0 – 1.0 (weighted avg)
    judge_model: str
    judge_reasoning: str
    judge_details: dict            # grounding/citation/completeness/language/hallucination

    # ── Control flow ───────────────────────────────────────────────────────
    retry_count: int
    status: str                    # new | in_progress | ready | unresolvable
    error_message: Optional[str]

    # ── Multi-turn conversation ────────────────────────────────────────────
    conversation_history: list
    turn_count: int

    # ── DPO / ORPO / KTO quality metadata ─────────────────────────────────
    rejected_answer: Optional[str]
    question_type: str             # factual | scope | process | compliance | comparative
    difficulty: str                # easy | medium | hard

    # ── Output tracking ────────────────────────────────────────────────────
    sample_id: Optional[str]
    batch_id: str
    record_index: int
