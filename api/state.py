"""
api/state.py — Shared in-memory run state for the Foundry API.

One RunManager singleton tracks all pipeline runs in the current process.
Each run stores: status, log lines, progress counters, analysis/calibration results.

Thread-safe for use from asyncio background tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunRecord:
    run_id: str
    batch_id: str
    status: str = "starting"          # starting | running | done | error
    progress_pct: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    records_written: int = 0
    dpo_pairs: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)
    analysis: Optional[dict[str, Any]] = None
    calibration: Optional[dict[str, Any]] = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.ended_at or time.time()
        return round(end - self.started_at, 1)


class RunManager:
    """
    Simple in-memory store for pipeline run records.
    Designed for single-process use (one Uvicorn worker).
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def create(self, run_id: str, batch_id: str) -> RunRecord:
        rec = RunRecord(run_id=run_id, batch_id=batch_id)
        self._runs[run_id] = rec
        return rec

    def get(self, run_id: str) -> Optional[RunRecord]:
        return self._runs.get(run_id)

    def update(self, run_id: str, **kwargs: Any) -> None:
        rec = self._runs.get(run_id)
        if rec is None:
            return
        for k, v in kwargs.items():
            if hasattr(rec, k):
                setattr(rec, k, v)

    def append_log(self, run_id: str, line: str) -> None:
        rec = self._runs.get(run_id)
        if rec is not None:
            rec.log_lines.append(line)
            # Parse progress hints from log lines
            _parse_progress(rec, line)

    def list_runs(self) -> list[RunRecord]:
        return list(self._runs.values())


# ---------------------------------------------------------------------------
# Log-line parser — extracts structured progress from main.py stdout
# ---------------------------------------------------------------------------

import re

_PROGRESS_RE = re.compile(
    r"Progress:\s*(\d+)\s*chunks\s*\|\s*(\d+)\s*ready\s*\|\s*(\d+)\s*unresolvable"
    r"\s*\|\s*(\d+)\s*records\s*\|\s*(\d+)\s*DPO"
)
_RECORD_RE = re.compile(r"✓ Record (\d+) written")
_FOUND_RE = re.compile(r"Found (\d+) pending chunks")
_DONE_RE = re.compile(r"ESG Data Foundry — Run Complete")
_ERROR_RE = re.compile(r"\[ERROR\]")


def _parse_progress(rec: RunRecord, line: str) -> None:
    m = _PROGRESS_RE.search(line)
    if m:
        rec.chunks_done = int(m.group(1))
        rec.records_written = int(m.group(4))
        rec.dpo_pairs = int(m.group(5))
        if rec.chunks_total > 0:
            rec.progress_pct = min(int(rec.chunks_done / rec.chunks_total * 100), 99)
        return

    m = _FOUND_RE.search(line)
    if m:
        rec.chunks_total = int(m.group(1))
        rec.status = "running"
        return

    m = _RECORD_RE.search(line)
    if m:
        rec.records_written = int(m.group(1)) + 1
        return

    if _DONE_RE.search(line):
        rec.status = "done"
        rec.progress_pct = 100
        rec.ended_at = time.time()
        return

    if _ERROR_RE.search(line) and rec.status == "running":
        # Don't switch to error on every error log — only fatal ones
        pass


# Singleton imported by routers
runs = RunManager()
