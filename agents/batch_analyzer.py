"""
agents/batch_analyzer.py — OpenAI Batch API integration for Analyzer and Judge agents.

50% cost reduction for non-time-critical LLM calls by batching requests
via OpenAI's Batch API (24h completion window).

When to use:
  - Bulk generation of Q&A pairs for entire document corpus
  - Judge evaluation of existing samples in DB (post-hoc scoring)
  - Translation runs (non-real-time)

When NOT to use:
  - Interactive sessions (real-time generation)
  - Retries (need immediate feedback)

Pipeline:
  1. Collect requests → JSONL input file
  2. Upload to OpenAI Files API
  3. Submit batch job
  4. Poll until complete (stored in openai_batch_jobs table)
  5. Download results → process → store in generated_samples
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

logger = logging.getLogger("foundry.agents.batch_analyzer")

_BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")
_BATCH_ENDPOINT = "/v1/chat/completions"
_POLL_INTERVAL_SEC = 30
_MAX_POLL_ATTEMPTS = 2880  # 24h / 30s = 2880


@dataclass
class BatchRequest:
    custom_id: str          # e.g. "chunk_{chunk_id}_persp_{perspective}"
    messages: list[dict]
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class BatchResult:
    custom_id: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: Optional[str] = None


class BatchAnalyzer:
    """
    Submit batches of LLM requests to OpenAI Batch API and retrieve results.
    Handles file upload, job submission, polling, and result parsing.
    """

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
            from config.settings import settings
            self._client = openai.OpenAI(api_key=settings.openai_api_key)
        except Exception as exc:
            logger.error("Cannot initialise OpenAI client: %s", exc)
        return self._client

    def prepare_batch_file(self, requests: List[BatchRequest]) -> str:
        """
        Write requests to a JSONL temp file and return the file path.
        Each line is an OpenAI Batch API request object.
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for req in requests:
            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": _BATCH_ENDPOINT,
                "body": {
                    "model": req.model,
                    "messages": req.messages,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                },
            }
            tmp.write(json.dumps(line, ensure_ascii=False) + "\n")
        tmp.close()
        logger.info("Batch file prepared: %d requests → %s", len(requests), tmp.name)
        return tmp.name

    def submit_batch(
        self,
        requests: List[BatchRequest],
        description: str = "Foundry batch job",
    ) -> Optional[str]:
        """
        Upload requests as a batch job to OpenAI. Returns batch_job_id or None.
        Stores job record in openai_batch_jobs table.
        """
        client = self._get_client()
        if client is None:
            return None

        file_path = self.prepare_batch_file(requests)
        try:
            with open(file_path, "rb") as f:
                uploaded = client.files.create(file=f, purpose="batch")
            logger.info("Batch file uploaded: file_id=%s", uploaded.id)

            batch_job = client.batches.create(
                input_file_id=uploaded.id,
                endpoint=_BATCH_ENDPOINT,
                completion_window=_BATCH_COMPLETION_WINDOW,
                metadata={"description": description},
            )

            self._persist_job(
                batch_job_id=batch_job.id,
                input_file_id=uploaded.id,
                sample_ids=[r.custom_id for r in requests],
            )

            logger.info(
                "Batch job submitted: id=%s requests=%d window=%s",
                batch_job.id, len(requests), _BATCH_COMPLETION_WINDOW,
            )
            return batch_job.id

        except Exception as exc:
            logger.error("Batch submission failed: %s", exc)
            return None
        finally:
            try:
                os.unlink(file_path)
            except Exception:
                pass

    def poll_until_complete(self, batch_job_id: str) -> Optional[str]:
        """
        Poll batch job status until completed or failed.
        Returns output_file_id or None.
        Updates openai_batch_jobs table.
        """
        client = self._get_client()
        if client is None:
            return None

        for attempt in range(_MAX_POLL_ATTEMPTS):
            try:
                job = client.batches.retrieve(batch_job_id)
                status = job.status

                if status == "completed":
                    self._update_job(batch_job_id, status="completed", output_file_id=job.output_file_id)
                    logger.info("Batch job %s completed. output_file_id=%s", batch_job_id, job.output_file_id)
                    return job.output_file_id

                if status in ("failed", "cancelled", "expired"):
                    self._update_job(batch_job_id, status=status, error=str(job.errors))
                    logger.error("Batch job %s terminal status: %s", batch_job_id, status)
                    return None

                if attempt % 10 == 0:
                    completed = getattr(job.request_counts, "completed", "?")
                    total = getattr(job.request_counts, "total", "?")
                    logger.info(
                        "Batch job %s: status=%s progress=%s/%s",
                        batch_job_id, status, completed, total,
                    )

            except Exception as exc:
                logger.warning("Batch poll error (attempt %d): %s", attempt, exc)

            time.sleep(_POLL_INTERVAL_SEC)

        logger.error("Batch job %s timed out after %d polls.", batch_job_id, _MAX_POLL_ATTEMPTS)
        return None

    def download_results(self, output_file_id: str) -> Iterator[BatchResult]:
        """
        Download and parse batch results. Yields BatchResult per request.
        """
        client = self._get_client()
        if client is None:
            return

        try:
            content = client.files.content(output_file_id).content
            lines = content.decode("utf-8").strip().split("\n")

            _COST_PER_1M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (5.00, 15.00)}

            for line in lines:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    custom_id = obj.get("custom_id", "")
                    resp = obj.get("response", {})
                    body = resp.get("body", {})

                    if resp.get("status_code") != 200:
                        yield BatchResult(
                            custom_id=custom_id,
                            content="",
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost_usd=0.0,
                            error=str(body.get("error", "unknown error")),
                        )
                        continue

                    choices = body.get("choices", [{}])
                    content_text = choices[0].get("message", {}).get("content", "") if choices else ""
                    usage = body.get("usage", {})
                    p_tok = usage.get("prompt_tokens", 0)
                    c_tok = usage.get("completion_tokens", 0)
                    model = body.get("model", "gpt-4o-mini")
                    c_in, c_out = _COST_PER_1M.get(model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model, (0.15, 0.60))
                    cost = (p_tok / 1_000_000) * c_in + (c_tok / 1_000_000) * c_out

                    yield BatchResult(
                        custom_id=custom_id,
                        content=content_text,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        cost_usd=cost,
                    )
                except Exception as exc:
                    logger.warning("Failed to parse batch result line: %s", exc)

        except Exception as exc:
            logger.error("Batch download failed for %s: %s", output_file_id, exc)

    def _persist_job(self, batch_job_id: str, input_file_id: str, sample_ids: list[str]) -> None:
        try:
            from sqlalchemy import create_engine, text
            from config.settings import settings

            engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=1)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO openai_batch_jobs (batch_job_id, status, input_file_id, sample_ids)
                        VALUES (:bid, 'submitted', :fid, ARRAY[:sample_ids]::text[])
                        ON CONFLICT (batch_job_id) DO NOTHING
                        """
                    ),
                    {"bid": batch_job_id, "fid": input_file_id, "sample_ids": sample_ids},
                )
            engine.dispose()
        except Exception as exc:
            logger.debug("_persist_job failed: %s", exc)

    def _update_job(
        self,
        batch_job_id: str,
        status: str,
        output_file_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        try:
            from sqlalchemy import create_engine, text
            from config.settings import settings

            engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=1)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE openai_batch_jobs
                        SET status = :status,
                            output_file_id = COALESCE(:ofid, output_file_id),
                            completed_at = CASE WHEN :status IN ('completed','failed') THEN NOW() ELSE completed_at END,
                            error_detail = COALESCE(:error, error_detail)
                        WHERE batch_job_id = :bid
                        """
                    ),
                    {"bid": batch_job_id, "status": status, "ofid": output_file_id, "error": error},
                )
            engine.dispose()
        except Exception as exc:
            logger.debug("_update_job failed: %s", exc)


# Module-level singleton
_batch_analyzer: Optional[BatchAnalyzer] = None


def get_batch_analyzer() -> BatchAnalyzer:
    global _batch_analyzer
    if _batch_analyzer is None:
        _batch_analyzer = BatchAnalyzer()
    return _batch_analyzer
