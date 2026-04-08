"""
utils/backoff.py — Tenacity-powered retry decorators.

Self-Check 2.0 patch: Exponential backoff for all OpenAI calls.
  - HTTP 429 (RateLimitError)  → wait 2 → 4 → 8 → 16 → 32 → 64 seconds
  - HTTP 5xx (APIStatusError)  → same ladder (transient server errors)
  - Any other openai.APIError  → reraise immediately (bad request, auth, etc.)

Batch API helper included: use submit_openai_batch() to get 50% cost discount
on bulk Judge evaluations; results polled asynchronously after ~1-2 hours.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception predicate helpers
# ---------------------------------------------------------------------------

def _is_rate_limit(exc: BaseException) -> bool:
    return isinstance(exc, openai.RateLimitError)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (openai.RateLimitError, openai.APIStatusError))


# ---------------------------------------------------------------------------
# Retry decorator factory
# ---------------------------------------------------------------------------

def openai_retry(func: Callable) -> Callable:
    """
    Decorator: retry *func* on transient OpenAI errors with exponential backoff.
    Configuration pulled from settings so it can be overridden via env vars.
    """
    return retry(
        reraise=True,
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIStatusError)),
        wait=wait_exponential(
            multiplier=1,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


# ---------------------------------------------------------------------------
# Async variant (for async agent code)
# ---------------------------------------------------------------------------

def openai_retry_async(func: Callable) -> Callable:
    return retry(
        reraise=True,
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIStatusError)),
        wait=wait_exponential(
            multiplier=1,
            min=settings.tenacity_initial_wait,
            max=settings.tenacity_max_wait,
        ),
        stop=stop_after_attempt(settings.tenacity_max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


# ---------------------------------------------------------------------------
# OpenAI Batch API helpers (50% cost discount, async delivery ~1-2h)
# ---------------------------------------------------------------------------

def build_batch_request(
    custom_id: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 512,
) -> dict:
    """Build one line of an OpenAI Batch API .jsonl input file."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
    }


async def submit_openai_batch(
    client: openai.AsyncOpenAI,
    requests: list[dict],
    description: str = "esg-judge-batch",
) -> str:
    """
    Upload *requests* as a .jsonl file, start a batch job, return batch_job_id.
    Caller is responsible for polling until status == 'completed'.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        for req in requests:
            tmp.write(json.dumps(req, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose="batch")

    batch_job = await client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    logger.info("OpenAI batch job submitted: %s", batch_job.id)
    Path(tmp_path).unlink(missing_ok=True)
    return batch_job.id


async def poll_batch_until_done(
    client: openai.AsyncOpenAI,
    batch_job_id: str,
    poll_interval: int = 60,
    timeout: int = 7200,
) -> list[dict]:
    """
    Poll *batch_job_id* every *poll_interval* seconds until completed or failed.
    Returns list of result dicts from the output .jsonl.
    Raises RuntimeError on failure or timeout.
    """
    elapsed = 0
    while elapsed < timeout:
        job = await client.batches.retrieve(batch_job_id)
        logger.info("Batch %s status: %s", batch_job_id, job.status)
        if job.status == "completed":
            content = await client.files.content(job.output_file_id)
            results = []
            for line in content.text.strip().splitlines():
                results.append(json.loads(line))
            return results
        if job.status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"Batch job {batch_job_id} ended with status {job.status}")
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Batch job {batch_job_id} did not complete within {timeout}s")
