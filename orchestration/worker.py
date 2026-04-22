"""
orchestration/worker.py — Temporal.io worker process.

Registers all Foundry activities and workflows, then polls the Temporal server
for tasks. Run one or more workers to scale pipeline throughput.

Usage:
    python -m orchestration.worker

    # With custom concurrency
    MAX_CONCURRENT_ACTIVITIES=16 python -m orchestration.worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from config.logging_config import configure_logging

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("foundry.worker")

TEMPORAL_URL = os.getenv("TEMPORAL_URL", "localhost:7233")
TASK_QUEUE = os.getenv("TEMPORAL_TASK_QUEUE", "foundry-tasks")
MAX_CONCURRENT_ACTIVITIES = int(os.getenv("MAX_CONCURRENT_ACTIVITIES", "8"))
MAX_CONCURRENT_WORKFLOW_TASKS = int(os.getenv("MAX_CONCURRENT_WORKFLOW_TASKS", "4"))


async def main() -> None:
    try:
        from temporalio.client import Client  # type: ignore
        from temporalio.worker import Worker  # type: ignore
    except ImportError:
        logger.error(
            "temporalio package not installed. Run: pip install temporalio\n"
            "Then start Temporal server: docker-compose --profile temporal up"
        )
        sys.exit(1)

    from orchestration.temporal_workflows import (
        FoundryDocumentWorkflow,
        calibrate_activity,
        cross_doc_activity,
        finalize_activity,
        ingest_document_activity,
        process_chunk_activity,
    )

    client = await Client.connect(TEMPORAL_URL)
    logger.info("Connected to Temporal server at %s", TEMPORAL_URL)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FoundryDocumentWorkflow],
        activities=[
            ingest_document_activity,
            calibrate_activity,
            process_chunk_activity,
            cross_doc_activity,
            finalize_activity,
        ],
        max_concurrent_activities=MAX_CONCURRENT_ACTIVITIES,
        max_concurrent_workflow_tasks=MAX_CONCURRENT_WORKFLOW_TASKS,
    )

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _handle_signal(_sig: int, _frame: object) -> None:
        logger.info("Shutdown signal received — stopping worker gracefully...")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "Worker started — queue=%s max_activities=%d max_workflow_tasks=%d",
        TASK_QUEUE,
        MAX_CONCURRENT_ACTIVITIES,
        MAX_CONCURRENT_WORKFLOW_TASKS,
    )

    async with worker:
        await stop_event.wait()

    logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
