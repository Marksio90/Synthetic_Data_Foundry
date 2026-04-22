"""
api/pipeline_worker.py — dedicated pipeline queue worker process.

Runs the queue executor without starting FastAPI server. Intended for separate
worker deployment/containers with SERVICE_ROLE=worker.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from api.bootstrap import setup_logging
from api.routers.pipeline import shutdown_executor, startup_executor
from config.settings import settings


async def run_worker() -> None:
    logger = setup_logging()
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def _stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            # Windows compatibility
            signal.signal(sig, lambda *_: _stop())

    logger.info("Starting dedicated pipeline worker (SERVICE_ROLE=%s)...", settings.service_role)
    await startup_executor()
    logger.info("Pipeline worker online. Waiting for stop signal...")

    await stop_event.wait()

    logger.info("Stopping pipeline worker...")
    await shutdown_executor()
    logger.info("Pipeline worker stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
