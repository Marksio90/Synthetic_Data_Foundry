"""
api/routers/chatbot.py — Chatbot studio endpoints.

Endpoints:
  GET  /api/chatbot/models          List models available in local Ollama
  POST /api/chatbot/chat            Single-turn chat with loaded model
  POST /api/chatbot/eval/run        Start evaluation run (async)
  GET  /api/chatbot/eval/status/{id} Evaluation run status + metrics
  GET  /api/chatbot/eval/log/{id}   Log lines (polling)
  GET  /api/chatbot/eval/runs       List all evaluation runs
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.state import runs
from config.settings import settings
from pathlib import Path

router = APIRouter()

_OLLAMA_URL = settings.ollama_url

OUTPUT_DIR = Path(settings.output_file).parent


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    model: str = Field(..., description="Ollama model name")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=64, le=4096)
    ollama_url: str = Field(default=_OLLAMA_URL)


class EvalRequest(BaseModel):
    model_name: str = Field(..., description="Ollama model name to evaluate")
    jsonl_path: str = Field(default="", description="Path to SFT JSONL (empty = auto-detect)")
    n_samples: int = Field(default=50, ge=5, le=500)
    ollama_url: str = Field(default=_OLLAMA_URL)
    seed: int = Field(default=42)


# ---------------------------------------------------------------------------
# GET /api/chatbot/models
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models(ollama_url: str = _OLLAMA_URL) -> dict:
    """List models available in the local Ollama instance."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in data.get("models", [])
            ]
            return {"models": models, "count": len(models), "ollama_url": ollama_url}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {ollama_url}: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /api/chatbot/chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """
    Single-turn (or multi-turn) chat with a model loaded in Ollama.
    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    """
    try:
        payload = {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{req.ollama_url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer ollama"},
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        return {
            "role": "assistant",
            "content": choice["message"]["content"],
            "model": data.get("model", req.model),
            "usage": data.get("usage", {}),
            "finish_reason": choice.get("finish_reason", "stop"),
        }
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {req.ollama_url}: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /api/chatbot/eval/run — async evaluation job
# ---------------------------------------------------------------------------

@router.post("/eval/run")
async def start_eval(req: EvalRequest) -> dict:
    """Start a model evaluation run as a background task."""
    run_id = "eval-" + uuid.uuid4().hex[:8]

    jsonl_path = req.jsonl_path or str(OUTPUT_DIR / "dataset_esg_v1.jsonl")

    runs.create(run_id, f"eval:{req.model_name}")
    runs.update(run_id, status="running")
    runs.append_log(run_id, f"[Eval] Run ID: {run_id}")
    runs.append_log(run_id, f"[Eval] Model: {req.model_name}")
    runs.append_log(run_id, f"[Eval] Dataset: {jsonl_path}")
    runs.append_log(run_id, f"[Eval] Próbki: {req.n_samples}")

    asyncio.create_task(
        _run_evaluation(
            run_id=run_id,
            model_name=req.model_name,
            jsonl_path=jsonl_path,
            n_samples=req.n_samples,
            ollama_url=req.ollama_url,
            seed=req.seed,
        )
    )

    return {"run_id": run_id, "status": "running", "model": req.model_name}


async def _run_evaluation(
    run_id: str,
    model_name: str,
    jsonl_path: str,
    n_samples: int,
    ollama_url: str,
    seed: int,
) -> None:
    """Run evaluation in a thread pool to avoid blocking the event loop."""
    import asyncio
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            _eval_sync,
            run_id, model_name, jsonl_path, n_samples, ollama_url, seed,
        )
        runs.update(run_id, status="done", progress_pct=100)
        runs.update(run_id, **{"analysis": result})
        runs.append_log(run_id, f"[Eval] avg_score={result.get('avg_score', '?')}")
        runs.append_log(run_id, f"[Eval] pass_rate_088={result.get('pass_rate_088', '?')}%")
        runs.append_log(run_id, "[Eval] ✅ Ewaluacja zakończona.")
    except Exception as exc:
        runs.update(run_id, status="error", error=str(exc))
        runs.append_log(run_id, f"[Eval] ❌ Błąd: {exc}")


def _eval_sync(
    run_id: str,
    model_name: str,
    jsonl_path: str,
    n_samples: int,
    ollama_url: str,
    seed: int,
) -> dict:
    """Synchronous evaluation (called from thread pool)."""
    from training.evaluate import evaluate_model

    def _log(msg: str) -> None:
        runs.append_log(run_id, msg)

    import logging
    # Monkey-patch logger to also stream to run log
    class _RunHandler(logging.Handler):
        def emit(self, record):
            _log(f"[Eval] {record.getMessage()}")

    eval_logger = logging.getLogger("training.evaluate")
    handler = _RunHandler()
    eval_logger.addHandler(handler)
    try:
        result = evaluate_model(
            model_name=model_name,
            jsonl_path=jsonl_path,
            n_samples=n_samples,
            ollama_url=ollama_url,
            seed=seed,
        )
    finally:
        eval_logger.removeHandler(handler)

    return result


# ---------------------------------------------------------------------------
# GET /api/chatbot/eval/status/{run_id}
# ---------------------------------------------------------------------------

@router.get("/eval/status/{run_id}")
def eval_status(run_id: str) -> dict:
    rec = runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {
        "run_id": rec.run_id,
        "status": rec.status,
        "elapsed_seconds": rec.elapsed_seconds,
        "metrics": rec.analysis,
        "error": rec.error,
        "log_lines": len(rec.log_lines),
    }


@router.get("/eval/log/{run_id}")
def eval_log(run_id: str, offset: int = 0, limit: int = 200) -> dict:
    rec = runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {
        "run_id": run_id,
        "lines": rec.log_lines[offset: offset + limit],
        "total_lines": len(rec.log_lines),
        "status": rec.status,
    }


# ---------------------------------------------------------------------------
# GET /api/chatbot/eval/runs
# ---------------------------------------------------------------------------

@router.get("/eval/runs")
def list_eval_runs() -> list[dict]:
    return [
        {
            "run_id": r.run_id,
            "batch_id": r.batch_id,
            "status": r.status,
            "elapsed_seconds": r.elapsed_seconds,
            "metrics": r.analysis,
        }
        for r in runs.list_runs()
        if r.run_id.startswith("eval-")
    ]
