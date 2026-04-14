"""
api/routers/training.py — Training pipeline orchestration.

Endpoints:
  GET  /api/training/hardware     HardwareInspector preview
  POST /api/training/gate         Quality Gate check before training
  POST /api/training/run          Start full training run (subprocess)
  GET  /api/training/status/{id}  Training run status + log
  GET  /api/training/log/{id}     Log lines (polling)
  POST /api/training/export       Build client ZIP package
  GET  /api/training/runs         List all training runs
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from api.state import runs
from config.settings import settings
from training.hardware_inspector import inspect as hw_inspect
from training.quality_gate import check_dataset

router = APIRouter()

OUTPUT_DIR = Path(settings.output_file).parent
_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TrainingRunRequest(BaseModel):
    jsonl_path: str = Field(default="", description="Path to SFT JSONL (empty = auto-detect)")
    dpo_path: str = Field(default="", description="Path to DPO JSONL (empty = auto-detect)")
    base_model: Optional[str] = Field(None, description="Override auto-detected model")
    lora_rank: Optional[int] = Field(None, ge=4, le=128)
    epochs: Optional[int] = Field(None, ge=1, le=10)
    run_name: str = Field(default="foundry-model")
    skip_dpo: bool = Field(False)
    skip_eval: bool = Field(False)


class ExportRequest(BaseModel):
    model_path: str = Field(..., description="Path to trained LoRA adapter directory")
    base_model: str = Field(default="meta-llama/Llama-3.2-3B-Instruct")
    model_name: str = Field(default="foundry-domain-model")
    domain_label: str = Field(default="ESG / Prawo korporacyjne UE")
    quantization: str = Field(default="Q4_K_M")


# ---------------------------------------------------------------------------
# GET /api/training/hardware
# ---------------------------------------------------------------------------

@router.get("/hardware")
def get_hardware() -> dict:
    """Return hardware profile and recommended model."""
    hw = hw_inspect()
    return hw.as_dict()


# ---------------------------------------------------------------------------
# POST /api/training/gate
# ---------------------------------------------------------------------------

@router.post("/gate")
def run_quality_gate(
    jsonl_path: str = "",
    dpo_path: str = "",
) -> dict:
    """Run dataset quality gate. Returns pass/fail + individual check results."""
    jpath = jsonl_path or str(OUTPUT_DIR / "dataset_esg_v1.jsonl")
    dpath = dpo_path or str(OUTPUT_DIR / "dataset_esg_v1_dpo.jsonl")

    result = check_dataset(jpath, dpath)
    return {
        "passed": result.passed,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "value": str(c.value),
                "threshold": str(c.threshold),
                "message": c.message,
            }
            for c in result.checks
        ],
        "warnings": result.warnings,
    }


# ---------------------------------------------------------------------------
# POST /api/training/run — start training subprocess
# ---------------------------------------------------------------------------

@router.post("/run")
async def start_training(req: TrainingRunRequest) -> dict:
    """
    Start full training pipeline as background subprocess.
    Runs train_sft.py → train_dpo.py → evaluate.py sequentially.
    """
    run_id = "train-" + uuid.uuid4().hex[:8]

    jsonl_path = req.jsonl_path or str(OUTPUT_DIR / "dataset_esg_v1.jsonl")
    dpo_path = req.dpo_path or str(OUTPUT_DIR / "dataset_esg_v1_dpo.jsonl")
    output_dir = str(OUTPUT_DIR / "models")

    # Auto-detect hardware if no override
    hw = hw_inspect()
    base_model = req.base_model or hw.recommended_model.name
    lora_rank = req.lora_rank or hw.recommended_model.lora_rank
    batch_size = hw.recommended_model.batch_size
    grad_accum = hw.recommended_model.grad_accum

    # Auto-tune config
    from training.auto_tuner import compute_config
    import json

    n_samples = 0
    jpath = Path(jsonl_path)
    if jpath.exists():
        n_samples = sum(1 for line in jpath.open("r", encoding="utf-8") if line.strip())

    config = compute_config(
        n_samples=n_samples,
        base_model=base_model,
        lora_rank=lora_rank,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_length=hw.recommended_model.max_seq_length,
        jsonl_path=jsonl_path,
        dpo_path=dpo_path,
        output_dir=output_dir,
        run_name=req.run_name,
    )

    if req.epochs:
        config.epochs_sft = req.epochs

    rec = runs.create(run_id, f"train:{req.run_name}")
    rec.analysis = {
        "base_model": base_model,
        "lora_rank": lora_rank,
        "epochs_sft": config.epochs_sft,
        "learning_rate": config.learning_rate,
        "n_samples": n_samples,
        "gpu": hw.gpu_name,
        "reasoning": config.auto_tuner_reasoning,
    }

    # Build training commands
    sft_cmd = [
        _PYTHON, "-m", "training.train_sft",
        "--jsonl", jsonl_path,
        "--model", base_model,
        "--output-dir", output_dir,
        "--lora-rank", str(lora_rank),
        "--epochs", str(config.epochs_sft),
        "--lr", str(config.learning_rate),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--max-seq-length", str(hw.recommended_model.max_seq_length),
        "--run-name", req.run_name,
    ]

    dpo_cmd = [
        _PYTHON, "-m", "training.train_dpo",
        "--dpo-jsonl", dpo_path,
        "--sft-model", f"{output_dir}/sft",
        "--output-dir", output_dir,
        "--base-model", base_model,
        "--run-name", f"{req.run_name}-dpo",
    ]

    env = os.environ.copy()

    runs.append_log(run_id, f"[Training] Run ID: {run_id}")
    runs.append_log(run_id, f"[Training] Model: {base_model}")
    runs.append_log(run_id, f"[Training] Dataset: {n_samples} próbek")
    for r in config.auto_tuner_reasoning:
        runs.append_log(run_id, f"[AutoTuner] {r}")
    runs.append_log(run_id, "[Training] Uruchamiam SFT training...")

    asyncio.create_task(
        _run_training_pipeline(
            run_id=run_id,
            sft_cmd=sft_cmd,
            dpo_cmd=dpo_cmd if not req.skip_dpo else None,
            env=env,
        )
    )

    return {
        "run_id": run_id,
        "status": "starting",
        "base_model": base_model,
        "epochs_sft": config.epochs_sft,
        "n_samples": n_samples,
        "estimated_hours": round(
            hw.recommended_model.estimated_hours_per_1k * n_samples / 1000, 1
        ),
    }


async def _run_training_pipeline(
    run_id: str,
    sft_cmd: list[str],
    dpo_cmd: Optional[list[str]],
    env: dict,
) -> None:
    """Run SFT → DPO sequentially, streaming logs."""
    runs.update(run_id, status="running")

    async def _stream(cmd: list[str], label: str) -> bool:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            runs.append_log(run_id, raw.decode("utf-8", errors="replace").rstrip())
        await proc.wait()
        if proc.returncode != 0:
            runs.append_log(run_id, f"[{label}] ❌ Exit code {proc.returncode}")
            return False
        runs.append_log(run_id, f"[{label}] ✅ Zakończony pomyślnie")
        return True

    try:
        ok = await _stream(sft_cmd, "SFT")
        if ok and dpo_cmd:
            runs.append_log(run_id, "[Training] Uruchamiam DPO alignment...")
            ok = await _stream(dpo_cmd, "DPO")

        runs.update(run_id, status="done" if ok else "error", progress_pct=100)
        runs.append_log(run_id, "[Training] Pipeline treningowy zakończony.")
    except Exception as exc:
        runs.update(run_id, status="error", error=str(exc))
        runs.append_log(run_id, f"[Training] ❌ Wyjątek: {exc}")


# ---------------------------------------------------------------------------
# GET /api/training/status/{run_id}
# ---------------------------------------------------------------------------

@router.get("/status/{run_id}")
def training_status(run_id: str) -> dict:
    rec = runs.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {
        "run_id": rec.run_id,
        "status": rec.status,
        "elapsed_seconds": rec.elapsed_seconds,
        "config": rec.analysis,
        "error": rec.error,
        "log_lines": len(rec.log_lines),
    }


@router.get("/log/{run_id}")
def training_log(run_id: str, offset: int = 0, limit: int = 200) -> dict:
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
# GET /api/training/runs
# ---------------------------------------------------------------------------

@router.get("/runs")
def list_training_runs() -> list[dict]:
    return [
        {
            "run_id": r.run_id,
            "batch_id": r.batch_id,
            "status": r.status,
            "elapsed_seconds": r.elapsed_seconds,
        }
        for r in runs.list_runs()
        if r.run_id.startswith("train-")
    ]


# ---------------------------------------------------------------------------
# POST /api/training/export
# ---------------------------------------------------------------------------

@router.post("/export")
async def export_model(req: ExportRequest) -> dict:
    """Build client ZIP package from trained model."""
    from training.export import merge_lora, convert_to_gguf, build_client_package

    export_run_id = "export-" + uuid.uuid4().hex[:8]
    rec = runs.create(export_run_id, f"export:{req.model_name}")
    runs.update(export_run_id, status="running")

    async def _do_export():
        try:
            runs.append_log(export_run_id, "[Export] Łączę LoRA z modelem bazowym...")
            merged = merge_lora(req.base_model, req.model_path, str(OUTPUT_DIR / "models"))

            runs.append_log(export_run_id, f"[Export] Konwertuję do GGUF ({req.quantization})...")
            gguf = convert_to_gguf(merged, str(OUTPUT_DIR / "models"), req.quantization)

            runs.append_log(export_run_id, "[Export] Generuję datacard...")
            from training.datacard import generate_datacard
            jsonl = OUTPUT_DIR / "dataset_esg_v1.jsonl"
            dpo_jsonl = OUTPUT_DIR / "dataset_esg_v1_dpo.jsonl"
            datacard_path = generate_datacard(
                jsonl_path=str(jsonl),
                dpo_path=str(dpo_jsonl) if dpo_jsonl.exists() else None,
                domain_label=req.domain_label,
                base_model=req.base_model,
            )

            runs.append_log(export_run_id, "[Export] Buduję paczkę dla klienta...")
            n_records = 0
            if jsonl.exists():
                n_records = sum(1 for l in jsonl.open("r", encoding="utf-8") if l.strip())

            zip_path = build_client_package(
                gguf_path=gguf,
                model_name=req.model_name,
                output_dir=str(OUTPUT_DIR),
                base_model=req.base_model,
                domain_label=req.domain_label,
                n_records=n_records,
                datacard_path=datacard_path,
            )
            runs.update(export_run_id, status="done", progress_pct=100)
            runs.update(export_run_id, **{"analysis": {"zip_path": zip_path}})
            runs.append_log(export_run_id, f"[Export] ✅ Paczka gotowa: {zip_path}")
        except Exception as exc:
            runs.update(export_run_id, status="error", error=str(exc))
            runs.append_log(export_run_id, f"[Export] ❌ Błąd: {exc}")

    asyncio.create_task(_do_export())
    return {"export_run_id": export_run_id, "status": "started"}


# ---------------------------------------------------------------------------
# GET /api/training/export/download/{export_run_id}
# ---------------------------------------------------------------------------

@router.get("/export/download/{export_run_id}")
def download_export(export_run_id: str) -> FileResponse:
    """Download the client ZIP package built by a completed export run."""
    rec = runs.get(export_run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Export run not found: {export_run_id}")
    if rec.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Export not complete yet (status={rec.status}). Try again later.",
        )
    zip_path = (rec.analysis or {}).get("zip_path")
    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(status_code=404, detail="ZIP file not found on server.")
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=Path(zip_path).name,
    )


# ---------------------------------------------------------------------------
# POST /api/training/datacard  — generate datacard on demand
# ---------------------------------------------------------------------------

class DatacardRequest(BaseModel):
    jsonl_path: str = Field(default="", description="Path to SFT JSONL (empty = auto-detect)")
    dpo_path: str = Field(default="", description="Path to DPO JSONL (empty = auto-detect)")
    domain_label: str = Field(default="ESG / Prawo korporacyjne UE")
    base_model: str = Field(default="")


@router.post("/datacard")
def generate_datacard_endpoint(req: DatacardRequest) -> dict:
    """Generate (or regenerate) datacard.json from the current dataset."""
    from training.datacard import generate_datacard

    jsonl = req.jsonl_path or str(OUTPUT_DIR / "dataset_esg_v1.jsonl")
    dpo = req.dpo_path or str(OUTPUT_DIR / "dataset_esg_v1_dpo.jsonl")

    try:
        path = generate_datacard(
            jsonl_path=jsonl,
            dpo_path=dpo if Path(dpo).exists() else None,
            domain_label=req.domain_label,
            base_model=req.base_model,
        )
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {"status": "ok", "path": path, "datacard": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
