"""
training/hardware_inspector.py — Automatyczna detekcja sprzętu GPU/CPU.

Sprawdza dostępne zasoby i rekomenduje:
  - Model bazowy (Phi-4-Mini / Llama-3.2-3B / Llama-3.1-8B / Llama-3.1-70B)
  - Parametry LoRA (rank, alpha, batch_size)
  - Fallback na CPU fine-tuning przez API (Together.ai / OpenAI) gdy brak GPU

Nie wymaga żadnych zewnętrznych bibliotek poza subprocess do nvidia-smi.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model profiles ordered by VRAM requirement (ascending)
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    name: str                  # HuggingFace model ID
    short_name: str            # display name
    vram_required_gb: float    # minimum VRAM for LoRA fine-tuning
    lora_rank: int
    lora_alpha: int
    batch_size: int
    grad_accum: int            # gradient accumulation steps
    max_seq_length: int
    estimated_hours_per_1k: float  # training hours per 1000 samples on RTX 4090


_MODEL_PROFILES: list[ModelProfile] = [
    ModelProfile(
        name="microsoft/Phi-4-mini-instruct",
        short_name="Phi-4-Mini (3.8B)",
        vram_required_gb=5.0,
        lora_rank=8, lora_alpha=16,
        batch_size=2, grad_accum=8,
        max_seq_length=4096,
        estimated_hours_per_1k=0.8,
    ),
    ModelProfile(
        name="meta-llama/Llama-3.2-3B-Instruct",
        short_name="Llama-3.2-3B",
        vram_required_gb=7.0,
        lora_rank=16, lora_alpha=32,
        batch_size=4, grad_accum=4,
        max_seq_length=8192,
        estimated_hours_per_1k=1.0,
    ),
    ModelProfile(
        name="meta-llama/Llama-3.1-8B-Instruct",
        short_name="Llama-3.1-8B",
        vram_required_gb=12.0,
        lora_rank=32, lora_alpha=64,
        batch_size=8, grad_accum=2,
        max_seq_length=8192,
        estimated_hours_per_1k=2.5,
    ),
    ModelProfile(
        name="meta-llama/Llama-3.1-70B-Instruct",
        short_name="Llama-3.1-70B",
        vram_required_gb=48.0,
        lora_rank=64, lora_alpha=128,
        batch_size=2, grad_accum=16,
        max_seq_length=8192,
        estimated_hours_per_1k=12.0,
    ),
]

_API_FALLBACK = ModelProfile(
    name="gpt-3.5-turbo",
    short_name="API fine-tune (no GPU)",
    vram_required_gb=0,
    lora_rank=0, lora_alpha=0,
    batch_size=16, grad_accum=1,
    max_seq_length=4096,
    estimated_hours_per_1k=0.5,
)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _get_gpu_vram_gb() -> float:
    """Return total VRAM in GB from nvidia-smi, or 0.0 if no GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # Sum up all GPUs (multi-GPU support)
        total_mib = sum(int(line.strip()) for line in out.splitlines() if line.strip())
        return round(total_mib / 1024, 1)
    except Exception:
        return 0.0


def _get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=10, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out.splitlines()[0].strip() if out else "Unknown GPU"
    except Exception:
        return "No GPU"


def _get_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1_048_576, 1)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    gpu_name: str
    vram_gb: float
    ram_gb: float
    has_gpu: bool
    recommended_model: ModelProfile
    reasoning: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"GPU: {self.gpu_name} ({self.vram_gb:.0f} GB VRAM)" if self.has_gpu else "GPU: brak",
            f"RAM: {self.ram_gb:.0f} GB",
            f"Rekomendowany model: {self.recommended_model.short_name}",
            f"LoRA rank: {self.recommended_model.lora_rank}",
            f"Batch size: {self.recommended_model.batch_size}",
        ]
        for r in self.reasoning:
            lines.append(f"  → {r}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
            "has_gpu": self.has_gpu,
            "model_name": self.recommended_model.name,
            "model_short_name": self.recommended_model.short_name,
            "lora_rank": self.recommended_model.lora_rank,
            "lora_alpha": self.recommended_model.lora_alpha,
            "batch_size": self.recommended_model.batch_size,
            "grad_accum": self.recommended_model.grad_accum,
            "max_seq_length": self.recommended_model.max_seq_length,
            "estimated_hours_per_1k": self.recommended_model.estimated_hours_per_1k,
        }


def inspect() -> HardwareProfile:
    """
    Detect hardware and recommend the best fitting model.
    Returns HardwareProfile with recommended ModelProfile.
    """
    vram = _get_gpu_vram_gb()
    gpu_name = _get_gpu_name()
    ram = _get_ram_gb()
    has_gpu = vram > 0

    reasoning: list[str] = []

    if not has_gpu:
        reasoning.append("Brak GPU — rekomendowany fallback na API fine-tuning")
        return HardwareProfile(
            gpu_name="No GPU",
            vram_gb=0,
            ram_gb=ram,
            has_gpu=False,
            recommended_model=_API_FALLBACK,
            reasoning=reasoning,
        )

    reasoning.append(f"Wykryto GPU: {gpu_name} ({vram:.0f} GB VRAM)")

    # Leave 2GB headroom for OS + other processes
    usable_vram = vram - 2.0

    # Find the best (largest) model that fits
    recommended = _MODEL_PROFILES[0]  # fallback: smallest
    for profile in reversed(_MODEL_PROFILES):
        if profile.vram_required_gb <= usable_vram:
            recommended = profile
            break

    reasoning.append(
        f"Dostępny VRAM (po marginesie 2GB): {usable_vram:.0f} GB "
        f"→ wybrany: {recommended.short_name}"
    )
    reasoning.append(
        f"LoRA rank={recommended.lora_rank}, batch_size={recommended.batch_size}, "
        f"grad_accum={recommended.grad_accum}"
    )

    logger.info("HardwareInspector: %s → %s", gpu_name, recommended.short_name)
    return HardwareProfile(
        gpu_name=gpu_name,
        vram_gb=vram,
        ram_gb=ram,
        has_gpu=True,
        recommended_model=recommended,
        reasoning=reasoning,
    )
