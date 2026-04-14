"""
training/auto_tuner.py — Automatyczne dobieranie hiperparametrów treningu.

Reguły:
  - Małe datasety (< 1000 próbek) → więcej epok, mniejszy LR
  - Duże datasety (> 5000 próbek) → mniej epok, wyższy LR
  - LR skalowany odwrotnie do rozmiaru modelu
  - early_stopping zawsze włączony (patience = 2 epoki bez poprawy)
  - Checkpoint co 0.5 epoki

Wynik: TrainingConfig gotowy do przekazania do train_sft.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Dataset
    jsonl_path: str = ""
    dpo_path: str = ""
    train_split: float = 0.80
    val_split: float = 0.20

    # Model
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_seq_length: int = 8192

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training loop
    epochs_sft: int = 3
    epochs_dpo: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 2   # epochs

    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10

    # Output
    output_dir: str = "/app/output/models"
    run_name: str = "foundry-sft"

    # Reasoning
    auto_tuner_reasoning: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "auto_tuner_reasoning"}


def compute_config(
    n_samples: int,
    base_model: str,
    lora_rank: int,
    batch_size: int,
    grad_accum: int,
    max_seq_length: int,
    jsonl_path: str = "",
    dpo_path: str = "",
    output_dir: str = "/app/output/models",
    run_name: str = "foundry-sft",
) -> TrainingConfig:
    """
    Compute optimal training hyperparameters from dataset size and model info.
    All decisions are deterministic heuristics — no LLM calls.
    """
    reasoning: list[str] = []

    # ── Epochs ───────────────────────────────────────────────────────────────
    if n_samples < 500:
        epochs = 5
        reasoning.append(f"Mały dataset ({n_samples}) → 5 epok")
    elif n_samples < 1000:
        epochs = 4
        reasoning.append(f"Mały-średni dataset ({n_samples}) → 4 epoki")
    elif n_samples < 3000:
        epochs = 3
        reasoning.append(f"Standardowy dataset ({n_samples}) → 3 epoki")
    elif n_samples < 8000:
        epochs = 2
        reasoning.append(f"Duży dataset ({n_samples}) → 2 epoki (unikamy overfittingu)")
    else:
        epochs = 1
        reasoning.append(f"Bardzo duży dataset ({n_samples}) → 1 epoka")

    # ── Learning rate ────────────────────────────────────────────────────────
    # Smaller models tolerate higher LR; scale inversely with param count
    if "70B" in base_model or "70b" in base_model:
        lr = 5e-5
        reasoning.append("70B model → lr=5e-5 (ostrożniejszy trening)")
    elif "8B" in base_model or "8b" in base_model:
        lr = 1e-4
        reasoning.append("8B model → lr=1e-4")
    else:
        lr = 2e-4
        reasoning.append("≤4B model → lr=2e-4")

    # Adjust LR for large datasets (avoid divergence)
    if n_samples > 5000:
        lr *= 0.5
        reasoning.append(f"Duży dataset → lr zmniejszony o 50% (={lr:.1e})")

    # ── Warmup ───────────────────────────────────────────────────────────────
    steps_per_epoch = max(n_samples // (batch_size * grad_accum), 1)
    warmup = min(int(steps_per_epoch * 0.1), 200)
    reasoning.append(f"Warmup: {warmup} kroków ({steps_per_epoch} kroków/epoka)")

    # ── Eval/save frequency ──────────────────────────────────────────────────
    save_steps = max(steps_per_epoch // 2, 10)
    eval_steps = max(steps_per_epoch // 4, 5)

    # ── LoRA alpha ───────────────────────────────────────────────────────────
    lora_alpha = lora_rank * 2

    return TrainingConfig(
        jsonl_path=jsonl_path,
        dpo_path=dpo_path,
        base_model=base_model,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        epochs_sft=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        warmup_steps=warmup,
        save_steps=save_steps,
        eval_steps=eval_steps,
        output_dir=output_dir,
        run_name=run_name,
        auto_tuner_reasoning=reasoning,
    )
