"""
training/train_sft.py — SFT fine-tuning z Unsloth + LoRA.

Używa biblioteki Unsloth (2-4x szybszy trening, 70% mniej VRAM niż standardowy HuggingFace).
Wejście:  dataset JSONL w formacie ChatML (pole "messages")
Wyjście:  LoRA adaptery w output_dir/sft/

Wymagania (instalowane przez docker/Dockerfile.trainer):
  unsloth[colab-new]>=2024.10
  transformers>=4.44.0
  datasets>=2.20.0
  trl>=0.10.0
  peft>=0.12.0

Uruchomienie bezpośrednie:
  python -m training.train_sft \\
    --jsonl /app/output/dataset_esg_v1.jsonl \\
    --model meta-llama/Llama-3.2-3B-Instruct \\
    --output-dir /app/output/models/sft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard: Unsloth is only available in the trainer container
# ---------------------------------------------------------------------------

def _check_deps() -> bool:
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _to_chat_format(records: list[dict]) -> list[dict]:
    """Convert JSONL records to Unsloth chat format."""
    formatted = []
    for rec in records:
        messages = rec.get("messages", [])
        if not messages:
            continue
        # Unsloth expects {"conversations": [{"role": ..., "content": ...}]}
        formatted.append({"conversations": messages})
    return formatted


def train_sft(
    jsonl_path: str,
    base_model: str,
    output_dir: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_length: int = 8192,
    warmup_steps: int = 100,
    val_split: float = 0.10,
    run_name: str = "foundry-sft",
) -> str:
    """
    Fine-tune base_model on JSONL dataset using Unsloth LoRA.

    Returns:
        Path to saved LoRA adapter directory.
    """
    if not _check_deps():
        logger.error(
            "Unsloth not installed. Run training inside docker/Dockerfile.trainer container."
        )
        sys.exit(1)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    logger.info("Loading base model: %s", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,           # auto-detect (float16 on consumer GPU)
        load_in_4bit=True,    # QLoRA: 4-bit quantization
    )

    # Apply chat template (tokenizer-specific)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load and format dataset
    logger.info("Loading dataset from: %s", jsonl_path)
    records = _load_jsonl(jsonl_path)
    logger.info("Loaded %d records", len(records))

    formatted = _to_chat_format(records)

    # Split train/val
    split_idx = int(len(formatted) * (1 - val_split))
    train_data = Dataset.from_list(formatted[:split_idx])
    val_data   = Dataset.from_list(formatted[split_idx:])
    logger.info("Train: %d | Val: %d", len(train_data), len(val_data))

    def formatting_func(examples):
        """Convert conversations to tokenized text."""
        convs = examples["conversations"]
        texts = []
        for conv in convs:
            text = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    # Training arguments
    out_path = Path(output_dir) / "sft"
    out_path.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=not _is_bf16_supported(),
        bf16=_is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if len(val_data) > 0 else "no",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        run_name=run_name,
        report_to="none",    # disable wandb
        seed=42,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data if len(val_data) > 0 else None,
        dataset_text_field="text",
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        args=args,
        packing=False,
    )

    logger.info("Starting SFT training (epochs=%d, lr=%.2e, batch=%d)...", epochs, learning_rate, batch_size)
    trainer.train()

    # Save LoRA adapters
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info("SFT complete. LoRA adapters saved to: %s", out_path)
    return str(out_path)


def _is_bf16_supported() -> bool:
    try:
        import torch
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="SFT fine-tuning with Unsloth LoRA")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--output-dir", default="/app/output/models")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    p.add_argument("--max-seq-length", type=int, default=8192,
                   help="Maximum sequence length for tokenisation and KV-cache")
    p.add_argument("--run-name", default="foundry-sft")
    args = p.parse_args()

    train_sft(
        jsonl_path=args.jsonl,
        base_model=args.model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
