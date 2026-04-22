"""
training/train_dpo.py — DPO alignment na parach preference.

Wejście:  LoRA adaptery z train_sft.py + plik DPO JSONL
          Format DPO: {"prompt": [...], "chosen": [...], "rejected": [...]}
Wyjście:  Zaktualizowane LoRA adaptery w output_dir/dpo/

Używa TRL DPOTrainer (HuggingFace) z modelem po SFT.
Czas: ~30-60 min dla 500 par na RTX 4090.

Uruchomienie bezpośrednie:
  python -m training.train_dpo \\
    --dpo-jsonl /app/output/dataset_esg_v1_dpo.jsonl \\
    --sft-model /app/output/models/sft \\
    --output-dir /app/output/models/dpo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from training.mlflow_tracker import FoundryMLflowTracker

logger = logging.getLogger(__name__)


def _check_deps() -> bool:
    try:
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


def _load_dpo_jsonl(path: str) -> list[dict]:
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


def train_dpo(
    dpo_jsonl_path: str,
    sft_model_path: str,
    output_dir: str,
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    epochs: float = 1.0,
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    grad_accum: int = 4,
    beta: float = 0.1,
    max_length: int = 2048,
    run_name: str = "foundry-dpo",
    batch_id: str = "manual",
) -> str:
    """
    DPO alignment from SFT model using TRL DPOTrainer.

    Returns:
        Path to saved DPO model directory.
    """
    if not _check_deps():
        logger.error("TRL not installed. Run inside docker/Dockerfile.trainer container.")
        sys.exit(1)

    from trl import DPOConfig, DPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from datasets import Dataset

    logger.info("Loading SFT model from: %s", sft_model_path)

    # Load base model + SFT LoRA adapters
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, sft_model_path)
    model.train()

    # Load DPO dataset
    logger.info("Loading DPO pairs from: %s", dpo_jsonl_path)
    records = _load_dpo_jsonl(dpo_jsonl_path)
    logger.info("Loaded %d DPO pairs", len(records))

    if len(records) < 10:
        logger.warning("Too few DPO pairs (%d) — skipping DPO training", len(records))
        return sft_model_path

    # Convert to TRL DPO format
    def _msgs_to_text(msgs: list[dict], tokenizer_obj) -> str:
        return tokenizer_obj.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    dpo_data = []
    for rec in records:
        prompt_msgs = rec.get("prompt", [])
        chosen_msgs = rec.get("chosen", [])
        rejected_msgs = rec.get("rejected", [])
        dpo_data.append({
            "prompt": _msgs_to_text(prompt_msgs, tokenizer),
            "chosen": _msgs_to_text(chosen_msgs, tokenizer),
            "rejected": _msgs_to_text(rejected_msgs, tokenizer),
        })

    dataset = Dataset.from_list(dpo_data)

    # Output directory
    out_path = Path(output_dir) / "dpo"
    out_path.mkdir(parents=True, exist_ok=True)

    # DPO config
    dpo_config = DPOConfig(
        output_dir=str(out_path),
        num_train_epochs=int(epochs),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        run_name=run_name,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # use implicit reference (PEFT handles this)
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=dataset,
    )

    tracker = FoundryMLflowTracker(experiment_name="foundry-dpo")
    with tracker.start_run(run_name=run_name, batch_id=batch_id, tags={"sft_model": sft_model_path}):
        tracker.log_hyperparams(
            base_model=base_model,
            sft_model=sft_model_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            grad_accum=grad_accum,
            beta=beta,
            max_length=max_length,
            dpo_pairs=len(records),
        )

        logger.info("Starting DPO training (beta=%.2f, epochs=%s)...", beta, epochs)
        trainer.train()

        for entry in (trainer.state.log_history or []):
            step = int(entry.get("step", 0))
            for key, val in entry.items():
                if isinstance(val, (int, float)) and key not in ("step", "epoch", "total_flos"):
                    tracker.log_metric(key, float(val), step=step)

        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
        logger.info("DPO complete. Model saved to: %s", out_path)

        tracker.log_model_artifact(str(out_path))
        tracker.register_model(
            model_name="foundry-dpo",
            model_version_tags={"batch_id": batch_id, "base_model": base_model},
        )

    return str(out_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="DPO alignment with TRL DPOTrainer")
    p.add_argument("--dpo-jsonl", required=True)
    p.add_argument("--sft-model", required=True)
    p.add_argument("--output-dir", default="/app/output/models")
    p.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--run-name", default="foundry-dpo")
    args = p.parse_args()

    train_dpo(
        dpo_jsonl_path=args.dpo_jsonl,
        sft_model_path=args.sft_model,
        output_dir=args.output_dir,
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
