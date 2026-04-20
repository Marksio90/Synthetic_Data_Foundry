"""
agents/hf_uploader.py — HuggingFace Hub Auto-Uploader

Automatycznie uploaduje:
  1. Dataset SFT (JSONL) → HuggingFace Hub dataset repo
  2. Dataset DPO/ORPO/KTO (JSONL) → HuggingFace Hub dataset repo
  3. Dataset card (README.md) z metadanymi

Wymagania:
  pip install huggingface_hub datasets

Użycie:
  from agents.hf_uploader import upload_dataset_to_hub
  upload_dataset_to_hub(
      sft_path="/app/output/dataset.jsonl",
      repo_id="org/my-esg-dataset",
      batch_id="esg-v1",
  )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


def _build_dataset_card(
    repo_id: str,
    batch_id: str,
    sft_count: int,
    dpo_count: int,
    orpo_count: int,
    kto_count: int,
    datacard: Optional[dict] = None,
) -> str:
    """Generuje README.md dla HuggingFace Hub dataset repo."""
    card = datacard or {}
    avg_score = card.get("avg_quality_score", 0.0)
    cross_pct = card.get("cross_doc_pct", 0.0)
    adv_pct = card.get("adversarial_pct", 0.0)

    return f"""---
license: cc-by-4.0
language:
- pl
- en
tags:
- esg
- legal
- synthetic
- instruction-tuning
- dpo
- orpo
- kto
dataset_info:
  splits:
  - name: train
    num_examples: {sft_count}
---

# ESG Synthetic Data Foundry — {batch_id}

Automatycznie wygenerowany dataset niszowych danych Q&A z dyrektyw UE ESG (CSRD, SFDR, Taksonomia, CSDDD).

## Statystyki

| Format | Rekordy |
|--------|---------|
| SFT (ChatML) | {sft_count} |
| DPO preference pairs | {dpo_count} |
| ORPO preference pairs | {orpo_count} |
| KTO labeled pairs | {kto_count} |

- **Średnia jakość (judge score):** {avg_score:.3f}
- **Cross-document Q&A:** {cross_pct:.1f}%
- **Adversarial (odmowy):** {adv_pct:.1f}%

## Perspektywy eksperckie

CFO | Prawnik | Audytor ESG | Analityk finansowy | Regulator | Akademik | Dziennikarz | Inwestor

## Formaty treningowe

- **SFT:** ChatML format (OpenAI fine-tuning compatible)
- **DPO/ORPO:** `prompt` + `chosen` + `rejected` (TRL DPOTrainer/ORPOTrainer)
- **KTO:** `prompt` + `completion` + `label` (TRL KTOTrainer)

## Generacja

Pipeline: LangGraph multi-turn → Constitutional AI → GPT-4o-mini Judge → MinHash dedup → B2B watermark

Batch ID: `{batch_id}`
"""


def _count_jsonl_lines(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for line in p.open("r", encoding="utf-8") if line.strip())


def upload_dataset_to_hub(
    sft_path: str,
    batch_id: str,
    dpo_path: Optional[str] = None,
    orpo_path: Optional[str] = None,
    kto_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    datacard: Optional[dict] = None,
    private: bool = True,
) -> bool:
    """
    Uploaduje dataset do HuggingFace Hub.

    Args:
        sft_path:   ścieżka do pliku JSONL SFT
        batch_id:   identyfikator batchu
        dpo_path:   ścieżka do pliku JSONL DPO (opcjonalny)
        orpo_path:  ścieżka do pliku JSONL ORPO (opcjonalny)
        kto_path:   ścieżka do pliku JSONL KTO (opcjonalny)
        repo_id:    HF Hub repo (domyślnie z settings.hf_dataset_repo)
        datacard:   słownik z metadanymi datasetu
        private:    czy repo ma być prywatne

    Returns:
        True jeśli sukces, False jeśli błąd lub brak konfiguracji
    """
    token = settings.hf_token
    repo = repo_id or settings.hf_dataset_repo

    if not token or not repo:
        logger.info("HF upload skipped: HF_TOKEN or HF_DATASET_REPO not configured")
        return False

    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore
    except ImportError:
        logger.warning("huggingface_hub not installed: pip install huggingface_hub")
        return False

    api = HfApi(token=token)

    # Utwórz repo jeśli nie istnieje
    try:
        create_repo(repo_id=repo, repo_type="dataset", private=private, token=token, exist_ok=True)
        logger.info("HF Hub repo ready: %s", repo)
    except Exception as exc:
        logger.error("Failed to create HF repo '%s': %s", repo, exc)
        return False

    sft_count = _count_jsonl_lines(sft_path)
    dpo_count = _count_jsonl_lines(dpo_path) if dpo_path else 0
    orpo_count = _count_jsonl_lines(orpo_path) if orpo_path else 0
    kto_count = _count_jsonl_lines(kto_path) if kto_path else 0

    # Upload plików
    files_to_upload: list[tuple[str, str]] = [(sft_path, "data/train_sft.jsonl")]
    if dpo_path and Path(dpo_path).exists():
        files_to_upload.append((dpo_path, "data/train_dpo.jsonl"))
    if orpo_path and Path(orpo_path).exists():
        files_to_upload.append((orpo_path, "data/train_orpo.jsonl"))
    if kto_path and Path(kto_path).exists():
        files_to_upload.append((kto_path, "data/train_kto.jsonl"))

    uploaded = 0
    for local_path, hub_path in files_to_upload:
        if not Path(local_path).exists():
            logger.warning("File not found, skipping: %s", local_path)
            continue
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hub_path,
                repo_id=repo,
                repo_type="dataset",
                token=token,
            )
            logger.info("✓ Uploaded %s → %s/%s", Path(local_path).name, repo, hub_path)
            uploaded += 1
        except Exception as exc:
            logger.error("Failed to upload %s: %s", local_path, exc)

    # Upload dataset card
    card_content = _build_dataset_card(
        repo_id=repo,
        batch_id=batch_id,
        sft_count=sft_count,
        dpo_count=dpo_count,
        orpo_count=orpo_count,
        kto_count=kto_count,
        datacard=datacard,
    )
    try:
        api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            token=token,
        )
        logger.info("✓ Dataset card uploaded to %s", repo)
    except Exception as exc:
        logger.warning("Failed to upload dataset card: %s", exc)

    logger.info(
        "HF Hub upload complete: %d/%d files → %s",
        uploaded, len(files_to_upload), repo,
    )
    return uploaded > 0


def upload_model_to_hub(
    model_path: str,
    repo_id: Optional[str] = None,
    private: bool = True,
) -> bool:
    """
    Uploaduje wytrenowany model (LoRA adapter lub pełny model) do HF Hub.

    Args:
        model_path: ścieżka do katalogu z modelem
        repo_id:    HF Hub model repo
        private:    czy repo ma być prywatne
    """
    token = settings.hf_token
    repo = repo_id or settings.hf_model_repo

    if not token or not repo:
        logger.info("Model upload skipped: HF_TOKEN or HF_MODEL_REPO not configured")
        return False

    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore
    except ImportError:
        logger.warning("huggingface_hub not installed: pip install huggingface_hub")
        return False

    model_dir = Path(model_path)
    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_path)
        return False

    api = HfApi(token=token)

    try:
        create_repo(repo_id=repo, repo_type="model", private=private, token=token, exist_ok=True)
    except Exception as exc:
        logger.error("Failed to create model repo '%s': %s", repo, exc)
        return False

    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo,
            repo_type="model",
            token=token,
        )
        logger.info("✓ Model uploaded to %s", repo)
        return True
    except Exception as exc:
        logger.error("Model upload failed: %s", exc)
        return False
