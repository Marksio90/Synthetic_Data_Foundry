"""
training/export.py — Eksport modelu do GGUF + paczka dla klienta.

Kroki:
  1. Merge LoRA adaptery z modelem bazowym (pełne wagi)
  2. Kwantyzacja do GGUF Q4_K_M przez llama.cpp
  3. Wygeneruj Ollama Modelfile z system promptem
  4. Zbuduj paczkę ZIP dla klienta (docker-compose + model + README)

Wymagania:
  - llama.cpp (convert_hf_to_gguf.py + quantize) dostępne jako llama-cpp-python
    lub w $PATH (w docker/Dockerfile.trainer)
  - Ollama CLI w $PATH (opcjonalne — do push lokalnie)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Merge LoRA + base model
# ---------------------------------------------------------------------------

def merge_lora(
    base_model: str,
    lora_path: str,
    output_dir: str,
) -> str:
    """
    Merge LoRA adapters into the base model and save full weights.
    Returns path to merged model directory.
    """
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("peft/transformers not installed.")
        sys.exit(1)

    merge_path = Path(output_dir) / "merged"
    merge_path.mkdir(parents=True, exist_ok=True)

    logger.info("Merging LoRA adapters into %s...", base_model)
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path,
        device_map="cpu",          # merge on CPU (less VRAM)
        torch_dtype="auto",
    )
    model = model.merge_and_unload()
    model.save_pretrained(str(merge_path))

    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.save_pretrained(str(merge_path))

    logger.info("Merged model saved to: %s", merge_path)
    return str(merge_path)


# ---------------------------------------------------------------------------
# Step 2: Convert to GGUF + quantize
# ---------------------------------------------------------------------------

def convert_to_gguf(
    merged_model_path: str,
    output_dir: str,
    quantization: str = "Q4_K_M",
) -> str:
    """
    Convert merged HuggingFace model to GGUF format using llama.cpp.
    Returns path to the .gguf file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    gguf_f32 = out_path / "model_f32.gguf"
    gguf_final = out_path / f"model_{quantization.lower()}.gguf"

    # Try using llama-cpp-python's convert utility
    convert_script = _find_convert_script()

    if convert_script:
        logger.info("Converting to GGUF (F32) using: %s", convert_script)
        result = subprocess.run(
            [sys.executable, str(convert_script),
             merged_model_path,
             "--outfile", str(gguf_f32),
             "--outtype", "f32"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.error("GGUF conversion failed: %s", result.stderr[:500])
            raise RuntimeError("GGUF conversion failed")

        # Quantize
        quantize_bin = _find_quantize_binary()
        if quantize_bin and gguf_f32.exists():
            logger.info("Quantizing to %s...", quantization)
            result = subprocess.run(
                [str(quantize_bin), str(gguf_f32), str(gguf_final), quantization],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0 and gguf_final.exists():
                gguf_f32.unlink(missing_ok=True)  # cleanup F32
                logger.info("Quantized GGUF saved: %s (%.1f GB)",
                           gguf_final, gguf_final.stat().st_size / 1e9)
                return str(gguf_final)
    else:
        logger.warning(
            "llama.cpp convert script not found. "
            "Saving merged model directory instead of GGUF."
        )
        return merged_model_path

    return str(gguf_f32) if gguf_f32.exists() else merged_model_path


def _find_convert_script() -> Optional[Path]:
    """Find llama.cpp convert_hf_to_gguf.py script."""
    candidates = [
        Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
        Path("/usr/local/lib/llama_cpp/convert_hf_to_gguf.py"),
        Path(shutil.which("convert_hf_to_gguf") or ""),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_quantize_binary() -> Optional[Path]:
    q = shutil.which("llama-quantize") or shutil.which("quantize")
    return Path(q) if q else None


# ---------------------------------------------------------------------------
# Step 3: Generate Ollama Modelfile
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PL = (
    "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
    "Odpowiadasz wyłącznie na podstawie dokumentów, na których zostałeś wytrenowany. "
    "Jeśli pytanie wykracza poza Twoją wiedzę, odpowiedz: \"Brak danych w zbiorze wiedzy.\""
)


def generate_modelfile(
    gguf_path: str,
    model_name: str,
    system_prompt: str = _SYSTEM_PROMPT_PL,
    context_length: int = 8192,
) -> str:
    """Generate Ollama Modelfile content."""
    gguf_filename = Path(gguf_path).name
    return f"""FROM ./{gguf_filename}

PARAMETER num_ctx {context_length}
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"{system_prompt}\"\"\"

# Model: {model_name}
# Generated by Foundry Studio
# Date: {datetime.now().strftime('%Y-%m-%d')}
"""


# ---------------------------------------------------------------------------
# Step 4: Build client ZIP package
# ---------------------------------------------------------------------------

_CLIENT_DOCKER_COMPOSE = """\
# =============================================================================
# {model_name} — Foundry Chatbot
# Uruchomienie: docker compose up -d
# Interfejs:    http://localhost:3000
# =============================================================================
version: "3.9"

services:
  ollama:
    image: ollama/ollama:latest
    container_name: chatbot_ollama
    volumes:
      - ./models:/root/.ollama/models
    ports:
      - "11434:11434"
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: chatbot_ui
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      WEBUI_AUTH: "False"          # wyłączone logowanie (lokalne wdrożenie)
    ports:
      - "3000:8080"
    depends_on:
      - ollama
    restart: unless-stopped
    volumes:
      - webui_data:/app/backend/data

  setup:
    image: ollama/ollama:latest
    depends_on:
      - ollama
    volumes:
      - ./models:/models
      - ./Modelfile:/Modelfile
    entrypoint: >
      sh -c "
        sleep 5 &&
        ollama create {model_name} -f /Modelfile &&
        echo 'Model załadowany: {model_name}'
      "
    restart: "no"

volumes:
  webui_data:
"""

_CLIENT_README = """\
# {model_name} — Chatbot AI

## Informacje o modelu

- **Nazwa:** {model_name}
- **Baza:** {base_model}
- **Domena:** {domain_label}
- **Wytrenowany:** {date}
- **Zbiór danych:** {n_records} Q&A pairs

## Zawartość paczki

```
{model_name_safe}/
├── docker-compose.yml  — definicja serwisów
├── Modelfile           — konfiguracja modelu Ollama
├── {gguf_filename}     — model w formacie GGUF (~{size_gb:.1f} GB)
├── datacard.json       — statystyki i jakość datasetu
└── README.md           — ta instrukcja
```

## Uruchomienie

**Wymagania:** Docker Desktop + 8 GB RAM (minimum)

```bash
# Rozpakuj i uruchom:
cd {model_name_safe}
docker compose up -d

# Poczekaj ~60 sekund na załadowanie modelu, następnie otwórz:
# http://localhost:3000
```

## Zatrzymanie

```bash
docker compose down
```

## Co model wie

Model odpowiada na pytania dotyczące:
{domain_label}

Zakres wiedzy jest ograniczony do dokumentów użytych podczas treningu.
Na pytania spoza zakresu model odpowie: "Brak danych w zbiorze wiedzy."

## Wsparcie techniczne

Paczka wygenerowana przez Foundry Studio.
"""


def build_client_package(
    gguf_path: str,
    model_name: str,
    output_dir: str,
    base_model: str = "Llama-3.2-3B",
    domain_label: str = "ESG / Prawo korporacyjne UE",
    n_records: int = 0,
    datacard_path: Optional[str] = None,
    system_prompt: str = _SYSTEM_PROMPT_PL,
) -> str:
    """
    Build a client delivery ZIP package.

    Returns:
        Path to the generated ZIP file.
    """
    model_name_safe = model_name.replace("/", "-").replace(" ", "_")
    gguf_file = Path(gguf_path)
    size_gb = gguf_file.stat().st_size / 1e9 if gguf_file.exists() else 0.0

    # Temp directory for package contents
    tmp_dir = Path(output_dir) / "pkg_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Copy GGUF
    if gguf_file.exists():
        shutil.copy2(gguf_path, tmp_dir / gguf_file.name)
    else:
        logger.warning("GGUF file not found: %s", gguf_path)

    # Modelfile
    modelfile_content = generate_modelfile(
        gguf_path=gguf_path,
        model_name=model_name,
        system_prompt=system_prompt,
    )
    (tmp_dir / "Modelfile").write_text(modelfile_content, encoding="utf-8")

    # docker-compose.yml
    (tmp_dir / "docker-compose.yml").write_text(
        _CLIENT_DOCKER_COMPOSE.format(model_name=model_name),
        encoding="utf-8",
    )

    # README.md
    (tmp_dir / "README.md").write_text(
        _CLIENT_README.format(
            model_name=model_name,
            model_name_safe=model_name_safe,
            base_model=base_model,
            domain_label=domain_label,
            date=datetime.now().strftime("%Y-%m-%d"),
            n_records=n_records,
            gguf_filename=gguf_file.name,
            size_gb=size_gb,
        ),
        encoding="utf-8",
    )

    # datacard.json
    if datacard_path and Path(datacard_path).exists():
        shutil.copy2(datacard_path, tmp_dir / "datacard.json")

    # Build ZIP
    zip_path = Path(output_dir) / f"{model_name_safe}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in tmp_dir.iterdir():
            zf.write(file_path, arcname=f"{model_name_safe}/{file_path.name}")

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Client package built: %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
    return str(zip_path)
