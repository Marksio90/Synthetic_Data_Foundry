#!/usr/bin/env bash
# =============================================================================
# scripts/foundry-ctl.sh — Foundry Studio controller
#
# Wywoływany przez Makefile:
#   make setup | up | status | generate | gate | train | export | chatbot | full-run
#
# Zmienne środowiskowe (opcjonalne):
#   CHUNK_LIMIT=50      — ogranicz generowanie do N chunków (domyślnie: bez limitu)
#   BATCH_ID=moj-run    — identyfikator batcha (domyślnie: auto)
#   MODEL_NAME=mój-model — nazwa modelu przy eksporcie (domyślnie: esg-model)
#   BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct — model bazowy do treningu
#   SKIP_DPO=1          — pomiń DPO alignment
#   DATA_DIR=/app/data/esg — podfolder z PDFami (domyślnie: /app/data)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Kolory
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}▶ $*${NC}"; }
success() { echo -e "${GREEN}✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $*${NC}"; }
error()   { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ---------------------------------------------------------------------------
# Domyślne zmienne
# ---------------------------------------------------------------------------
CHUNK_LIMIT="${CHUNK_LIMIT:-0}"
BATCH_ID="${BATCH_ID:-}"
MODEL_NAME="${MODEL_NAME:-esg-model}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
SKIP_DPO="${SKIP_DPO:-0}"
DATA_DIR="${DATA_DIR:-/app/data}"

SFT_JSONL="/app/output/dataset_esg_v1.jsonl"
DPO_JSONL="/app/output/dataset_esg_v1_dpo.jsonl"
OUTPUT_DIR="/app/output/models"

# ---------------------------------------------------------------------------
# setup — pierwsze uruchomienie
# ---------------------------------------------------------------------------
cmd_setup() {
    info "Foundry Studio — setup"

    # Sprawdź docker
    command -v docker &>/dev/null || error "Docker nie jest zainstalowany."
    docker compose version &>/dev/null || error "Docker Compose V2 nie jest dostępny."

    # Skopiuj .env
    if [ ! -f .env ]; then
        cp .env.example .env
        success "Skopiowano .env.example → .env"
        warn "Uzupełnij klucze API w pliku .env przed uruchomieniem!"
    else
        success ".env już istnieje"
    fi

    # Utwórz foldery
    mkdir -p data output
    success "Foldery data/ i output/ gotowe"

    # Zbuduj obrazy
    info "Budowanie obrazów Docker..."
    docker compose build
    success "Obrazy zbudowane"

    echo ""
    echo -e "${BOLD}Gotowe. Następny krok:${NC}"
    echo "  1. Edytuj .env — wpisz OPENAI_API_KEY i inne klucze"
    echo "  2. Wrzuć PDFy do folderu data/"
    echo "  3. make up        ← uruchom core stack"
    echo "  4. make generate  ← uruchom pipeline"
}

# ---------------------------------------------------------------------------
# up — uruchom core stack (postgres + API + UI)
# ---------------------------------------------------------------------------
cmd_up() {
    info "Uruchamiam core stack (postgres + foundry-api + foundry-ui)..."
    docker compose up -d
    echo ""
    info "Czekam aż serwisy będą gotowe..."
    sleep 3
    cmd_status
    echo ""
    success "Stack gotowy"
    echo -e "  UI:     ${CYAN}http://localhost:8501${NC}"
    echo -e "  API:    ${CYAN}http://localhost:8080/docs${NC}"
}

# ---------------------------------------------------------------------------
# status — health check
# ---------------------------------------------------------------------------
cmd_status() {
    info "Status serwisów:"
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null \
        || docker compose ps
}

# ---------------------------------------------------------------------------
# generate — uruchom pipeline Q&A
# ---------------------------------------------------------------------------
cmd_generate() {
    info "Uruchamiam pipeline generowania Q&A..."

    ARGS=()

    # DATA_DIR lub domyślnie /app/data
    ARGS+=("--data-dir" "$DATA_DIR")

    # BATCH_ID (opcjonalne)
    if [ -n "$BATCH_ID" ]; then
        ARGS+=("--batch-id" "$BATCH_ID")
    fi

    # CHUNK_LIMIT (opcjonalne)
    if [ "$CHUNK_LIMIT" -gt 0 ] 2>/dev/null; then
        ARGS+=("--chunk-limit" "$CHUNK_LIMIT")
        warn "Limit: $CHUNK_LIMIT chunków"
    fi

    info "Komenda: docker compose --profile pipeline run --rm app ${ARGS[*]}"
    docker compose --profile pipeline run --rm app "${ARGS[@]}"
    success "Pipeline zakończony"
}

# ---------------------------------------------------------------------------
# gate — sprawdź jakość datasetu (Quality Gate)
# ---------------------------------------------------------------------------
cmd_gate() {
    info "Sprawdzam jakość datasetu (Quality Gate)..."
    docker compose --profile train run --rm foundry-trainer \
        python -c "
import sys, json
from training.quality_gate import check_dataset
result = check_dataset(
    jsonl_path='$SFT_JSONL',
    dpo_jsonl_path='$DPO_JSONL',
)
print()
for c in result.checks:
    mark = '✓' if c.passed else '✗'
    print(f'  {mark}  {c.name}: {c.message}')
if result.warnings:
    print()
    for w in result.warnings:
        print(f'  ⚠  {w}')
print()
if result.passed:
    print('  ✅  Quality Gate: PASS — dataset gotowy do treningu')
    sys.exit(0)
else:
    print('  ❌  Quality Gate: FAIL — sprawdź problemy powyżej')
    sys.exit(1)
"
}

# ---------------------------------------------------------------------------
# train — SFT + DPO training
# ---------------------------------------------------------------------------
cmd_train() {
    info "Uruchamiam SFT training (${BASE_MODEL})..."
    docker compose --profile train run --rm foundry-trainer \
        python -m training.train_sft \
            --jsonl       "$SFT_JSONL" \
            --model       "$BASE_MODEL" \
            --output-dir  "$OUTPUT_DIR" \
            --lora-rank   16 \
            --epochs      3 \
            --grad-accum  4 \
            --max-seq-length 8192
    success "SFT zakończony"

    if [ "$SKIP_DPO" = "1" ]; then
        warn "SKIP_DPO=1 — pomijam DPO alignment"
        return
    fi

    info "Uruchamiam DPO alignment..."
    docker compose --profile train run --rm foundry-trainer \
        python -m training.train_dpo \
            --dpo-jsonl  "$DPO_JSONL" \
            --sft-model  "$OUTPUT_DIR/sft" \
            --output-dir "$OUTPUT_DIR"
    success "DPO zakończony"
}

# ---------------------------------------------------------------------------
# export — eksport modelu do GGUF + ZIP dla klienta
# ---------------------------------------------------------------------------
cmd_export() {
    info "Generuję datacard..."
    docker compose --profile train run --rm foundry-trainer \
        python -m training.datacard \
            --jsonl "$SFT_JSONL" \
            --dpo   "$DPO_JSONL" \
    || warn "Datacard nie mógł być wygenerowany (dataset może być pusty)"

    info "Eksportuję model '${MODEL_NAME}' do GGUF + ZIP..."
    docker compose --profile train run --rm foundry-trainer \
        python -c "
from training.export import merge_lora, convert_to_gguf, build_client_package
import os
sft_dir   = '$OUTPUT_DIR/sft'
dpo_dir   = '$OUTPUT_DIR/dpo'
model_dir = dpo_dir if os.path.isdir(dpo_dir) else sft_dir
print(f'  Używam modelu: {model_dir}')
merged = merge_lora(model_dir, '/app/output/models/merged')
gguf   = convert_to_gguf(merged, '/app/output/models/gguf')
pkg    = build_client_package(
    model_name='$MODEL_NAME',
    gguf_path=gguf,
    jsonl_path='$SFT_JSONL',
    output_dir='/app/output',
)
print(f'  Paczka ZIP: {pkg}')
"
    success "Eksport zakończony — plik ZIP w output/"
}

# ---------------------------------------------------------------------------
# chatbot — uruchom Ollama + Open WebUI
# ---------------------------------------------------------------------------
cmd_chatbot() {
    info "Uruchamiam Chatbot Studio (Ollama + Open WebUI)..."
    docker compose --profile chatbot up -d
    sleep 3
    success "Chatbot gotowy"
    echo -e "  Open WebUI: ${CYAN}http://localhost:3000${NC}"
    echo -e "  Ollama API: ${CYAN}http://localhost:11434${NC}"
    echo ""
    echo "  Załaduj model:"
    echo "    docker exec foundry_ollama ollama pull llama3.2"
    echo "    docker exec foundry_ollama ollama pull phi4-mini"
}

# ---------------------------------------------------------------------------
# full-run — pełny automat: generate → gate → train → export → chatbot
# ---------------------------------------------------------------------------
cmd_full_run() {
    echo -e "${BOLD}"
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║  Foundry Studio — PEŁNY AUTOMAT          ║"
    echo "  ║  generate → gate → train → export → chat ║"
    echo "  ╚══════════════════════════════════════════╝"
    echo -e "${NC}"

    cmd_generate
    echo ""

    if cmd_gate; then
        info "Quality Gate PASS — kontynuuję trening"
    else
        warn "Quality Gate FAIL — dataset może być za mały lub za słabej jakości"
        read -r -p "Kontynuować mimo to? [y/N] " ans
        [ "${ans,,}" = "y" ] || { info "Przerwano przez użytkownika."; exit 0; }
    fi
    echo ""

    cmd_train
    echo ""

    cmd_export
    echo ""

    cmd_chatbot
    echo ""

    success "PEŁNY AUTOMAT zakończony pomyślnie"
    echo -e "  UI:         ${CYAN}http://localhost:8501${NC}"
    echo -e "  API:        ${CYAN}http://localhost:8080/docs${NC}"
    echo -e "  Open WebUI: ${CYAN}http://localhost:3000${NC}"
}

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
COMMAND="${1:-help}"
case "$COMMAND" in
    setup)    cmd_setup    ;;
    up)       cmd_up       ;;
    status)   cmd_status   ;;
    generate) cmd_generate ;;
    gate)     cmd_gate     ;;
    train)    cmd_train    ;;
    export)   cmd_export   ;;
    chatbot)  cmd_chatbot  ;;
    full-run) cmd_full_run ;;
    *)
        echo "Użycie: $0 {setup|up|status|generate|gate|train|export|chatbot|full-run}"
        exit 1
        ;;
esac
