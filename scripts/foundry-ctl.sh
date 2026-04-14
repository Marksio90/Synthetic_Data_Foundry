#!/usr/bin/env bash
# =============================================================================
# scripts/foundry-ctl.sh — Foundry Studio Controller
#
# Automatyzuje cały pipeline: setup → generowanie → gate → trening → eksport → chatbot
# Używa wyłącznie docker compose i docker exec.
#
# Użycie:
#   ./scripts/foundry-ctl.sh setup
#   ./scripts/foundry-ctl.sh up
#   ./scripts/foundry-ctl.sh full-run
#   ./scripts/foundry-ctl.sh generate [--limit N]
#   ./scripts/foundry-ctl.sh gate
#   ./scripts/foundry-ctl.sh train [--skip-dpo]
#   ./scripts/foundry-ctl.sh export [--model-name NAME]
#   ./scripts/foundry-ctl.sh chatbot
#   ./scripts/foundry-ctl.sh status
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Kolory
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BLUE}[Foundry]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
step() { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}\n"; }

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
API_URL="${API_URL:-http://localhost:8080}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MODEL_NAME="${MODEL_NAME:-foundry-domain-model}"
CHUNK_LIMIT="${CHUNK_LIMIT:-0}"
SKIP_DPO="${SKIP_DPO:-0}"

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Pomocnicze funkcje
# ---------------------------------------------------------------------------

has_gpu() {
    command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null
}

api_ready() {
    curl -sf "${API_URL}/health" &>/dev/null
}

ollama_ready() {
    curl -sf "${OLLAMA_URL}/api/tags" &>/dev/null
}

# Czeka aż serwis HTTP odpowie lub timeout
wait_for_http() {
    local url="$1" label="$2" max="${3:-60}"
    log "Czekam na $label ($url)..."
    for i in $(seq 1 "$max"); do
        if curl -sf "$url" &>/dev/null; then
            ok "$label gotowy"
            return 0
        fi
        printf '.'
        sleep 2
    done
    echo ""
    err "Timeout: $label nie odpowiedział po $((max * 2))s"
    return 1
}

# Wywołanie API — zwraca JSON
api_post() {
    local path="$1"; shift
    curl -sf -X POST "${API_URL}${path}" \
        -H "Content-Type: application/json" \
        -d "${1:-{}}"
}

api_get() {
    local path="$1"
    curl -sf "${API_URL}${path}"
}

# Czeka na zakończenie async runu (polling co 5s)
wait_for_run() {
    local status_path="$1" label="$2" max="${3:-3600}"
    log "Czekam na zakończenie: $label..."
    local elapsed=0
    while [ "$elapsed" -lt "$max" ]; do
        local resp
        resp=$(curl -sf "${API_URL}${status_path}" 2>/dev/null || echo '{"status":"unknown"}')
        local status
        status=$(echo "$resp" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        case "$status" in
            done)
                ok "$label zakończony pomyślnie"
                return 0
                ;;
            error)
                local errmsg
                errmsg=$(echo "$resp" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
                err "$label zakończony błędem: $errmsg"
                return 1
                ;;
            *)
                printf "  [%ds] status: %s\r" "$elapsed" "$status"
                sleep 5
                elapsed=$((elapsed + 5))
                ;;
        esac
    done
    err "Timeout po ${max}s"
    return 1
}

# Wypisz ostatnie logi runu
show_run_log() {
    local log_path="$1" n="${2:-20}"
    local resp
    resp=$(curl -sf "${API_URL}${log_path}?limit=${n}&offset=0" 2>/dev/null || echo '{}')
    echo "$resp" | grep -o '"lines":\[[^]]*\]' | \
        sed 's/"lines":\[//;s/\]$//' | \
        tr ',' '\n' | sed 's/^"//;s/"$//' | \
        sed 's/\\n/\n/g;s/\\t/\t/g'
}

# ---------------------------------------------------------------------------
# CMD: setup
# ---------------------------------------------------------------------------
cmd_setup() {
    step "Setup środowiska"

    # Sprawdź docker
    if ! command -v docker &>/dev/null; then
        err "Docker nie jest zainstalowany. Zainstaluj Docker Desktop."
        exit 1
    fi
    ok "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"

    # Sprawdź .env
    if [ ! -f .env ]; then
        cp .env.example .env
        warn ".env nie istniał — skopiowano z .env.example"
        warn "Uzupełnij OPENAI_API_KEY i POSTGRES_PASSWORD w pliku .env!"
        warn "  \$EDITOR .env"
        exit 1
    else
        ok ".env istnieje"
    fi

    # Sprawdź wymagane klucze
    local missing=0
    for key in OPENAI_API_KEY POSTGRES_PASSWORD; do
        val=$(grep "^${key}=" .env | cut -d'=' -f2- | tr -d ' ')
        if [ -z "$val" ] || [[ "$val" == *"CHANGE_ME"* ]] || [[ "$val" == "sk-"*"XXXX"* ]]; then
            err "Brak lub placeholder w .env: $key"
            missing=1
        else
            ok "$key ustawiony"
        fi
    done
    [ "$missing" -eq 1 ] && exit 1

    # GPU
    if has_gpu; then
        ok "GPU wykryty: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    else
        warn "Brak GPU — trening będzie niedostępny (pipeline i UI działają bez GPU)"
    fi

    # Katalog data/
    mkdir -p data output
    pdf_count=$(find data/ -name "*.pdf" 2>/dev/null | wc -l)
    if [ "$pdf_count" -eq 0 ]; then
        warn "Brak plików PDF w data/ — dodaj dokumenty przed uruchomieniem pipeline'u"
    else
        ok "Znaleziono $pdf_count plik(ów) PDF w data/"
    fi

    ok "Setup zakończony. Uruchom: make up"
}

# ---------------------------------------------------------------------------
# CMD: up
# ---------------------------------------------------------------------------
cmd_up() {
    step "Uruchamiam core stack"

    docker compose up -d --remove-orphans

    wait_for_http "${API_URL}/health" "foundry-api" 60
    wait_for_http "http://localhost:8501" "foundry-ui" 30

    echo ""
    ok "Stack uruchomiony:"
    echo -e "  ${CYAN}UI${NC}      → http://localhost:8501"
    echo -e "  ${CYAN}API${NC}     → http://localhost:8080"
    echo -e "  ${CYAN}Swagger${NC} → http://localhost:8080/docs"
}

# ---------------------------------------------------------------------------
# CMD: status
# ---------------------------------------------------------------------------
cmd_status() {
    step "Status serwisów"

    echo -e "${BOLD}Docker containers:${NC}"
    docker compose ps 2>/dev/null || true

    echo ""
    echo -e "${BOLD}API health:${NC}"
    if api_ready; then
        resp=$(api_get "/health")
        ok "API: $resp"
        stats=$(api_get "/api/samples/stats" 2>/dev/null || echo '{}')
        total=$(echo "$stats" | grep -o '"total":[0-9]*' | cut -d: -f2)
        avg=$(echo "$stats" | grep -o '"avg_quality_score":[0-9.]*' | cut -d: -f2)
        echo "  Q&A w bazie: ${total:-0} | avg score: ${avg:---}"
    else
        err "API niedostępne"
    fi

    echo ""
    echo -e "${BOLD}Ollama:${NC}"
    if ollama_ready; then
        models=$(curl -sf "${OLLAMA_URL}/api/tags" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ', ' || echo "brak")
        ok "Ollama aktywny | Modele: ${models%,}"
    else
        warn "Ollama niedostępny (uruchom: make chatbot)"
    fi

    echo ""
    echo -e "${BOLD}GPU:${NC}"
    if has_gpu; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | \
            while IFS=',' read -r name total free; do
                ok "  $name | VRAM: $total total, $free free"
            done
    else
        warn "Brak GPU"
    fi
}

# ---------------------------------------------------------------------------
# CMD: generate
# ---------------------------------------------------------------------------
cmd_generate() {
    step "Pipeline generowania Q&A"

    if ! api_ready; then
        err "API niedostępne. Uruchom najpierw: make up"
        exit 1
    fi

    # Znajdź pliki PDF
    mapfile -t pdfs < <(find data/ -name "*.pdf" -printf "%f\n" 2>/dev/null | sort)
    if [ "${#pdfs[@]}" -eq 0 ]; then
        err "Brak plików PDF w data/. Dodaj dokumenty i spróbuj ponownie."
        exit 1
    fi

    ok "Znalezione PDFy: ${pdfs[*]}"

    # Buduj payload JSON
    filenames_json=$(printf '"%s",' "${pdfs[@]}")
    filenames_json="[${filenames_json%,}]"

    batch_id="auto-$(date +%Y%m%d-%H%M%S)"

    payload=$(cat <<EOF
{
  "filenames": $filenames_json,
  "batch_id": "$batch_id",
  "chunk_limit": $CHUNK_LIMIT
}
EOF
)

    log "Uruchamiam pipeline (batch: $batch_id, chunk_limit: $CHUNK_LIMIT)..."
    resp=$(api_post "/api/pipeline/run" "$payload")
    run_id=$(echo "$resp" | grep -o '"run_id":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$run_id" ]; then
        err "Nie udało się uruchomić pipeline'u: $resp"
        exit 1
    fi

    ok "Run ID: $run_id"
    wait_for_run "/api/pipeline/status/$run_id" "Pipeline" 7200

    # Podsumowanie
    stats=$(api_get "/api/samples/stats" 2>/dev/null || echo '{}')
    total=$(echo "$stats" | grep -o '"total":[0-9]*' | cut -d: -f2)
    avg=$(echo "$stats" | grep -o '"avg_quality_score":[0-9.]*' | cut -d: -f2)
    echo ""
    ok "Dataset: ${total:-0} próbek | avg score: ${avg:---}"
}

# ---------------------------------------------------------------------------
# CMD: gate
# ---------------------------------------------------------------------------
cmd_gate() {
    step "Quality Gate — sprawdzam dataset"

    if ! api_ready; then
        err "API niedostępne. Uruchom najpierw: make up"
        exit 1
    fi

    resp=$(api_post "/api/training/gate" "{}")
    passed=$(echo "$resp" | grep -o '"passed":[^,}]*' | cut -d: -f2 | tr -d ' ')

    echo ""
    echo "$resp" | grep -o '"name":"[^"]*","passed":[^,]*,"value":"[^"]*","threshold":"[^"]*","message":"[^"]*"' | \
    while IFS= read -r check; do
        name=$(echo "$check" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
        cpassed=$(echo "$check" | grep -o '"passed":[^,]*' | cut -d: -f2 | tr -d ' ')
        msg=$(echo "$check" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
        if [ "$cpassed" = "true" ]; then
            ok "$name — $msg"
        else
            err "$name — $msg"
        fi
    done

    echo ""
    if [ "$passed" = "true" ]; then
        ok "Quality Gate PASSED — dataset gotowy do treningu"
        return 0
    else
        err "Quality Gate FAILED — popraw dataset przed treningiem"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# CMD: train
# ---------------------------------------------------------------------------
cmd_train() {
    step "Trening modelu (SFT → DPO)"

    if ! has_gpu; then
        err "Brak GPU — trening wymaga NVIDIA GPU"
        exit 1
    fi

    if ! api_ready; then
        err "API niedostępne. Uruchom najpierw: make up"
        exit 1
    fi

    skip_dpo_flag="false"
    [ "$SKIP_DPO" = "1" ] && skip_dpo_flag="true"

    payload=$(cat <<EOF
{
  "run_name": "$MODEL_NAME",
  "skip_dpo": $skip_dpo_flag
}
EOF
)

    log "Uruchamiam trening (model: $MODEL_NAME, skip_dpo: $skip_dpo_flag)..."
    resp=$(api_post "/api/training/run" "$payload")
    run_id=$(echo "$resp" | grep -o '"run_id":"[^"]*"' | cut -d'"' -f4)
    base_model=$(echo "$resp" | grep -o '"base_model":"[^"]*"' | cut -d'"' -f4)
    n_samples=$(echo "$resp" | grep -o '"n_samples":[0-9]*' | cut -d: -f2)
    est_hours=$(echo "$resp" | grep -o '"estimated_hours":[0-9.]*' | cut -d: -f2)

    if [ -z "$run_id" ]; then
        err "Nie udało się uruchomić treningu: $resp"
        exit 1
    fi

    ok "Run ID: $run_id"
    ok "Model: $base_model | Próbki: $n_samples | Szac. czas: ${est_hours}h"

    wait_for_run "/api/training/status/$run_id" "Trening" 14400

    echo ""
    log "Ostatnie logi treningu:"
    show_run_log "/api/training/log/$run_id" 15
    echo ""

    # Zapisz run_id do pliku tymczasowego dla kolejnych kroków
    echo "$run_id" > /tmp/foundry_last_train_id
}

# ---------------------------------------------------------------------------
# CMD: export
# ---------------------------------------------------------------------------
cmd_export() {
    step "Eksport modelu → GGUF + ZIP"

    if ! api_ready; then
        err "API niedostępne. Uruchom najpierw: make up"
        exit 1
    fi

    payload=$(cat <<EOF
{
  "model_path": "/app/output/models/sft",
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "model_name": "$MODEL_NAME",
  "domain_label": "ESG / Prawo korporacyjne UE",
  "quantization": "Q4_K_M"
}
EOF
)

    log "Uruchamiam eksport modelu: $MODEL_NAME..."
    resp=$(api_post "/api/training/export" "$payload")
    export_run_id=$(echo "$resp" | grep -o '"export_run_id":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$export_run_id" ]; then
        err "Nie udało się uruchomić eksportu: $resp"
        exit 1
    fi

    ok "Export run ID: $export_run_id"
    wait_for_run "/api/training/status/$export_run_id" "Eksport" 3600

    # Pobierz ścieżkę ZIP
    status_resp=$(api_get "/api/training/status/$export_run_id")
    zip_path=$(echo "$status_resp" | grep -o '"zip_path":"[^"]*"' | cut -d'"' -f4)

    if [ -n "$zip_path" ]; then
        zip_name=$(basename "$zip_path")
        ok "Paczka dla klienta: $zip_path"

        # Pobierz ZIP z API
        log "Pobieram ZIP: $zip_name..."
        curl -sf "${API_URL}/api/training/export/download/${export_run_id}" \
            -o "output/${zip_name}" && \
            ok "ZIP zapisany: output/${zip_name} ($(du -sh "output/${zip_name}" | cut -f1))"
    fi

    echo "$export_run_id" > /tmp/foundry_last_export_id
}

# ---------------------------------------------------------------------------
# CMD: chatbot
# ---------------------------------------------------------------------------
cmd_chatbot() {
    step "Chatbot Studio — Ollama + Open WebUI"

    log "Uruchamiam profil chatbot..."
    docker compose --profile chatbot up -d

    wait_for_http "${OLLAMA_URL}/api/tags" "Ollama" 60
    wait_for_http "http://localhost:3000" "Open WebUI" 60

    # Jeśli jest gotowy model — załaduj automatycznie
    gguf_file=$(find output/ -name "*.gguf" 2>/dev/null | head -1)
    modelfile="output/Modelfile"

    if [ -n "$gguf_file" ] && [ -f "$modelfile" ]; then
        log "Ładuję model do Ollamy: $MODEL_NAME..."
        # Skopiuj model do kontenera
        docker exec foundry_ollama mkdir -p /models
        docker cp "$gguf_file" "foundry_ollama:/models/$(basename "$gguf_file")"
        docker cp "$modelfile" "foundry_ollama:/Modelfile"
        docker exec foundry_ollama ollama create "$MODEL_NAME" -f /Modelfile && \
            ok "Model '$MODEL_NAME' załadowany do Ollamy"
    else
        warn "Brak pliku .gguf lub Modelfile w output/ — załaduj model ręcznie"
        warn "  docker cp model.gguf foundry_ollama:/models/"
        warn "  docker exec foundry_ollama ollama create $MODEL_NAME -f /Modelfile"
    fi

    echo ""
    ok "Chatbot gotowy:"
    echo -e "  ${CYAN}Open WebUI${NC} → http://localhost:3000"
    echo -e "  ${CYAN}Ollama API${NC} → http://localhost:11434"
    echo -e "  ${CYAN}Test w UI${NC}  → http://localhost:8501 → strona Chatbot"
}

# ---------------------------------------------------------------------------
# CMD: full-run — cały pipeline od A do Z
# ---------------------------------------------------------------------------
cmd_full_run() {
    step "FULL AUTO RUN — generowanie → gate → trening → eksport → chatbot"

    echo -e "${BOLD}Konfiguracja:${NC}"
    echo "  PDFy:        $(find data/ -name "*.pdf" 2>/dev/null | wc -l) plik(ów) w data/"
    echo "  GPU:         $(has_gpu && echo "TAK ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))" || echo "NIE")"
    echo "  Model name:  $MODEL_NAME"
    echo "  Chunk limit: $([ "$CHUNK_LIMIT" = "0" ] && echo "wszystkie" || echo "$CHUNK_LIMIT")"
    echo "  Skip DPO:    $([ "$SKIP_DPO" = "1" ] && echo "TAK" || echo "NIE")"
    echo ""

    # 1. Core stack
    cmd_up

    # 2. Generowanie
    cmd_generate

    # 3. Quality Gate — jeśli nie przejdzie, pytamy czy kontynuować
    if ! cmd_gate; then
        warn "Quality Gate nie przeszedł."
        read -p "Kontynuować trening mimo to? [y/N] " ans
        [ "${ans:-N}" = "y" ] || { log "Przerwano. Popraw dataset i uruchom: make gate && make train"; exit 0; }
    fi

    # 4. Trening — tylko jeśli GPU dostępny
    if has_gpu; then
        cmd_train
        cmd_export

        # 5. Chatbot
        cmd_chatbot
    else
        warn "Brak GPU — pomijam trening i eksport"
        warn "Gdy będziesz mieć GPU, uruchom: make train && make export && make chatbot"
    fi

    step "FULL AUTO RUN zakończony"
    echo ""
    ok "Platforma gotowa:"
    echo -e "  ${CYAN}UI/AutoPilot${NC} → http://localhost:8501"
    echo -e "  ${CYAN}API/Swagger${NC}  → http://localhost:8080/docs"
    if has_gpu; then
        echo -e "  ${CYAN}Chatbot${NC}      → http://localhost:3000"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
CMD="${1:-help}"
shift || true

case "$CMD" in
    setup)    cmd_setup    "$@" ;;
    up)       cmd_up       "$@" ;;
    status)   cmd_status   "$@" ;;
    generate) cmd_generate "$@" ;;
    gate)     cmd_gate     "$@" ;;
    train)    cmd_train    "$@" ;;
    export)   cmd_export   "$@" ;;
    chatbot)  cmd_chatbot  "$@" ;;
    full-run) cmd_full_run "$@" ;;
    help|--help|-h)
        echo "Użycie: $0 {setup|up|status|generate|gate|train|export|chatbot|full-run}"
        ;;
    *)
        err "Nieznana komenda: $CMD"
        echo "Użycie: $0 {setup|up|status|generate|gate|train|export|chatbot|full-run}"
        exit 1
        ;;
esac
