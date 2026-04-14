# =============================================================================
# scripts/foundry.ps1 — Foundry Studio Controller (Windows PowerShell)
#
# Uruchamianie:
#   .\scripts\foundry.ps1 setup
#   .\scripts\foundry.ps1 up
#   .\scripts\foundry.ps1 full-run
#
# Lub z parametrami:
#   $env:MODEL_NAME="my-model"; .\scripts\foundry.ps1 full-run
#   $env:CHUNK_LIMIT="50";      .\scripts\foundry.ps1 generate
#   $env:SKIP_DPO="1";          .\scripts\foundry.ps1 train
# =============================================================================

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    [string]$ModelName    = $env:MODEL_NAME   ?? "foundry-domain-model",
    [string]$ChunkLimit   = $env:CHUNK_LIMIT  ?? "0",
    [string]$SkipDpo      = $env:SKIP_DPO     ?? "0",
    [string]$ApiUrl       = $env:API_URL       ?? "http://localhost:8080",
    [string]$OllamaUrl    = $env:OLLAMA_URL    ?? "http://localhost:11434"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

# ---------------------------------------------------------------------------
# Kolory / output
# ---------------------------------------------------------------------------
function Log   { param($msg) Write-Host "[Foundry] $msg" -ForegroundColor Cyan }
function Ok    { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Warn  { param($msg) Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Err   { param($msg) Write-Host "  [X]  $msg" -ForegroundColor Red }
function Step  { param($msg) Write-Host "`n== $msg ==" -ForegroundColor Magenta }

# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------
function HasGpu {
    try { nvidia-smi 2>$null | Out-Null; return $true }
    catch { return $false }
}

function ApiReady {
    try {
        $r = Invoke-WebRequest -Uri "$ApiUrl/health" -UseBasicParsing -TimeoutSec 3
        return $r.StatusCode -eq 200
    } catch { return $false }
}

function OllamaReady {
    try {
        $r = Invoke-WebRequest -Uri "$OllamaUrl/api/tags" -UseBasicParsing -TimeoutSec 3
        return $r.StatusCode -eq 200
    } catch { return $false }
}

function WaitForHttp {
    param([string]$Url, [string]$Label, [int]$MaxTries = 60)
    Log "Czekam na $Label ($Url)..."
    for ($i = 1; $i -le $MaxTries; $i++) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { Ok "$Label gotowy"; return }
        } catch {}
        Write-Host "." -NoNewline
        Start-Sleep 2
    }
    Write-Host ""
    Err "Timeout: $Label nie odpowiedział po $($MaxTries * 2)s"
    throw "Service timeout: $Label"
}

function ApiPost {
    param([string]$Path, [string]$Body = "{}")
    $r = Invoke-WebRequest -Uri "$ApiUrl$Path" `
         -Method POST -Body $Body `
         -ContentType "application/json" `
         -UseBasicParsing -TimeoutSec 30
    return ($r.Content | ConvertFrom-Json)
}

function ApiGet {
    param([string]$Path)
    $r = Invoke-WebRequest -Uri "$ApiUrl$Path" -UseBasicParsing -TimeoutSec 10
    return ($r.Content | ConvertFrom-Json)
}

function WaitForRun {
    param([string]$StatusPath, [string]$Label, [int]$MaxSeconds = 14400)
    Log "Czekam na zakończenie: $Label..."
    $elapsed = 0
    while ($elapsed -lt $MaxSeconds) {
        try {
            $resp = ApiGet $StatusPath
            switch ($resp.status) {
                "done"  { Ok "$Label zakończony pomyślnie"; return $true }
                "error" { Err "$Label zakończony błędem: $($resp.error)"; return $false }
                default {
                    Write-Host "  [$elapsed`s] status: $($resp.status)" -NoNewline
                    Write-Host "`r" -NoNewline
                }
            }
        } catch {
            Write-Host "." -NoNewline
        }
        Start-Sleep 5
        $elapsed += 5
    }
    Err "Timeout po ${MaxSeconds}s"
    return $false
}

# ---------------------------------------------------------------------------
# CMD: setup
# ---------------------------------------------------------------------------
function Cmd-Setup {
    Step "Setup środowiska"

    # Docker
    try {
        $v = (docker --version) -replace "Docker version ", ""
        Ok "Docker: $v"
    } catch {
        Err "Docker nie jest zainstalowany lub nie działa."
        Err "Pobierz Docker Desktop: https://www.docker.com/products/docker-desktop"
        exit 1
    }

    # .env
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Warn ".env nie istniał — skopiowano z .env.example"
        Warn "Uzupełnij OPENAI_API_KEY i POSTGRES_PASSWORD w pliku .env!"
        Warn "Otwórz: notepad .env"
        exit 1
    } else {
        Ok ".env istnieje"
    }

    # Sprawdź wymagane klucze
    $envContent = Get-Content ".env" -Raw
    $missing = $false
    foreach ($key in @("OPENAI_API_KEY", "POSTGRES_PASSWORD")) {
        if ($envContent -match "^${key}=(.+)$") {
            $val = $Matches[1].Trim()
            if ($val -match "CHANGE_ME|XXXX|^$") {
                Err "Brak lub placeholder w .env: $key"
                $missing = $true
            } else {
                Ok "$key ustawiony"
            }
        } else {
            Err "Brak w .env: $key"
            $missing = $true
        }
    }
    if ($missing) {
        Err "Uzupełnij .env i uruchom ponownie: .\scripts\foundry.ps1 setup"
        exit 1
    }

    # GPU
    if (HasGpu) {
        $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        Ok "GPU wykryty: $gpuName"
    } else {
        Warn "Brak GPU — trening będzie niedostępny (pipeline i UI działają bez GPU)"
    }

    # data/
    New-Item -ItemType Directory -Force -Path "data","output" | Out-Null
    $pdfCount = (Get-ChildItem "data" -Filter "*.pdf" -ErrorAction SilentlyContinue).Count
    if ($pdfCount -eq 0) {
        Warn "Brak plików PDF w data\ — dodaj dokumenty przed uruchomieniem pipeline'u"
    } else {
        Ok "Znaleziono $pdfCount plik(ów) PDF w data\"
    }

    Ok "Setup zakończony."
    Write-Host ""
    Write-Host "Następny krok: .\scripts\foundry.ps1 up" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# CMD: up
# ---------------------------------------------------------------------------
function Cmd-Up {
    Step "Uruchamiam core stack"
    docker compose up -d --remove-orphans
    WaitForHttp "$ApiUrl/health" "foundry-api" 60
    WaitForHttp "http://localhost:8501" "foundry-ui" 30
    Write-Host ""
    Ok "Stack uruchomiony:"
    Write-Host "  UI      -> http://localhost:8501" -ForegroundColor Cyan
    Write-Host "  API     -> http://localhost:8080" -ForegroundColor Cyan
    Write-Host "  Swagger -> http://localhost:8080/docs" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# CMD: status
# ---------------------------------------------------------------------------
function Cmd-Status {
    Step "Status serwisów"

    Write-Host "`nDocker containers:" -ForegroundColor White
    docker compose ps

    Write-Host "`nAPI health:" -ForegroundColor White
    if (ApiReady) {
        $health = ApiGet "/health"
        Ok "API: status=$($health.status)"
        try {
            $stats = ApiGet "/api/samples/stats"
            Write-Host "  Q&A w bazie: $($stats.total) | avg score: $($stats.avg_quality_score)"
        } catch {}
    } else {
        Err "API niedostepne"
    }

    Write-Host "`nOllama:" -ForegroundColor White
    if (OllamaReady) {
        $tags = ApiGet "/api/chatbot/models" 2>$null
        $modelNames = ($tags.models | ForEach-Object { $_.name }) -join ", "
        Ok "Ollama aktywny | Modele: $(if($modelNames) {$modelNames} else {'brak'})"
    } else {
        Warn "Ollama niedostepny (uruchom: .\scripts\foundry.ps1 chatbot)"
    }

    Write-Host "`nGPU:" -ForegroundColor White
    if (HasGpu) {
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>$null | ForEach-Object {
            Ok "  $_"
        }
    } else {
        Warn "Brak GPU"
    }
}

# ---------------------------------------------------------------------------
# CMD: generate
# ---------------------------------------------------------------------------
function Cmd-Generate {
    Step "Pipeline generowania Q&A"

    if (-not (ApiReady)) {
        Err "API niedostepne. Uruchom najpierw: .\scripts\foundry.ps1 up"
        exit 1
    }

    $pdfs = Get-ChildItem "data" -Filter "*.pdf" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
    if (-not $pdfs) {
        Err "Brak plikow PDF w data\. Dodaj dokumenty i sprobuj ponownie."
        exit 1
    }

    Ok "Znalezione PDFy: $($pdfs -join ', ')"

    $filenamesJson = ($pdfs | ForEach-Object { "`"$_`"" }) -join ","
    $batchId = "auto-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

    $payload = @"
{
  "filenames": [$filenamesJson],
  "batch_id": "$batchId",
  "chunk_limit": $ChunkLimit
}
"@

    Log "Uruchamiam pipeline (batch: $batchId, chunk_limit: $ChunkLimit)..."
    $resp = ApiPost "/api/pipeline/run" $payload
    $runId = $resp.run_id

    if (-not $runId) {
        Err "Nie udalo sie uruchomic pipeline'u"
        exit 1
    }

    Ok "Run ID: $runId"
    $ok = WaitForRun "/api/pipeline/status/$runId" "Pipeline" 7200

    try {
        $stats = ApiGet "/api/samples/stats"
        Write-Host ""
        Ok "Dataset: $($stats.total) probek | avg score: $($stats.avg_quality_score)"
    } catch {}

    if (-not $ok) { exit 1 }
}

# ---------------------------------------------------------------------------
# CMD: gate
# ---------------------------------------------------------------------------
function Cmd-Gate {
    Step "Quality Gate — sprawdzam dataset"

    if (-not (ApiReady)) {
        Err "API niedostepne. Uruchom najpierw: .\scripts\foundry.ps1 up"
        exit 1
    }

    $resp = ApiPost "/api/training/gate" "{}"
    Write-Host ""

    foreach ($check in $resp.checks) {
        if ($check.passed) {
            Ok "$($check.name) — $($check.message)"
        } else {
            Err "$($check.name) — $($check.message)"
        }
    }

    Write-Host ""
    if ($resp.passed) {
        Ok "Quality Gate PASSED — dataset gotowy do treningu"
        return $true
    } else {
        Err "Quality Gate FAILED — popraw dataset przed treningiem"
        return $false
    }
}

# ---------------------------------------------------------------------------
# CMD: train
# ---------------------------------------------------------------------------
function Cmd-Train {
    Step "Trening modelu (SFT -> DPO)"

    if (-not (HasGpu)) {
        Err "Brak GPU — trening wymaga NVIDIA GPU z zainstalowanymi sterownikami"
        exit 1
    }

    if (-not (ApiReady)) {
        Err "API niedostepne. Uruchom najpierw: .\scripts\foundry.ps1 up"
        exit 1
    }

    $skipDpoFlag = if ($SkipDpo -eq "1") { "true" } else { "false" }

    $payload = @"
{
  "run_name": "$ModelName",
  "skip_dpo": $skipDpoFlag
}
"@

    Log "Uruchamiam trening (model: $ModelName)..."
    $resp = ApiPost "/api/training/run" $payload
    $runId = $resp.run_id

    if (-not $runId) {
        Err "Nie udalo sie uruchomic treningu"
        exit 1
    }

    Ok "Run ID: $runId"
    Ok "Model: $($resp.base_model) | Probki: $($resp.n_samples) | Szac. czas: $($resp.estimated_hours)h"

    $ok = WaitForRun "/api/training/status/$runId" "Trening" 14400
    $runId | Set-Content "$env:TEMP\foundry_last_train_id"

    if (-not $ok) { exit 1 }
}

# ---------------------------------------------------------------------------
# CMD: export
# ---------------------------------------------------------------------------
function Cmd-Export {
    Step "Eksport modelu -> GGUF + ZIP"

    if (-not (ApiReady)) {
        Err "API niedostepne. Uruchom najpierw: .\scripts\foundry.ps1 up"
        exit 1
    }

    $payload = @"
{
  "model_path": "/app/output/models/sft",
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "model_name": "$ModelName",
  "domain_label": "ESG / Prawo korporacyjne UE",
  "quantization": "Q4_K_M"
}
"@

    Log "Uruchamiam eksport modelu: $ModelName..."
    $resp = ApiPost "/api/training/export" $payload
    $exportRunId = $resp.export_run_id

    if (-not $exportRunId) {
        Err "Nie udalo sie uruchomic eksportu"
        exit 1
    }

    Ok "Export run ID: $exportRunId"
    $ok = WaitForRun "/api/training/status/$exportRunId" "Eksport" 3600

    if ($ok) {
        $status = ApiGet "/api/training/status/$exportRunId"
        $zipPath = $status.config.zip_path
        if ($zipPath) {
            $zipName = Split-Path $zipPath -Leaf
            Ok "Paczka gotowa: $zipPath"

            Log "Pobieram ZIP: $zipName..."
            $downloadUrl = "$ApiUrl/api/training/export/download/$exportRunId"
            Invoke-WebRequest -Uri $downloadUrl -OutFile "output\$zipName" -UseBasicParsing
            $sizeMB = [math]::Round((Get-Item "output\$zipName").Length / 1MB, 1)
            Ok "ZIP zapisany: output\$zipName ($sizeMB MB)"
        }
    }

    $exportRunId | Set-Content "$env:TEMP\foundry_last_export_id"
    if (-not $ok) { exit 1 }
}

# ---------------------------------------------------------------------------
# CMD: chatbot
# ---------------------------------------------------------------------------
function Cmd-Chatbot {
    Step "Chatbot Studio — Ollama + Open WebUI"

    Log "Uruchamiam profil chatbot..."
    docker compose --profile chatbot up -d

    WaitForHttp "$OllamaUrl/api/tags" "Ollama" 60
    WaitForHttp "http://localhost:3000" "Open WebUI" 60

    # Auto-ładuj model jeśli istnieje
    $ggufFile = Get-ChildItem "output" -Filter "*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
    $modelfile = "output\Modelfile"

    if ($ggufFile -and (Test-Path $modelfile)) {
        Log "Laduje model do Ollamy: $ModelName..."
        docker exec foundry_ollama mkdir -p /models
        docker cp $ggufFile.FullName "foundry_ollama:/models/$($ggufFile.Name)"
        docker cp $modelfile "foundry_ollama:/Modelfile"
        docker exec foundry_ollama ollama create $ModelName -f /Modelfile
        Ok "Model '$ModelName' zaladowany do Ollamy"
    } else {
        Warn "Brak pliku .gguf lub Modelfile w output\ — zaladuj model recznie"
        Warn "  docker cp model.gguf foundry_ollama:/models/"
        Warn "  docker exec foundry_ollama ollama create $ModelName -f /Modelfile"
    }

    Write-Host ""
    Ok "Chatbot gotowy:"
    Write-Host "  Open WebUI -> http://localhost:3000" -ForegroundColor Cyan
    Write-Host "  Ollama API -> http://localhost:11434" -ForegroundColor Cyan
    Write-Host "  Test w UI  -> http://localhost:8501 -> strona Chatbot" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# CMD: down
# ---------------------------------------------------------------------------
function Cmd-Down {
    Step "Zatrzymuje wszystko"
    docker compose --profile chatbot --profile train --profile gpu down
    Ok "Wszystkie serwisy zatrzymane"
}

# ---------------------------------------------------------------------------
# CMD: full-run
# ---------------------------------------------------------------------------
function Cmd-FullRun {
    Step "FULL AUTO RUN — generowanie -> gate -> trening -> eksport -> chatbot"

    $pdfCount = (Get-ChildItem "data" -Filter "*.pdf" -ErrorAction SilentlyContinue).Count
    $gpuAvail = if (HasGpu) { "TAK ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1))" } else { "NIE" }

    Write-Host ""
    Write-Host "  Konfiguracja:" -ForegroundColor White
    Write-Host "    PDFy:        $pdfCount plik(ow) w data\"
    Write-Host "    GPU:         $gpuAvail"
    Write-Host "    Model name:  $ModelName"
    Write-Host "    Chunk limit: $(if($ChunkLimit -eq '0'){'wszystkie'}else{$ChunkLimit})"
    Write-Host "    Skip DPO:    $(if($SkipDpo -eq '1'){'TAK'}else{'NIE'})"
    Write-Host ""

    # 1. Core stack
    Cmd-Up

    # 2. Generowanie Q&A
    Cmd-Generate

    # 3. Quality Gate
    $gatePassed = Cmd-Gate
    if (-not $gatePassed) {
        Warn "Quality Gate nie przeszedl."
        $ans = Read-Host "Kontynuowac trening mimo to? [y/N]"
        if ($ans -ne "y") {
            Log "Przerwano. Popraw dataset i uruchom: .\scripts\foundry.ps1 gate"
            exit 0
        }
    }

    # 4. Trening — tylko z GPU
    if (HasGpu) {
        Cmd-Train
        Cmd-Export
        Cmd-Chatbot
    } else {
        Warn "Brak GPU — pomijam trening i eksport"
        Warn "Gdy bedziesz miec GPU, uruchom:"
        Warn "  .\scripts\foundry.ps1 train"
        Warn "  .\scripts\foundry.ps1 export"
        Warn "  .\scripts\foundry.ps1 chatbot"
    }

    Step "FULL AUTO RUN zakończony"
    Write-Host ""
    Ok "Platforma gotowa:"
    Write-Host "  UI/AutoPilot -> http://localhost:8501" -ForegroundColor Cyan
    Write-Host "  API/Swagger  -> http://localhost:8080/docs" -ForegroundColor Cyan
    if (HasGpu) {
        Write-Host "  Chatbot      -> http://localhost:3000" -ForegroundColor Cyan
    }
    Write-Host ""
}

# ---------------------------------------------------------------------------
# CMD: help
# ---------------------------------------------------------------------------
function Cmd-Help {
    Write-Host ""
    Write-Host "  Foundry Studio — Windows PowerShell Controller" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  SETUP & LIFECYCLE"
    Write-Host "    .\scripts\foundry.ps1 setup        Pierwsze uruchomienie"
    Write-Host "    .\scripts\foundry.ps1 up           Uruchom core stack (API + UI + baza)"
    Write-Host "    .\scripts\foundry.ps1 down         Zatrzymaj wszystko"
    Write-Host "    .\scripts\foundry.ps1 status       Health check wszystkich serwisow"
    Write-Host ""
    Write-Host "  PIPELINE"
    Write-Host "    .\scripts\foundry.ps1 full-run     PELNY AUTOMAT (polecane)"
    Write-Host "    .\scripts\foundry.ps1 generate     Generuj Q&A z PDFow w data\"
    Write-Host "    .\scripts\foundry.ps1 gate         Sprawdz jakosc datasetu"
    Write-Host "    .\scripts\foundry.ps1 train        Trening SFT + DPO (wymaga GPU)"
    Write-Host "    .\scripts\foundry.ps1 export       Eksportuj model do GGUF + ZIP"
    Write-Host "    .\scripts\foundry.ps1 chatbot      Uruchom Ollama + Open WebUI"
    Write-Host ""
    Write-Host "  ZMIENNE (opcjonalne, ustaw przed uruchomieniem):"
    Write-Host "    `$env:MODEL_NAME='my-model'  — nazwa modelu (domyslnie: foundry-domain-model)"
    Write-Host "    `$env:CHUNK_LIMIT='50'        — limit chunkow (0 = wszystkie)"
    Write-Host "    `$env:SKIP_DPO='1'            — pomij DPO alignment"
    Write-Host ""
    Write-Host "  Przyklad:"
    Write-Host "    `$env:MODEL_NAME='esg-v2'; .\scripts\foundry.ps1 full-run" -ForegroundColor Yellow
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($Command) {
    "setup"    { Cmd-Setup }
    "up"       { Cmd-Up }
    "down"     { Cmd-Down }
    "status"   { Cmd-Status }
    "generate" { Cmd-Generate }
    "gate"     { Cmd-Gate }
    "train"    { Cmd-Train }
    "export"   { Cmd-Export }
    "chatbot"  { Cmd-Chatbot }
    "full-run" { Cmd-FullRun }
    "help"     { Cmd-Help }
    default    {
        Err "Nieznana komenda: $Command"
        Cmd-Help
        exit 1
    }
}
