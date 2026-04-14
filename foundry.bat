@echo off
setlocal EnableDelayedExpansion
:: =============================================================================
:: foundry.bat  --  Foundry Studio Controller for Windows
::
:: Usage:
::   foundry setup
::   foundry up
::   foundry full-run
::   foundry generate
::   foundry gate
::   foundry train
::   foundry export
::   foundry chatbot
::   foundry down
::   foundry status
::   foundry logs
::
:: Optional env overrides (set before running):
::   set MODEL_NAME=my-model  && foundry train
::   set CHUNK_LIMIT=50       && foundry generate
::   set SKIP_DPO=1           && foundry train
:: =============================================================================

:: Defaults
if "%MODEL_NAME%"==""  set MODEL_NAME=foundry-domain-model
if "%CHUNK_LIMIT%"=="" set CHUNK_LIMIT=0
if "%SKIP_DPO%"==""    set SKIP_DPO=0
if "%API_URL%"==""     set API_URL=http://localhost:8080
if "%OLLAMA_URL%"==""  set OLLAMA_URL=http://localhost:11434

set CMD=%~1
if "%CMD%"=="" set CMD=help

goto :%CMD% 2>nul || (
    echo [ERROR] Unknown command: %CMD%
    goto :help
)

:: =============================================================================
:help
:: =============================================================================
echo.
echo   Foundry Studio - Windows Controller
echo.
echo   SETUP ^& LIFECYCLE
echo     foundry setup        First run - check .env, Docker, GPU
echo     foundry up           Start core stack (API + UI + database)
echo     foundry down         Stop everything
echo     foundry status       Health check all services
echo     foundry logs         Show all container logs
echo.
echo   PIPELINE
echo     foundry full-run     FULL AUTO: generate -^> gate -^> train -^> export -^> chatbot
echo     foundry generate     Generate Q^&A from PDFs in data\
echo     foundry gate         Check dataset quality
echo     foundry train        SFT + DPO training (requires GPU)
echo     foundry export       Export model to GGUF + client ZIP
echo     foundry chatbot      Start Ollama + Open WebUI
echo.
echo   Optional env vars (set before running):
echo     set MODEL_NAME=my-model  ^&^& foundry train
echo     set CHUNK_LIMIT=50       ^&^& foundry generate
echo     set SKIP_DPO=1           ^&^& foundry train
echo.
goto :eof

:: =============================================================================
:setup
:: =============================================================================
echo.
echo == Setup ==
echo.

:: Check Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Start Docker Desktop and try again.
    exit /b 1
)
for /f "tokens=3" %%v in ('docker --version') do echo [OK] Docker %%v

:: Check .env
if not exist ".env" (
    if not exist ".env.example" (
        echo [ERROR] .env.example not found. Are you in the project root?
        exit /b 1
    )
    copy ".env.example" ".env" >nul
    echo [!] .env not found - copied from .env.example
    echo [!] Edit .env and set OPENAI_API_KEY and POSTGRES_PASSWORD
    echo     notepad .env
    exit /b 1
)
echo [OK] .env found

:: Check required keys
findstr /C:"OPENAI_API_KEY=sk-" ".env" >nul 2>&1
if errorlevel 1 (
    echo [!] OPENAI_API_KEY looks empty or unset in .env
) else (
    echo [OK] OPENAI_API_KEY set
)

findstr /R /C:"POSTGRES_PASSWORD=.\+" ".env" >nul 2>&1
if errorlevel 1 (
    echo [!] POSTGRES_PASSWORD looks empty in .env
) else (
    echo [OK] POSTGRES_PASSWORD set
)

:: Check GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [!] No GPU detected - training will be unavailable
) else (
    for /f "tokens=1,2 delims=," %%a in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        echo [OK] GPU detected: %%a
    )
)

:: Check data folder
if not exist "data\" mkdir data
if not exist "output\" mkdir output

set PDF_COUNT=0
for %%f in (data\*.pdf) do set /a PDF_COUNT+=1
if %PDF_COUNT%==0 (
    echo [!] No PDF files in data\ - add documents before running the pipeline
) else (
    echo [OK] Found %PDF_COUNT% PDF file(s) in data\
)

echo.
echo Setup complete. Next step:
echo   foundry up
goto :eof

:: =============================================================================
:up
:: =============================================================================
echo.
echo == Starting core stack ==
echo.
docker compose up -d --remove-orphans
if errorlevel 1 exit /b 1

call :_wait_http "%API_URL%/health" "foundry-api" 60
call :_wait_http "http://localhost:8501" "foundry-ui" 30

echo.
echo [OK] Stack running:
echo   UI      --^> http://localhost:8501
echo   API     --^> http://localhost:8080
echo   Swagger --^> http://localhost:8080/docs
goto :eof

:: =============================================================================
:down
:: =============================================================================
echo.
echo == Stopping all services ==
docker compose --profile chatbot --profile train --profile gpu down
echo [OK] All services stopped
goto :eof

:: =============================================================================
:status
:: =============================================================================
echo.
echo == Service status ==
echo.
echo --- Docker containers ---
docker compose ps
echo.

echo --- API health ---
curl -sf "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API not available
) else (
    echo [OK] API responding
    for /f %%r in ('curl -sf "%API_URL%/api/samples/stats" 2^>nul') do (
        echo   Stats: %%r
    )
)

echo.
echo --- Ollama ---
curl -sf "%OLLAMA_URL%/api/tags" >nul 2>&1
if errorlevel 1 (
    echo [!] Ollama not available (run: foundry chatbot)
) else (
    echo [OK] Ollama running
)

echo.
echo --- GPU ---
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [!] No GPU
) else (
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
)
goto :eof

:: =============================================================================
:generate
:: =============================================================================
echo.
echo == Pipeline - generating Q^&A ==
echo.

curl -sf "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API not available. Run: foundry up
    exit /b 1
)

:: Collect PDF filenames into JSON array
set FILENAMES_JSON=
set PDF_COUNT=0
for %%f in (data\*.pdf) do (
    set /a PDF_COUNT+=1
    if "!FILENAMES_JSON!"=="" (
        set FILENAMES_JSON="%%~nxf"
    ) else (
        set FILENAMES_JSON=!FILENAMES_JSON!,"%%~nxf"
    )
)

if %PDF_COUNT%==0 (
    echo [ERROR] No PDF files found in data\
    exit /b 1
)
echo [OK] Found PDFs: %FILENAMES_JSON%

:: Generate batch ID
for /f "tokens=1-2 delims= " %%a in ('wmic os get LocalDateTime /value ^| findstr "LocalDateTime"') do set DT=%%a
set DT=%DT:LocalDateTime=%
set BATCH_ID=auto-%DT:~0,14%

:: Write payload to temp file (avoids quoting issues)
set PAYLOAD_FILE=%TEMP%\foundry_payload.json
(
echo {
echo   "filenames": [%FILENAMES_JSON%],
echo   "batch_id": "%BATCH_ID%",
echo   "chunk_limit": %CHUNK_LIMIT%
echo }
) > "%PAYLOAD_FILE%"

echo Starting pipeline (batch: %BATCH_ID%, chunk_limit: %CHUNK_LIMIT%)...
for /f "delims=" %%r in ('curl -sf -X POST "%API_URL%/api/pipeline/run" -H "Content-Type: application/json" -d @"%PAYLOAD_FILE%"') do set RESP=%%r

:: Extract run_id
for /f "tokens=2 delims=:," %%i in ('echo !RESP! ^| findstr /c:"run_id"') do (
    set RUN_ID=%%~i
    set RUN_ID=!RUN_ID:"=!
    set RUN_ID=!RUN_ID: =!
)

if "%RUN_ID%"=="" (
    echo [ERROR] Failed to start pipeline
    echo Response: !RESP!
    exit /b 1
)
echo [OK] Run ID: %RUN_ID%

call :_wait_run "/api/pipeline/status/%RUN_ID%" "Pipeline" 7200
goto :eof

:: =============================================================================
:gate
:: =============================================================================
echo.
echo == Quality Gate ==
echo.

curl -sf "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API not available. Run: foundry up
    exit /b 1
)

for /f "delims=" %%r in ('curl -sf -X POST "%API_URL%/api/training/gate" -H "Content-Type: application/json" -d "{}"') do set GATE_RESP=%%r

echo !GATE_RESP! | findstr /c:"\"passed\":true" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Quality Gate FAILED - fix dataset before training
    echo Response: !GATE_RESP!
    exit /b 1
) else (
    echo [OK] Quality Gate PASSED - dataset ready for training
)
goto :eof

:: =============================================================================
:train
:: =============================================================================
echo.
echo == Model training (SFT -^> DPO) ==
echo.

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No GPU detected - training requires NVIDIA GPU
    exit /b 1
)

curl -sf "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API not available. Run: foundry up
    exit /b 1
)

if "%SKIP_DPO%"=="1" (set SKIP_DPO_FLAG=true) else (set SKIP_DPO_FLAG=false)

set TRAIN_PAYLOAD_FILE=%TEMP%\foundry_train.json
(
echo {
echo   "run_name": "%MODEL_NAME%",
echo   "skip_dpo": %SKIP_DPO_FLAG%
echo }
) > "%TRAIN_PAYLOAD_FILE%"

echo Starting training (model: %MODEL_NAME%, skip_dpo: %SKIP_DPO_FLAG%)...
for /f "delims=" %%r in ('curl -sf -X POST "%API_URL%/api/training/run" -H "Content-Type: application/json" -d @"%TRAIN_PAYLOAD_FILE%"') do set RESP=%%r

for /f "tokens=2 delims=:," %%i in ('echo !RESP! ^| findstr /c:"run_id"') do (
    set RUN_ID=%%~i
    set RUN_ID=!RUN_ID:"=!
    set RUN_ID=!RUN_ID: =!
)

if "%RUN_ID%"=="" (
    echo [ERROR] Failed to start training
    exit /b 1
)
echo [OK] Run ID: %RUN_ID%

call :_wait_run "/api/training/status/%RUN_ID%" "Training" 14400
echo %RUN_ID% > "%TEMP%\foundry_last_train_id"
goto :eof

:: =============================================================================
:export
:: =============================================================================
echo.
echo == Export model -^> GGUF + client ZIP ==
echo.

curl -sf "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API not available. Run: foundry up
    exit /b 1
)

set EXPORT_PAYLOAD_FILE=%TEMP%\foundry_export.json
(
echo {
echo   "model_path": "/app/output/models/sft",
echo   "base_model": "meta-llama/Llama-3.2-3B-Instruct",
echo   "model_name": "%MODEL_NAME%",
echo   "domain_label": "ESG / Prawo korporacyjne UE",
echo   "quantization": "Q4_K_M"
echo }
) > "%EXPORT_PAYLOAD_FILE%"

echo Starting export (model: %MODEL_NAME%)...
for /f "delims=" %%r in ('curl -sf -X POST "%API_URL%/api/training/export" -H "Content-Type: application/json" -d @"%EXPORT_PAYLOAD_FILE%"') do set RESP=%%r

for /f "tokens=2 delims=:," %%i in ('echo !RESP! ^| findstr /c:"export_run_id"') do (
    set EXPORT_RUN_ID=%%~i
    set EXPORT_RUN_ID=!EXPORT_RUN_ID:"=!
    set EXPORT_RUN_ID=!EXPORT_RUN_ID: =!
)

if "%EXPORT_RUN_ID%"=="" (
    echo [ERROR] Failed to start export
    exit /b 1
)
echo [OK] Export run ID: %EXPORT_RUN_ID%

call :_wait_run "/api/training/status/%EXPORT_RUN_ID%" "Export" 3600

:: Download the ZIP
set ZIP_NAME=%MODEL_NAME%.zip
echo Downloading ZIP: %ZIP_NAME%...
curl -sf "%API_URL%/api/training/export/download/%EXPORT_RUN_ID%" -o "output\%ZIP_NAME%"
if errorlevel 1 (
    echo [!] Could not download ZIP - check output\ folder on server
) else (
    echo [OK] ZIP saved: output\%ZIP_NAME%
)
echo %EXPORT_RUN_ID% > "%TEMP%\foundry_last_export_id"
goto :eof

:: =============================================================================
:chatbot
:: =============================================================================
echo.
echo == Chatbot - Ollama + Open WebUI ==
echo.

docker compose --profile chatbot up -d
if errorlevel 1 exit /b 1

call :_wait_http "%OLLAMA_URL%/api/tags" "Ollama" 60
call :_wait_http "http://localhost:3000" "Open WebUI" 60

:: Auto-load model if GGUF exists
set GGUF_FILE=
for %%f in (output\*.gguf) do if "%GGUF_FILE%"=="" set GGUF_FILE=%%f

if not "%GGUF_FILE%"=="" (
    if exist "output\Modelfile" (
        echo Loading model into Ollama: %MODEL_NAME%...
        docker exec foundry_ollama mkdir -p /models
        docker cp "%GGUF_FILE%" foundry_ollama:/models/
        docker cp "output\Modelfile" foundry_ollama:/Modelfile
        docker exec foundry_ollama ollama create %MODEL_NAME% -f /Modelfile
        echo [OK] Model '%MODEL_NAME%' loaded into Ollama
    ) else (
        echo [!] No Modelfile in output\ - run export first
    )
) else (
    echo [!] No .gguf file in output\ - run export first
)

echo.
echo [OK] Chatbot ready:
echo   Open WebUI --^> http://localhost:3000
echo   Ollama API --^> http://localhost:11434
echo   Studio UI  --^> http://localhost:8501
goto :eof

:: =============================================================================
:logs
:: =============================================================================
docker compose logs -f
goto :eof

:: =============================================================================
:full-run
:: =============================================================================
echo.
echo == FULL AUTO RUN ==
echo.

set PDF_COUNT=0
for %%f in (data\*.pdf) do set /a PDF_COUNT+=1

nvidia-smi >nul 2>&1
if errorlevel 1 (set GPU_STATUS=NO) else (set GPU_STATUS=YES)

echo   PDFs:        %PDF_COUNT% file(s) in data\
echo   GPU:         %GPU_STATUS%
echo   Model name:  %MODEL_NAME%
echo   Chunk limit: %CHUNK_LIMIT%
echo   Skip DPO:    %SKIP_DPO%
echo.

call :up
call :generate

call :gate
if errorlevel 1 (
    echo [!] Quality Gate failed.
    set /p CONT="Continue anyway? [y/N]: "
    if /i "!CONT!" neq "y" (
        echo Stopped. Fix dataset then run: foundry gate
        exit /b 0
    )
)

if "%GPU_STATUS%"=="YES" (
    call :train
    call :export
    call :chatbot
) else (
    echo [!] No GPU - skipping train / export / chatbot
    echo     When GPU is available run: foundry train
)

echo.
echo == FULL AUTO RUN complete ==
echo.
echo   UI/AutoPilot --^> http://localhost:8501
echo   API/Swagger  --^> http://localhost:8080/docs
if "%GPU_STATUS%"=="YES" (
    echo   Chatbot      --^> http://localhost:3000
)
goto :eof

:: =============================================================================
:: Internal helpers (prefix _ so they don't show in help)
:: =============================================================================

:_wait_http
:: %1 = URL, %2 = label, %3 = max tries
set _URL=%~1
set _LABEL=%~2
set _MAX=%~3
if "%_MAX%"=="" set _MAX=30
echo Waiting for %_LABEL% (%_URL%)...
for /l %%i in (1,1,%_MAX%) do (
    curl -sf "%_URL%" >nul 2>&1
    if not errorlevel 1 (
        echo [OK] %_LABEL% ready
        goto :eof
    )
    timeout /t 2 /nobreak >nul
)
echo [ERROR] Timeout: %_LABEL% did not respond
exit /b 1

:_wait_run
:: %1 = status path, %2 = label, %3 = max seconds
set _PATH=%~1
set _LABEL=%~2
set _MAX=%~3
if "%_MAX%"=="" set _MAX=3600
set _ELAPSED=0
echo Waiting for %_LABEL% to finish...
:_wait_run_loop
if %_ELAPSED% geq %_MAX% (
    echo [ERROR] Timeout after %_MAX%s
    exit /b 1
)
for /f "delims=" %%r in ('curl -sf "%API_URL%%_PATH%" 2^>nul') do set _RESP=%%r
echo !_RESP! | findstr /c:"\"status\":\"done\"" >nul 2>&1
if not errorlevel 1 (
    echo [OK] %_LABEL% completed successfully
    goto :eof
)
echo !_RESP! | findstr /c:"\"status\":\"error\"" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] %_LABEL% failed
    echo !_RESP!
    exit /b 1
)
set /a _ELAPSED+=5
timeout /t 5 /nobreak >nul
echo   [%_ELAPSED%s] running...
goto :_wait_run_loop
