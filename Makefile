# =============================================================================
# Foundry Studio — Makefile
# Jeden interfejs do całej platformy. Używa wyłącznie docker compose.
#
# Szybki start:
#   make setup      ← pierwsze uruchomienie (kopiuje .env, sprawdza wymagania)
#   make up         ← uruchamia core stack (API + UI + baza)
#   make full-run   ← PEŁNY AUTOMAT: generowanie → trening → eksport → chatbot
#
# Poszczególne kroki:
#   make generate   ← uruchamia pipeline przez API (PDFy z data/)
#   make gate       ← sprawdza jakość datasetu
#   make train      ← SFT + DPO training (wymaga GPU)
#   make export     ← eksport modelu do GGUF + ZIP dla klienta
#   make chatbot    ← uruchamia Ollama + Open WebUI
#
# Zarządzanie:
#   make down       ← zatrzymuje wszystko
#   make logs       ← logi wszystkich serwisów
#   make status     ← health check wszystkich serwisów
#   make clean      ← usuwa kontenery i wolumeny (UWAGA: kasuje dane)
# =============================================================================

SHELL := /bin/bash
CTL   := ./scripts/foundry-ctl.sh

.DEFAULT_GOAL := help

.PHONY: help setup up down full-run generate gate train export chatbot \
        logs logs-api logs-ui logs-frontend logs-trainer status clean rebuild \
        check-merge-conflicts

# ---------------------------------------------------------------------------
# Pomoc
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "  Foundry Studio — dostępne komendy:"
	@echo ""
	@echo "  SETUP & LIFECYCLE"
	@echo "    make setup        Pierwsze uruchomienie — konfiguracja środowiska"
	@echo "    make up           Uruchom core stack (API + UI + baza)"
	@echo "    make down         Zatrzymaj wszystko"
	@echo "    make rebuild      Przebuduj obrazy i uruchom"
	@echo "    make status       Health check wszystkich serwisów"
	@echo "    make clean        Usuń kontenery + wolumeny (kasuje dane!)"
	@echo ""
	@echo "  PIPELINE"
	@echo "    make full-run     PEŁNY AUTOMAT: gen → gate → train → export → chatbot"
	@echo "    make generate     Uruchom pipeline generowania Q&A (PDFy z data/)"
	@echo "    make gate         Sprawdź jakość datasetu (Quality Gate)"
	@echo "    make train        Uruchom SFT + DPO training (GPU)"
	@echo "    make export       Eksportuj model do GGUF + ZIP"
	@echo "    make chatbot      Uruchom Ollama + Open WebUI"
	@echo ""
	@echo "  LOGI"
	@echo "    make logs         Logi wszystkich serwisów"
	@echo "    make logs-api     Logi API"
	@echo "    make logs-ui      Logi frontendu (alias)"
	@echo "    make logs-frontend Logi frontendu"
	@echo "    make logs-trainer Logi kontenera treningowego"
	@echo ""
	@echo "  WERYFIKACJA"
	@echo "    make check-merge-conflicts  Sprawdź czy repo nie zawiera markerów konfliktów Git"
	@echo ""
	@echo "  Zmienne (opcjonalne):"
	@echo "    MODEL_NAME=my-model make export"
	@echo "    CHUNK_LIMIT=50    make generate"
	@echo "    SKIP_DPO=1        make train"
	@echo ""

# ---------------------------------------------------------------------------
# Setup & lifecycle
# ---------------------------------------------------------------------------

setup:
	@bash $(CTL) setup

up:
	@bash $(CTL) up

down:
	@docker compose down

rebuild:
	@docker compose build
	@bash $(CTL) up

status:
	@bash $(CTL) status

clean:
	@echo "UWAGA: To usunie wszystkie kontenery i wolumeny (dane zostaną skasowane)!"
	@read -p "Kontynuować? [y/N] " ans && [ "$$ans" = "y" ] || exit 1
	@docker compose --profile train --profile chatbot down -v
	@echo "Wyczyszczono."

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

full-run:
	@bash $(CTL) full-run

generate:
	@bash $(CTL) generate

gate:
	@bash $(CTL) gate

train:
	@bash $(CTL) train

export:
	@bash $(CTL) export

chatbot:
	@bash $(CTL) chatbot

# ---------------------------------------------------------------------------
# Logi
# ---------------------------------------------------------------------------

logs:
	@docker compose logs -f

logs-api:
	@docker compose logs -f foundry-api

logs-ui:
	@docker compose logs -f frontend

logs-frontend:
	@docker compose logs -f frontend

logs-trainer:
	@docker compose --profile train logs -f foundry-trainer

# ---------------------------------------------------------------------------
# Weryfikacja
# ---------------------------------------------------------------------------

check-merge-conflicts:
	@echo "Sprawdzam markery konfliktów Git..."
	@if rg -n --hidden --glob '!.git' '^(<<<<<<<|=======|>>>>>>>)' .; then \
		echo "❌ Wykryto markery konfliktów."; \
		exit 1; \
	else \
		echo "✅ Brak markerów konfliktów."; \
	fi
