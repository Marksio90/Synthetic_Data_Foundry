# 🏭 Synthetic Data Foundry

![Wersja](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green.svg)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

**Synthetic Data Foundry** to zaawansowana, wielomodułowa platforma służąca do kompleksowego pozyskiwania, generowania, ewaluacji i przetwarzania danych syntetycznych, a także do automatycznego trenowania modeli językowych (LLM). Projekt integruje rozbudowany system agentowy (wielowarstwowe crawlery, weryfikatorów, symulatorów) z potężnym backendem analitycznym i dwoma interfejsami graficznymi (Next.js oraz Streamlit).

---

## ✨ Główne funkcjonalności

* **🕵️ Wielowarstwowe pozyskiwanie danych (Crawling & Ingestion):**
    * System warstwowy krawlerów (Layer A-E) z planistą (`scheduler`) i deduplikacją.
    * Integracja z WebSub dla otrzymywania treści w czasie rzeczywistym.
    * Moduły ekstrakcji m.in. z plików audio, dokumentów tekstowych i innych formatów.
* **🤖 Zaawansowany ekosystem agentów (Agent Framework):**
    * Sędziowie (`judge.py`) i recenzenci (`auto_reviewer.py`) do oceny jakości wygenerowanych lub zebranych treści.
    * Agenci symulujący konwersacje (`simulator.py`) i kalibrujący dane (`calibrator.py`).
    * Agenci odpowiedzialni za przestrzeganie zasad konstytucyjnych (Constitutional AI - `constitutional.py`).
    * Rozpoznawanie i poszukiwanie nowych tematów (`topic_scout.py`).
* **⚙️ Orkiestracja Potoków Przetwarzania (Pipelines):**
    * Grafowe przepływy pracy dla przetwarzania dokumentów (`pipeline/graph.py`).
    * Wbudowane wsparcie dla ukrytych znaków wodnych (Watermarking), które oznaczają wygenerowane syntetycznie dane jako bezpieczne lub przypisane do platformy (`pipeline/watermark.py`, `scripts/verify_watermark.py`).
* **🧠 Moduł Treningowy Modeli (Training):**
    * Zautomatyzowane procesy trenowania metodą SFT (Supervised Fine-Tuning) oraz DPO (Direct Preference Optimization).
    * Wbudowana automatyczna optymalizacja hiperparametrów (`auto_tuner.py`).
    * Inspektor sprzętu weryfikujący dostępność GPU (`hardware_inspector.py`).
    * Zarządzanie jakością ("Quality Gate") i automatyczny eksport wytrenowanych modeli na platformę Hugging Face (`hf_uploader.py`).
* **💻 Dwa Interfejsy Użytkownika:**
    * **Frontend (Next.js):** Nowoczesna aplikacja webowa udostępniająca widoki do obsługi datasetów, autopilota, chatbota i logów w czasie rzeczywistym.
    * **UI (Streamlit):** Alternatywny, analityczny panel kontrolny dla badaczy danych i inżynierów do zarządzania dokumentami, treningiem i ewaluacją (strony 1-6).

---

## 🏗 Architektura i struktura projektu

Projekt jest podzielony na niezależne mikro-moduły, z których każdy odpowiada za specyficzny etap cyklu życia danych syntetycznych.

```text
Synthetic_Data_Foundry/
├── agents/             # Główny katalog dla inteligentnych agentów
│   ├── crawlers/       # Skrypty odpowiedzialne za przeszukiwanie i pobieranie danych (warstwy A-E)
│   ├── extractors/     # Narzędzia do pozyskiwania tekstu z różnych źródeł (np. audio)
│   ├── judge.py / auto_reviewer.py # Agenci oceniający i kontrolujący jakość danych
│   └── hf_uploader.py  # Agent wypychający gotowe modele/datasety na Hugging Face
├── api/                # Serwer Backendowy
│   ├── main.py         # Punkt wejścia aplikacji FastAPI
│   ├── routers/        # Endpointy API (chatbot, documents, pipeline, scout, training)
│   └── db.py           # Obsługa połączeń z bazą danych
├── db/                 # Zarządzanie strukturą i logiką bazy danych (Modele, Repozytoria SQLAlchemy)
├── docker/             # Definicje środowisk Docker dla różnych modułów (API, Frontend, UI, Trainer)
├── frontend/           # Aplikacja Next.js (React, Tailwind CSS, TypeScript)
├── init/               # Skrypty inicjalizujące bazę danych (01_schema.sql)
├── pipeline/           # Definicje grafów przetwarzania (LangGraph/Custom) i watermarking
├── scripts/            # Skrypty narzędziowe (np. foundry-ctl.sh, verify_watermark.py)
├── training/           # Komponenty do uczenia i oceny modeli LLM (SFT, DPO, ewaluacja)
├── ui/                 # Interfejs badawczy Streamlit
└── utils/              # Funkcje pomocnicze (ponawianie żądań, klasyfikacja, deduplikacja)
