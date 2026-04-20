# Ultra‑PRO Enterprise Upgrade Plan — Synthetic Data Foundry

## Cel dokumentu
Ten plan to **praktyczna mapa modernizacji** dopasowana do obecnej architektury repo (pipeline grafowy, multi-agent, quality gate, FastAPI, trening SFT/DPO). Zakłada wdrożenia iteracyjne, z szybkim ROI i kontrolą ryzyka.

---

## 1) Ulepszenia algorytmowe (najwyższy wpływ na jakość)

### A. Adaptive Retrieval Scoring + Self-Reranking
**Dlaczego:** obecny pipeline używa hybrydy wektor/BM25 i stałych wag. To działa, ale nie adaptuje się do typu pytania i jakości kontekstu.

**Propozycja:**
- Dodać lekki reranker punktujący top‑k chunków na podstawie: zgodności semantycznej, pokrycia encji prawnych, aktualności dokumentu i rozrzutu źródeł.
- Dynamicznie zmieniać wagi vector/BM25 w zależności od klasy pytania (`factual`, `procedural`, `comparative`) i długości zapytania.
- Zapisywać telemetryczne feature’y retrievalu do późniejszego auto-tuningu.

**Integracja:**
- `agents/expert.py` (retrieve_context)
- `config/settings.py` (progi i wagi dynamiczne)
- `db/models.py` + `db/repository.py` (telemetria retrieval)

**KPI:** +8–15% grounding_score, -20% halucynacji przy pytaniach trudnych.

### B. Judge Ensemble + Uncertainty Calibration
**Dlaczego:** pojedynczy sędzia z fallbackiem jest mocny, ale można poprawić stabilność oceny i ograniczyć false positives/false negatives.

**Propozycja:**
- Tryb „2-stage judge”: szybka ocena + druga ocena tylko dla próbek z niską pewnością lub dużą rozbieżnością subscore.
- Wprowadzić kalibrację (isotonic/Platt) na bazie historycznych decyzji `human_reviewed`.
- W quality gate liczyć **kalibrowany score**, nie surowy.

**Integracja:**
- `agents/judge.py`
- `agents/auto_reviewer.py`
- `training/quality_gate.py`

**KPI:** +10–20% trafności auto-approve, mniej odrzuceń wartościowych próbek.

### C. Active Learning Loop dla strefy „grey"
**Dlaczego:** strefa szara to największa dźwignia jakościowa.

**Propozycja:**
- Priorytetyzować do przeglądu człowieka próbki o najwyższej niepewności i największym wpływie na model (informative sampling).
- Po decyzji człowieka automatycznie aktualizować progi i reguły auto-review.

**Integracja:**
- `agents/auto_reviewer.py`
- `ui/app.py` (kolejka review z rankingiem)
- `training/auto_tuner.py` (aktualizacja progów)

**KPI:** skrócenie czasu review o 30–50% przy tym samym wolumenie.

### D. Multi-turn Curriculum + Difficulty Shaping
**Dlaczego:** obecnie multi-turn jest kontrolowany progiem procentowym, ale bez strategii progresji trudności.

**Propozycja:**
- Generować zestawy według poziomów trudności (L1–L4), gdzie L3/L4 wymuszają cross-paragraph reasoning.
- Zwiększać udział trudnych próbek adaptacyjnie, gdy model osiąga stabilny wynik na L1/L2.

**Integracja:**
- `agents/simulator.py`
- `pipeline/graph.py`
- `training/evaluate.py`

**KPI:** poprawa generalizacji i robustness po fine-tuningu.

---

## 2) Ulepszenia agentowe (architektura i niezawodność)

### A. Agent Contract Layer (typed I/O)
**Dlaczego:** przy dużej liczbie agentów rośnie ryzyko „dryfu” pól w stanie i cichych regresji.

**Propozycja:**
- Wprowadzić jawne kontrakty wejście/wyjście per agent (Pydantic models + walidatory).
- Dodać „state sanitizer” przed i po każdym węźle grafu.

**Integracja:**
- `pipeline/state.py`
- `pipeline/graph.py`
- `agents/*.py`

**KPI:** znaczący spadek błędów integracyjnych między agentami.

### B. Policy Engine dla Constitutional AI
**Dlaczego:** zasady konstytucyjne są skuteczne, ale warto je parametryzować domenowo (np. ESG, compliance, med/legal).

**Propozycja:**
- Wydzielić polityki do wersjonowanych plików YAML/JSON.
- Per-klient/per-batch włączać różne profile zasad.

**Integracja:**
- `agents/constitutional.py`
- `config/settings.py`
- nowy katalog `policies/`

**KPI:** większa kontrola compliance i łatwiejszy audyt.

### C. Agentic Retry Intelligence
**Dlaczego:** obecne retry jest technicznie poprawne, ale nie odróżnia semantycznie przyczyn porażki.

**Propozycja:**
- Retry policy zależne od typu błędu: retrieval_fail, judge_parse_fail, rate_limit, hallucination_risk.
- Dla retry semantycznego: modyfikacja promptu lub dołączenie innego kontekstu, zamiast ślepego powtórzenia.

**Integracja:**
- `pipeline/graph.py`
- `agents/expert.py`
- `agents/judge.py`

**KPI:** mniej „pustych retry”, lepsza skuteczność przy tej samej liczbie prób.

---

## 3) Ulepszenia kontekstowe (RAG, pamięć, governance)

### A. Context Provenance Graph
**Dlaczego:** enterprise wymaga ścieżki dowodowej „skąd ta odpowiedź”.

**Propozycja:**
- Zapisywać graf pochodzenia: dokument → chunk → odpowiedź → ocena sędziego → decyzja review.
- Eksportować „audit bundle” do JSON dla każdej próbki.

**Integracja:**
- `db/models.py`
- `db/repository.py`
- `utils/datacard.py`

**KPI:** pełny audytowalny lineage (compliance-ready).

### B. Temporal-Aware Context Windows
**Dlaczego:** dla regulacji i newsów czas obowiązywania treści ma krytyczne znaczenie.

**Propozycja:**
- Dodać metadane effective_date / publication_date / superseded_by.
- Retrieval ma preferować aktualnie obowiązujące źródła i oznaczać konflikt wersji.

**Integracja:**
- `agents/ingestor.py`
- `agents/doc_analyzer.py`
- `agents/expert.py`

**KPI:** mniej odpowiedzi opartych o nieaktualne przepisy.

### C. Tenant-Aware Context Isolation (B2B)
**Dlaczego:** przy wielu klientach kluczowa jest separacja wiedzy i polityk.

**Propozycja:**
- Namespace per tenant dla indeksów, datasetów i zasad konstytucyjnych.
- Globalny guardrail, który blokuje retrieval cross-tenant.

**Integracja:**
- `api/security.py`
- `db/models.py`
- `pipeline/graph.py`

**KPI:** bezpieczeństwo danych klasy enterprise.

---

## 4) Platforma, operacyjność i koszty

### A. Observability 360
- OpenTelemetry traces dla każdego kroku grafu.
- Metryki kosztowe per batch/per tenant/per model.
- SLO dashboard: latency p95, success rate, rejection rate, retry efficiency.

### B. FinOps & Model Routing
- Routing modeli wg klasy zadania (cheap/fast vs high-accuracy).
- Budżety kosztowe i automatyczne „cost brakes” po przekroczeniu limitu.
- Raport koszt/jakość jako część quality gate.

### C. Hardening produkcyjny
- Circuit breaker dla providerów LLM.
- Idempotency keys dla endpointów uruchamiających pipeline.
- Dead-letter queue dla zadań nieudanych po wszystkich retrach.

---

## 5) Plan wdrożenia 30/60/90 dni

### 0–30 dni (Quick Wins)
1. Agent Contract Layer + walidacja stanu.
2. Rozszerzenie telemetry (retrieval/judge/retry).
3. Grey-zone prioritization w auto-review.
4. KPI dashboard v1.

### 31–60 dni (Core Intelligence)
1. Adaptive retrieval + reranker.
2. 2-stage judge + kalibracja confidence.
3. Curriculum generation L1–L4.

### 61–90 dni (Enterprise Scale)
1. Tenant isolation + policy profiles.
2. Context provenance graph + audit bundle.
3. FinOps routing i automatyczne limity kosztowe.

---

## 6) Priorytety wykonawcze (Top 10)
1. **Adaptive retrieval reranker**
2. **Judge calibration + ensemble**
3. **Grey-zone active learning**
4. **Typed agent contracts**
5. **Context provenance lineage**
6. **Temporal validity metadata**
7. **Tenant-aware isolation**
8. **SLO + cost observability**
9. **Smart retry policies**
10. **Curriculum difficulty shaping**

---

## 7) Ryzyka i mitigacje
- **Ryzyko:** większa złożoność pipeline.  
  **Mitigacja:** feature flags + rollout canary per moduł.
- **Ryzyko:** wzrost kosztów przy ensemble judge.  
  **Mitigacja:** włączać stage‑2 tylko dla low-confidence.
- **Ryzyko:** przeciążenie review teamu.  
  **Mitigacja:** active learning ranking + SLA dla kolejek.

---

## 8) Definicja sukcesu (Enterprise DoD)
- +12% jakości końcowej (ważony composite score).
- -25% halucynacji krytycznych.
- -30% czasu manualnego review na 1000 próbek.
- 100% audytowalnych rekordów z lineage.
- Przewidywalny koszt per rekord (odchylenie < 15%).

---

Jeśli chcesz, mogę od razu przygotować **konkretny backlog techniczny (epiki → stories → zadania) pod ten plan**, wraz z priorytetami, estymacjami i propozycją branch strategy dla zespołu.
