# Audyt ulepszeń, poprawek i wzrostu platformy Synthetic Data Foundry

**Data audytu:** 2026-04-21 (UTC)  
**Zakres:** analiza pełnego drzewa repozytorium (`REPO_FULL_TREE.txt`) + przegląd kluczowych modułów backend/frontend/DevOps.

---

## 1) Executive summary (stan obecny)

Platforma ma mocny fundament architektoniczny (modułowy backend FastAPI, osobny frontend Next.js, pipeline treningowy i crawlerowy), ale jest w fazie **"feature-rich / hardening-poor"**: dużo funkcji, za mało mechanizmów jakości i operacyjnej spójności.

### Najważniejsze wnioski

1. **Najwyższe ryzyko operacyjne:** brak testów automatycznych (0 plików testowych), co utrudnia bezpieczne wdrażanie zmian.  
2. **Najwyższe ryzyko produktowe:** rozjazd API ↔ frontend (typy i autoryzacja), co może generować błędy runtime mimo poprawnej kompilacji.  
3. **Najwyższe ryzyko utrzymaniowe:** niespójności w warstwie operacyjnej (Makefile vs docker-compose, stare odwołania do serwisów/profili).  
4. **Największa szansa wzrostu:** wykorzystanie już istniejącej orkiestracji runów/logów i Gap Scouta do wdrożenia „enterprise-grade” observability, SLA i governance danych.

---

## 2) Co działa dobrze (mocne strony)

- **Czytelny podział domenowy projektu** (agents/api/training/pipeline/frontend/ui), który ułatwia skalowanie zespołowe.  
- **Centralna konfiguracja przez `pydantic-settings`** i szeroki zakres parametrów środowiskowych.  
- **Warstwa bezpieczeństwa dla endpointów krytycznych** przez `X-API-Key` + porównanie stałoczasowe (`hmac.compare_digest`).  
- **Rozbudowana orkiestracja procesów długotrwałych** (run status, logi, background tasks).  
- **Świadome elementy operacyjne**: healthcheck, GZip, CORS, scheduler, profile kontenerów.

---

## 3) Krytyczne luki i poprawki (priorytet P0/P1)

## P0 — jakość i niezawodność

### 3.1 Brak testów automatycznych
- W repozytorium nie ma katalogu/unit testów i brak jest plików testowych.
- Efekt: wysoka regresyjność przy zmianach API, pipeline i crawlerów.

**Rekomendacja (P0):**
- Wprowadzić minimalny pakiet testów: 
  - API smoke tests (health, docs, auth fail/success),
  - testy kontraktów odpowiedzi endpointów `/api/samples`, `/api/pipeline`, `/api/training`,
  - testy jednostkowe parserów/validatorów settings.

### 3.2 Ryzyko niedziałającego frontendu przez auth
- Kluczowe routery backendu (`documents`, `pipeline`, `training`) wymagają `X-API-Key`.
- Frontendowy wrapper `apiFetch` nie dokłada nagłówka `X-API-Key`.
- Efekt: frontend może dostawać 401/503 na podstawowych flow (upload, pipeline, training).

**Rekomendacja (P0):**
- Dodać obsługę admin key po stronie frontend (env + bezpieczny proxy route / server actions),
- rozdzielić endpointy operatorskie od user-facing i wdrożyć role/zakresy uprawnień.

## P1 — spójność DevOps

### 3.3 Niespójny Makefile względem compose
- `make logs-ui` odwołuje się do `foundry-ui`, którego nie ma w `docker-compose.yml`.
- `clean` używa profilu `gpu`, choć w compose brak usług z tym profilem (występuje tylko w komentarzach).

**Rekomendacja (P1):**
- zsynchronizować Makefile z realną definicją usług,
- dodać automatyczny check zgodności komend operacyjnych (np. skrypt walidacyjny w CI).

### 3.4 Konflikt portu 3000 (frontend vs open-webui)
- Frontend i Open WebUI mapują hosta na port 3000.
- Efekt: profil `chatbot` koliduje z frontendem i może powodować awarie uruchomienia.

**Rekomendacja (P1):**
- zmienić mapowanie Open WebUI na inny port hosta (np. 3001),
- opisać to jasno w README i Makefile.

---

## 4) Luki architektoniczne i techniczne (P1/P2)

### 4.1 Rozjazd kontraktów typów API ↔ frontend
- `frontend/lib/api.ts` definiuje `Sample.id` jako `number`, gdy backend zwraca UUID/string.
- Takie rozjazdy powodują błędy UI i utrudniają refaktoryzację.

**Rekomendacja (P1):**
- wygenerować klienta TS z OpenAPI (`/openapi.json`) albo wprowadzić wspólne DTO schema package.

### 4.2 Nadmiar odpowiedzialności `api/main.py`
- `api/main.py` łączy konfigurację lifecycle, scheduler, WebSub, middleware i routing.
- Trudniej testować i rozwijać niezależnie.

**Rekomendacja (P2):**
- wydzielić moduły bootstrap (`startup.py`, `scheduler.py`, `middlewares.py`),
- ograniczyć `main.py` do kompozycji aplikacji.

### 4.3 Dwie warstwy UI (Next.js + Streamlit)
- Repo posiada zarówno frontend Next.js, jak i rozbudowane strony Streamlit.
- Bez jasnej strategii to zwiększa koszt utrzymania i ryzyko niespójności funkcji.

**Rekomendacja (P2):**
- zdecydować model docelowy (np. Next.js jako production UI, Streamlit jako wewnętrzne narzędzie R&D),
- oznaczyć status komponentów i zakres wsparcia.

---

## 5) Plan wzrostu platformy (90 dni)

## Faza 1 (0–30 dni): Stabilizacja i bezpieczeństwo zmian
- Testy API + kontrakty odpowiedzi.
- Naprawa auth flow frontend↔API.
- Synchronizacja Makefile/compose i korekta konfliktów portów.
- Ustandaryzowany pipeline CI: lint + test + typecheck + smoke docker compose.

**KPI:**
- min. 35% pokrycia krytycznych modułów,
- 0 błędów typu „service not found” w komendach operatorskich,
- 100% krytycznych flow frontendowych działających z auth.

## Faza 2 (31–60 dni): Observability + governance danych
- Śledzenie jakości runów (time-to-run, fail reason taxonomy, retry rate).
- Dashboard SLO/SLA dla pipeline i training.
- Wersjonowanie datasetów + podpisy integralności + lineage.

**KPI:**
- MTTR < 30 min dla awarii pipeline,
- pełna identyfikowalność: run → batch → model artifact.

## Faza 3 (61–90 dni): Skalowanie produktu i monetyzacja B2B
- Feature flags per tenant (perspektywy, progi jakości, watermark policy).
- Polityki kosztowe (budżet tokenów / run, limit równoległości).
- Gotowe „pakiety wdrożeniowe” (compliance profile, raporty audytowe dla klienta).

**KPI:**
- skrócenie czasu onboardingu nowego klienta o 50%,
- +30% przepustowości runów przy tym samym koszcie infrastruktury.

---

## 6) Backlog rekomendowanych inicjatyw

### Quick wins (1 sprint)
- [ ] Naprawić `Makefile: logs-ui` i komendy profili.
- [ ] Ujednolicić typy `Sample` (UUID string) w frontendzie.
- [ ] Dodać `X-API-Key` flow do frontendu.
- [ ] Rozdzielić porty frontend/open-webui.

### Mid-term (2–4 sprinty)
- [ ] Generacja SDK frontend z OpenAPI.
- [ ] Refaktoryzacja `api/main.py` na moduły bootstrap.
- [ ] Testy integracyjne pipeline/training z mockami usług zewnętrznych.

### Long-term (kwartał)
- [ ] Multi-tenant billing + cost controls.
- [ ] Rejestrowanie polityk jakości i audytowalności per klient.
- [ ] API publiczne z limitami i planami taryfowymi.

---

## 7) Podsumowanie końcowe

Projekt ma potencjał na platformę klasy enterprise, ale **najpierw wymaga warstwy stabilizacji**: testów, zgodności kontraktów i uporządkowania operacyjnego. Po wykonaniu tych kroków można bezpiecznie przyspieszyć rozwój funkcji wzrostowych (multi-tenant, governance, cost controls), co realnie zwiększy wartość biznesową i przewidywalność dostarczania.
