"""
ui/app.py — Foundry Studio — główna aplikacja Streamlit.

Uruchomienie:
    streamlit run ui/app.py --server.port 8501

Nawigacja odbywa się przez sidebar (strony w ui/pages/).
"""

import os

import requests
import streamlit as st

# Konfiguracja strony — musi być pierwszym wywołaniem st.*
st.set_page_config(
    page_title="Foundry Studio",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8080")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
st.session_state.setdefault("api_url", API_URL)


def _get(url: str, timeout: int = 4) -> dict | list | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar — nawigacja i status serwisów
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏭 Foundry Studio")
    st.caption("Synthetic Data Foundry — AutoPilot")
    st.divider()

    # API health
    health = _get(f"{API_URL}/health")
    if health:
        st.success("API: połączono ✓")
    else:
        st.error(f"API: brak połączenia ✗")
        st.caption(f"Adres: {API_URL}")

    # Ollama health
    ollama_ok = _get(f"{OLLAMA_URL}/api/tags") is not None
    if ollama_ok:
        st.success("Ollama: aktywny ✓")
    else:
        st.warning("Ollama: niedostępny")
        st.caption("Uruchom: docker compose --profile chatbot up -d")

    st.divider()
    st.markdown("""
**Nawigacja:**
- 📄 Dokumenty — wgraj PDFy
- 🚀 AutoPilot — uruchom pipeline
- 📊 Dataset — przeglądaj Q&A
- 🔍 Review — zatwierdź próbki
- 🧠 Trening — fine-tuning modelu
- 💬 Chatbot — testuj i ewaluuj
    """)

# ---------------------------------------------------------------------------
# Strona główna — dashboard
# ---------------------------------------------------------------------------

st.title("🏭 Foundry Studio")
st.markdown(
    "Platforma do automatycznego tworzenia syntetycznych zbiorów danych "
    "i trenowania modeli językowych na niszowych dokumentach."
)

# ---------------------------------------------------------------------------
# Wiersz 1 — Dane
# ---------------------------------------------------------------------------

st.subheader("Dane")

col1, col2, col3, col4 = st.columns(4)

stats = _get(f"{API_URL}/api/samples/stats")
docs_data = _get(f"{API_URL}/api/documents")

with col1:
    total_docs = (docs_data or {}).get("total", "—")
    st.metric("Dokumenty", total_docs)

with col2:
    total_qa = (stats or {}).get("total", "—")
    st.metric("Wygenerowane Q&A", total_qa)

with col3:
    dpo = (stats or {}).get("dpo_pairs", "—")
    st.metric("Pary DPO", dpo)

with col4:
    avg = (stats or {}).get("avg_quality_score")
    st.metric(
        "Avg. score",
        f"{avg:.3f}" if avg is not None else "—",
        delta=f"+{(avg - 0.88):.3f} vs próg" if avg is not None else None,
        delta_color="normal",
    )

# ---------------------------------------------------------------------------
# Wiersz 2 — Pipeline + Training status
# ---------------------------------------------------------------------------

st.divider()
col_pipe, col_train = st.columns(2)

with col_pipe:
    st.subheader("Pipeline (ostatni run)")
    runs_data = _get(f"{API_URL}/api/pipeline/runs")
    if runs_data:
        last = runs_data[-1]
        status = last.get("status", "unknown")
        icon = {"done": "✅", "error": "❌", "running": "⏳", "starting": "⏳"}.get(status, "⏸️")
        elapsed = last.get("elapsed_seconds", 0)
        st.markdown(
            f"{icon} **{last.get('run_id', '—')}**  \n"
            f"Status: `{status}` | Czas: {int(elapsed // 60)}m {int(elapsed % 60)}s"
        )
        prog = last.get("progress_pct", 0)
        if prog:
            st.progress(int(prog) / 100, text=f"{prog:.0f}%")
        chunks = last.get("chunks_done", 0)
        records = last.get("records_done", 0)
        if chunks or records:
            st.caption(f"Chunki: {chunks} | Q&A: {records}")
    else:
        st.info("Brak uruchomionych pipeline'ów.")

with col_train:
    st.subheader("Trening (ostatni run)")
    train_runs = _get(f"{API_URL}/api/training/runs")
    if train_runs:
        last_t = train_runs[-1]
        status_t = last_t.get("status", "unknown")
        icon_t = {"done": "✅", "error": "❌", "running": "⏳", "starting": "⏳"}.get(status_t, "⏸️")
        elapsed_t = last_t.get("elapsed_seconds", 0)
        st.markdown(
            f"{icon_t} **{last_t.get('run_id', '—')}**  \n"
            f"Status: `{status_t}` | Czas: {int(elapsed_t // 60)}m {int(elapsed_t % 60)}s"
        )
    else:
        st.info("Brak uruchomionych runów treningowych.")

# ---------------------------------------------------------------------------
# Wiersz 3 — Perspektywy + Trudności
# ---------------------------------------------------------------------------

if stats and stats.get("total", 0) > 0:
    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Perspektywy Q&A")
        persp = stats.get("perspectives", {})
        total = stats.get("total", 1)
        for k, v in sorted(persp.items(), key=lambda x: -x[1]):
            pct = v / total * 100
            st.write(f"**{k}**: {v} ({pct:.1f}%)")
            st.progress(pct / 100)

    with c2:
        st.subheader("Trudność Q&A")
        diff = stats.get("difficulties", {})
        for k, v in sorted(diff.items(), key=lambda x: -x[1]):
            pct = v / total * 100
            st.write(f"**{k}**: {v} ({pct:.1f}%)")
            st.progress(pct / 100)

# ---------------------------------------------------------------------------
# Wiersz 4 — Aktywne modele Ollama
# ---------------------------------------------------------------------------

if ollama_ok:
    st.divider()
    st.subheader("Modele Ollama")
    models_data = _get(f"{OLLAMA_URL}/api/tags")
    if models_data:
        models = models_data.get("models", [])
        if models:
            cols = st.columns(min(len(models), 4))
            for i, m in enumerate(models[:4]):
                with cols[i]:
                    size_gb = m.get("size", 0) / 1e9
                    st.metric(m["name"], f"{size_gb:.2f} GB")
        else:
            st.info("Brak załadowanych modeli. Uruchom eksport i załaduj GGUF do Ollama.")
    else:
        st.warning("Nie udało się pobrać listy modeli z Ollama.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Foundry Studio | "
    f"API: {API_URL} | "
    f"Ollama: {OLLAMA_URL}"
)
