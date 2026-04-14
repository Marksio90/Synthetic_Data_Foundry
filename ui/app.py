"""
ui/app.py — Foundry Studio — główna aplikacja Streamlit.

Uruchomienie:
    streamlit run ui/app.py --server.port 8501

Nawigacja odbywa się przez sidebar (strony w ui/pages/).
"""

import os

import streamlit as st

# Konfiguracja strony — musi być pierwszym wywołaniem st.*
st.set_page_config(
    page_title="Foundry Studio",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8080")
st.session_state.setdefault("api_url", API_URL)

# ---------------------------------------------------------------------------
# Sidebar — nawigacja i status API
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏭 Foundry Studio")
    st.caption("Synthetic Data Foundry — AutoPilot")
    st.divider()

    # Sprawdź połączenie z API
    try:
        import requests
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok:
            st.success("API: połączono ✓")
        else:
            st.warning(f"API: błąd {r.status_code}")
    except Exception:
        st.error("API: brak połączenia ✗")
        st.caption(f"Oczekiwany adres: {API_URL}")

    st.divider()
    st.markdown("""
**Nawigacja:**
- 📄 Dokumenty — wgraj PDFy
- 🚀 AutoPilot — uruchom pipeline
- 📊 Dataset — przeglądaj Q&A
    """)

# ---------------------------------------------------------------------------
# Strona główna — dashboard
# ---------------------------------------------------------------------------

st.title("🏭 Foundry Studio")
st.markdown(
    "Platforma do automatycznego tworzenia syntetycznych zbiorów danych "
    "i trenowania modeli językowych na niszowych dokumentach."
)

col1, col2, col3 = st.columns(3)

try:
    import requests
    stats = requests.get(f"{API_URL}/api/samples/stats", timeout=5).json()
    docs  = requests.get(f"{API_URL}/api/documents", timeout=5).json()

    with col1:
        st.metric("Dokumenty", docs.get("total", 0))
    with col2:
        st.metric("Wygenerowane Q&A", stats.get("total", 0))
    with col3:
        st.metric("Avg. score sędziego",
                  f"{stats.get('avg_quality_score', 0):.2f}" if stats.get("total") else "—")

    if stats.get("total", 0) > 0:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Perspektywy")
            for k, v in stats.get("perspectives", {}).items():
                st.write(f"**{k}**: {v}")
        with c2:
            st.subheader("Trudność")
            for k, v in stats.get("difficulties", {}).items():
                st.write(f"**{k}**: {v}")
except Exception:
    with col1:
        st.metric("Dokumenty", "—")
    with col2:
        st.metric("Q&A", "—")
    with col3:
        st.metric("Score", "—")

st.info("Użyj nawigacji po lewej stronie aby rozpocząć pracę.")
