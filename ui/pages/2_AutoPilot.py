"""
ui/pages/2_AutoPilot.py — Jeden przycisk, pełna automatyzacja.

Flow:
  1. Wybierz dokumenty z listy (lub wszystkie)
  2. Kliknij [🚀 FULL AUTO RUN]
  3. AutoPilot analizuje domenę, kalibruje parametry, uruchamia pipeline
  4. Obserwuj live log i postęp
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = st.session_state.get("api_url", os.getenv("API_URL", "http://localhost:8080"))

st.set_page_config(page_title="AutoPilot — Foundry Studio", page_icon="🚀", layout="wide")
st.title("🚀 AutoPilot")
st.caption("Jeden przycisk — automatyczne generowanie datasetu.")

# ---------------------------------------------------------------------------
# Fetch document list
# ---------------------------------------------------------------------------

try:
    docs_resp = requests.get(f"{API_URL}/api/documents", timeout=10)
    all_docs = docs_resp.json().get("documents", []) if docs_resp.ok else []
except Exception:
    all_docs = []
    st.error("Nie można połączyć się z API.")

if not all_docs:
    st.warning("Brak dokumentów. Przejdź do strony 📄 Dokumenty i wgraj pliki PDF.")
    st.stop()

# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------

st.subheader("1. Wybierz dokumenty")
all_names = [d["filename"] for d in all_docs]
selected = st.multiselect(
    "Dokumenty do przetworzenia:",
    options=all_names,
    default=all_names,
    help="Domyślnie wszystkie — AutoPilot wykryje domenę automatycznie.",
)

if not selected:
    st.info("Wybierz przynajmniej jeden dokument.")
    st.stop()

# ---------------------------------------------------------------------------
# AutoPilot preview (DocAnalyzer + Calibrator — live before run)
# ---------------------------------------------------------------------------

st.subheader("2. Podgląd decyzji AutoPilota")

if st.button("🔍 Analizuj dokumenty (podgląd)", help="Szybka analiza bez uruchamiania pipeline'u"):
    with st.spinner("Analizuję..."):
        try:
            r = requests.post(
                f"{API_URL}/api/pipeline/analyze",
                json={"filenames": selected},
                timeout=15,
            )
            if r.ok:
                st.session_state["analysis_preview"] = r.json()
            else:
                st.error(f"Błąd analizy: {r.text}")
        except Exception as e:
            st.error(str(e))

preview = st.session_state.get("analysis_preview")
if preview:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decyzje AutoPilota:**")
        for decision in preview.get("auto_decisions", []):
            icon = "⚠️" if "Niska" in decision or "brak" in decision.lower() else "🤖"
            st.write(f"{icon} {decision}")

    with col2:
        calib = preview.get("calibration", {})
        if calib:
            st.markdown("**Skalibrowane parametry:**")
            st.metric("quality_threshold", calib.get("quality_threshold", "—"))
            st.metric("max_turns", calib.get("max_turns", "—"))
            st.metric("adversarial_ratio", calib.get("adversarial_ratio", "—"))
            with st.expander("Uzasadnienie kalibracji"):
                for r_line in calib.get("reasoning", []):
                    st.write(f"• {r_line}")

st.divider()

# ---------------------------------------------------------------------------
# Optional overrides (collapsed by default)
# ---------------------------------------------------------------------------

with st.expander("⚙️ Opcjonalne ustawienia ręczne (pozostaw puste = auto)", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        manual_threshold = st.number_input(
            "quality_threshold", min_value=0.5, max_value=1.0,
            value=0.0, step=0.01,
            help="0 = auto (zalecane)"
        )
    with col_b:
        manual_turns = st.number_input(
            "max_turns", min_value=1, max_value=5,
            value=0, step=1,
            help="0 = auto"
        )
    with col_c:
        manual_adversarial = st.number_input(
            "adversarial_ratio", min_value=0.0, max_value=0.3,
            value=0.0, step=0.01,
            help="0 = auto"
        )
    with col_d:
        chunk_limit = st.number_input(
            "chunk_limit (0=wszystkie)", min_value=0, max_value=10000,
            value=0, step=10,
        )
    custom_batch_id = st.text_input(
        "Batch ID (puste = auto-generowany)", placeholder="np. banking-2026-v1"
    )

# ---------------------------------------------------------------------------
# THE button
# ---------------------------------------------------------------------------

st.subheader("3. Uruchom")

if "active_run_id" not in st.session_state:
    st.session_state["active_run_id"] = None

run_in_progress = st.session_state["active_run_id"] is not None

if not run_in_progress:
    if st.button("🚀 FULL AUTO RUN", type="primary", use_container_width=True):
        payload = {
            "filenames": selected,
            "batch_id": custom_batch_id or None,
            "chunk_limit": chunk_limit,
            "quality_threshold": manual_threshold if manual_threshold > 0 else None,
            "max_turns": manual_turns if manual_turns > 0 else None,
            "adversarial_ratio": manual_adversarial if manual_adversarial > 0 else None,
        }
        with st.spinner("Uruchamiam AutoPilota..."):
            try:
                r = requests.post(f"{API_URL}/api/pipeline/run", json=payload, timeout=15)
                if r.ok:
                    data = r.json()
                    st.session_state["active_run_id"] = data["run_id"]
                    st.session_state["active_batch_id"] = data["batch_id"]
                    st.rerun()
                else:
                    st.error(f"Błąd startu: {r.text}")
            except Exception as e:
                st.error(str(e))
else:
    if st.button("⏹ Anuluj / Nowy run", type="secondary", use_container_width=True):
        st.session_state["active_run_id"] = None
        st.session_state.pop("analysis_preview", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Progress display (polling)
# ---------------------------------------------------------------------------

if st.session_state.get("active_run_id"):
    run_id = st.session_state["active_run_id"]
    batch_id = st.session_state.get("active_batch_id", "")

    st.divider()
    st.subheader(f"📡 Pipeline w toku — batch: {batch_id}")

    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    log_placeholder = st.empty()
    log_offset = st.session_state.get("log_offset", 0)

    # Fetch status
    try:
        s_resp = requests.get(f"{API_URL}/api/pipeline/status/{run_id}", timeout=5)
        status_data = s_resp.json() if s_resp.ok else {}
    except Exception:
        status_data = {}

    status = status_data.get("status", "unknown")
    pct = status_data.get("progress_pct", 0)
    chunks_done = status_data.get("chunks_done", 0)
    chunks_total = status_data.get("chunks_total", 0)
    records = status_data.get("records_written", 0)
    dpo = status_data.get("dpo_pairs", 0)
    elapsed = status_data.get("elapsed_seconds", 0)

    # Status badge
    with status_placeholder.container():
        if status == "done":
            st.success("✅ Pipeline zakończony!")
            st.session_state["active_run_id"] = None
            st.session_state["log_offset"] = 0
        elif status == "error":
            st.error(f"❌ Błąd: {status_data.get('error', 'nieznany')}")
            st.session_state["active_run_id"] = None
            st.session_state["log_offset"] = 0
        else:
            st.info(f"⚙️ Status: **{status}** — {elapsed:.0f}s")
            st.progress(pct / 100)

    # Metrics row
    with metrics_placeholder.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunki", f"{chunks_done}/{chunks_total}" if chunks_total else str(chunks_done))
        m2.metric("Postęp", f"{pct}%")
        m3.metric("Q&A gotowych", records)
        m4.metric("Pary DPO", dpo)

    # Fetch new log lines
    try:
        l_resp = requests.get(
            f"{API_URL}/api/pipeline/log/{run_id}",
            params={"offset": log_offset, "limit": 100},
            timeout=5,
        )
        if l_resp.ok:
            log_data = l_resp.json()
            new_lines = log_data.get("lines", [])
            total_lines = log_data.get("total_lines", 0)

            # Store accumulated lines in session state
            existing = st.session_state.get("log_lines", [])
            existing.extend(new_lines)
            st.session_state["log_lines"] = existing
            st.session_state["log_offset"] = total_lines
    except Exception:
        pass

    all_log = st.session_state.get("log_lines", [])
    with log_placeholder.container():
        st.markdown("**Log na żywo:**")
        log_text = "\n".join(all_log[-150:])  # ostatnie 150 linii
        st.code(log_text, language="text")

    # Auto-refresh while running
    if status in ("starting", "running", "unknown"):
        time.sleep(2)
        st.rerun()
    else:
        # Clean up session state
        st.session_state.pop("log_lines", None)
        st.session_state.pop("log_offset", None)
