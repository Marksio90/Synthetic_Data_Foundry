"""
ui/pages/2_AutoPilot.py — Jeden przycisk, pełna automatyzacja.
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = st.session_state.get("api_url", os.getenv("API_URL", "http://localhost:8080"))

st.set_page_config(page_title="AutoPilot — Foundry Studio", page_icon="🚀", layout="wide")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px 3px;
}
.badge-green  { background: #1a4731; color: #4ade80; border: 1px solid #166534; }
.badge-yellow { background: #422006; color: #fbbf24; border: 1px solid #92400e; }
.badge-blue   { background: #172554; color: #60a5fa; border: 1px solid #1e40af; }
.badge-gray   { background: #1f2937; color: #9ca3af; border: 1px solid #374151; }
.badge-red    { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
.param-card {
    background: #1e2533;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
}
.param-label { color: #9ca3af; font-size: 12px; margin-bottom: 2px; }
.param-value { color: #f9fafb; font-size: 22px; font-weight: 700; }
.param-desc  { color: #6b7280; font-size: 11px; margin-top: 2px; }
.warn-box {
    background: #422006;
    border: 1px solid #92400e;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    color: #fbbf24;
    font-size: 13px;
}
.info-box {
    background: #172554;
    border: 1px solid #1e40af;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    color: #93c5fd;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AutoPilot")
st.caption("Analizuje dokumenty, kalibruje parametry i generuje dataset — w pełni automatycznie.")

# ---------------------------------------------------------------------------
# Fetch document list
# ---------------------------------------------------------------------------
try:
    docs_resp = requests.get(f"{API_URL}/api/documents", timeout=10)
    all_docs = docs_resp.json().get("documents", []) if docs_resp.ok else []
except Exception:
    all_docs = []
    st.error("Nie można połączyć się z API. Sprawdź czy stack jest uruchomiony.")

if not all_docs:
    st.warning("Brak dokumentów. Przejdź do strony **📄 Dokumenty** i wgraj pliki PDF.")
    st.stop()

# ---------------------------------------------------------------------------
# STEP 1 — Document selection
# ---------------------------------------------------------------------------
st.markdown("### 1 — Wybierz dokumenty")
all_names = [d["filename"] for d in all_docs]

col_sel, col_info = st.columns([3, 1])
with col_sel:
    selected = st.multiselect(
        "Dokumenty do przetworzenia:",
        options=all_names,
        default=all_names,
        label_visibility="collapsed",
    )
with col_info:
    st.markdown(f"<br><span style='color:#9ca3af'>{len(selected)} / {len(all_names)} plików</span>",
                unsafe_allow_html=True)

if not selected:
    st.info("Wybierz przynajmniej jeden dokument.")
    st.stop()

# ---------------------------------------------------------------------------
# STEP 2 — Analysis preview
# ---------------------------------------------------------------------------
st.markdown("### 2 — Analiza i kalibracja")

col_btn, col_hint = st.columns([2, 5])
with col_btn:
    do_analyze = st.button("🔍 Analizuj dokumenty", use_container_width=True,
                           help="Szybki podgląd decyzji AutoPilota bez uruchamiania pipeline'u")
with col_hint:
    st.markdown("<br><span style='color:#6b7280; font-size:13px'>Opcjonalnie — AutoPilot uruchomi analizę automatycznie przy starcie</span>",
                unsafe_allow_html=True)

if do_analyze:
    with st.spinner("Analizuję dokumenty..."):
        try:
            r = requests.post(
                f"{API_URL}/api/pipeline/analyze",
                json={"filenames": selected},
                timeout=30,
            )
            if r.ok:
                st.session_state["analysis_preview"] = r.json()
                st.session_state.pop("analysis_for", None)
            else:
                st.error(f"Błąd analizy: {r.text}")
        except Exception as e:
            st.error(str(e))

preview = st.session_state.get("analysis_preview")
if preview:
    calib = preview.get("calibration") or {}
    decisions = preview.get("auto_decisions", [])
    lang = preview.get("language", "")
    domain = preview.get("domain_label", "—")
    confidence = preview.get("domain_confidence", 0.0)
    perspectives = preview.get("perspectives", [])

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: Decisions ──────────────────────────────────────────────────────
    with col_left:
        st.markdown("**Decyzje AutoPilota**")

        # Language badge
        if lang:
            lang_upper = lang.upper()
            if lang.lower() == "pl":
                st.markdown(f'<span class="badge badge-green">🌐 Język: {lang_upper} — natywny</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge badge-yellow">🌐 Język: {lang_upper} → tłumaczenie</span>',
                            unsafe_allow_html=True)

        # Domain badge
        conf_pct = int(confidence * 100)
        conf_color = "badge-green" if conf_pct >= 70 else ("badge-yellow" if conf_pct >= 40 else "badge-red")
        st.markdown(
            f'<span class="badge {conf_color}">🏷 {domain} ({conf_pct}%)</span>',
            unsafe_allow_html=True,
        )

        # Perspectives
        if perspectives:
            persp_html = " ".join(
                f'<span class="badge badge-blue">👤 {p}</span>' for p in perspectives
            )
            st.markdown(persp_html, unsafe_allow_html=True)

        st.markdown("")

        # Decisions — separate warnings from info
        warnings = [d for d in decisions if any(w in d for w in ["Niska", "brak", "błąd", "⚠"])]
        infos = [d for d in decisions if d not in warnings]

        for d in infos:
            # strip leading emoji duplicates
            text = d.lstrip("🤖 ").strip()
            st.markdown(f'<div class="info-box">🤖 {text}</div>', unsafe_allow_html=True)
        for d in warnings:
            text = d.lstrip("⚠️ ").strip()
            st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

    # ── Right: Calibration params ────────────────────────────────────────────
    with col_right:
        st.markdown("**Skalibrowane parametry**")
        if calib:
            qt   = calib.get("quality_threshold", "—")
            mt   = calib.get("max_turns", "—")
            ar   = calib.get("adversarial_ratio", "—")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="param-card">
                    <div class="param-label">quality_threshold</div>
                    <div class="param-value">{qt}</div>
                    <div class="param-desc">Minimalny wynik oceny</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="param-card">
                    <div class="param-label">max_turns</div>
                    <div class="param-value">{mt}</div>
                    <div class="param-desc">Tury konwersacji</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="param-card">
                    <div class="param-label">adversarial_ratio</div>
                    <div class="param-value">{ar}</div>
                    <div class="param-desc">Pytania trudne</div>
                </div>""", unsafe_allow_html=True)

            reasoning = calib.get("reasoning", [])
            if reasoning:
                with st.expander("📋 Uzasadnienie kalibracji", expanded=False):
                    for line in reasoning:
                        st.markdown(f"- {line}")
        else:
            st.markdown('<div class="info-box">Parametry zostaną dobrane automatycznie przy starcie.</div>',
                        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STEP 3 — Optional overrides + Run
# ---------------------------------------------------------------------------
st.markdown("### 3 — Uruchom")

with st.expander("⚙️ Zaawansowane — nadpisz parametry (domyślnie: auto)", expanded=False):
    st.caption("Zostaw domyślne wartości jeśli nie masz powodu zmieniać — AutoPilot dobierze optymalne.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        use_threshold = st.toggle("Ręczny quality_threshold", value=False)
        manual_threshold = st.slider(
            "quality_threshold", min_value=0.5, max_value=1.0,
            value=0.82, step=0.01,
            disabled=not use_threshold,
            help="Minimalna jakość próbki żeby trafiła do datasetu",
        ) if use_threshold else None
    with col_b:
        use_turns = st.toggle("Ręczny max_turns", value=False)
        manual_turns = st.select_slider(
            "max_turns", options=[1, 2, 3, 4, 5],
            value=3,
            disabled=not use_turns,
            help="Liczba tur konwersacji na chunk (1 = single Q&A, 3 = zalecane)",
        ) if use_turns else None
    with col_c:
        use_adversarial = st.toggle("Ręczny adversarial_ratio", value=False)
        manual_adversarial = st.slider(
            "adversarial_ratio", min_value=0.0, max_value=0.3,
            value=0.10, step=0.01,
            disabled=not use_adversarial,
            help="Frakcja trudnych / mylących pytań",
        ) if use_adversarial else None

    st.divider()
    col_d, col_e = st.columns(2)
    with col_d:
        chunk_limit = st.number_input(
            "chunk_limit", min_value=0, max_value=10000, value=0, step=10,
            help="Ogranicz do N chunków (0 = wszystkie). Przydatne do testów.",
        )
    with col_e:
        custom_batch_id = st.text_input(
            "Batch ID", placeholder="np. banking-2026-v1",
            help="Puste = auto-generowany UUID",
        )

# ── Summary before launch ────────────────────────────────────────────────────
if st.session_state.get("active_run_id") is None:
    summary_parts = [f"**{len(selected)}** doc(s)"]
    if manual_threshold is not None:
        summary_parts.append(f"threshold={manual_threshold}")
    if manual_turns is not None:
        summary_parts.append(f"turns={manual_turns}")
    if chunk_limit:
        summary_parts.append(f"limit={chunk_limit} chunków")
    st.caption("Uruchomi: " + " · ".join(summary_parts))

    if st.button("🚀 FULL AUTO RUN", type="primary", use_container_width=True):
        payload = {
            "filenames": selected,
            "batch_id": custom_batch_id or None,
            "chunk_limit": chunk_limit,
            "quality_threshold": manual_threshold,
            "max_turns": manual_turns,
            "adversarial_ratio": manual_adversarial,
        }
        with st.spinner("Uruchamiam AutoPilota..."):
            try:
                r = requests.post(f"{API_URL}/api/pipeline/run", json=payload, timeout=15)
                if r.ok:
                    data = r.json()
                    st.session_state["active_run_id"] = data["run_id"]
                    st.session_state["active_batch_id"] = data["batch_id"]
                    st.session_state.pop("analysis_preview", None)
                    st.rerun()
                else:
                    st.error(f"Błąd startu: {r.text}")
            except Exception as e:
                st.error(str(e))
else:
    if st.button("⏹ Anuluj / Nowy run", type="secondary", use_container_width=True):
        st.session_state["active_run_id"] = None
        st.session_state.pop("analysis_preview", None)
        st.session_state.pop("log_lines", None)
        st.session_state.pop("log_offset", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Progress display (polling)
# ---------------------------------------------------------------------------
if st.session_state.get("active_run_id"):
    run_id    = st.session_state["active_run_id"]
    batch_id  = st.session_state.get("active_batch_id", "")

    st.divider()
    st.markdown(f"### 📡 Pipeline — batch: `{batch_id}`")

    status_ph  = st.empty()
    metrics_ph = st.empty()
    log_ph     = st.empty()
    log_offset = st.session_state.get("log_offset", 0)

    try:
        s_resp = requests.get(f"{API_URL}/api/pipeline/status/{run_id}", timeout=5)
        status_data = s_resp.json() if s_resp.ok else {}
    except Exception:
        status_data = {}

    status       = status_data.get("status", "unknown")
    pct          = status_data.get("progress_pct", 0)
    chunks_done  = status_data.get("chunks_done", 0)
    chunks_total = status_data.get("chunks_total", 0)
    records      = status_data.get("records_written", 0)
    dpo          = status_data.get("dpo_pairs", 0)
    elapsed      = status_data.get("elapsed_seconds", 0)

    with status_ph.container():
        if status == "done":
            st.success(f"✅ Pipeline zakończony! Czas: {elapsed:.0f}s")
            st.session_state["active_run_id"] = None
            st.session_state["log_offset"] = 0
        elif status == "error":
            st.error(f"❌ Błąd: {status_data.get('error', 'nieznany')}")
            st.session_state["active_run_id"] = None
            st.session_state["log_offset"] = 0
        else:
            st.progress(int(pct) / 100, text=f"⚙️ {status} — {elapsed:.0f}s")

    with metrics_ph.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunki",      f"{chunks_done}/{chunks_total}" if chunks_total else str(chunks_done))
        m2.metric("Postęp",      f"{pct}%")
        m3.metric("Q&A gotowych", records)
        m4.metric("Pary DPO",    dpo)

    try:
        l_resp = requests.get(
            f"{API_URL}/api/pipeline/log/{run_id}",
            params={"offset": log_offset, "limit": 100},
            timeout=5,
        )
        if l_resp.ok:
            log_data   = l_resp.json()
            new_lines  = log_data.get("lines", [])
            total_lines = log_data.get("total_lines", 0)
            existing   = st.session_state.get("log_lines", [])
            existing.extend(new_lines)
            st.session_state["log_lines"]  = existing
            st.session_state["log_offset"] = total_lines
    except Exception:
        pass

    all_log = st.session_state.get("log_lines", [])
    with log_ph.container():
        st.markdown("**Log na żywo:**")
        st.code("\n".join(all_log[-150:]), language="text")

    if status in ("starting", "running", "unknown"):
        time.sleep(2)
        st.rerun()
    else:
        st.session_state.pop("log_lines", None)
        st.session_state.pop("log_offset", None)
