"""
ui/pages/3_Dataset.py — Przeglądarka wygenerowanych Q&A pairs.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = st.session_state.get("api_url", os.getenv("API_URL", "http://localhost:8080"))

st.set_page_config(page_title="Dataset — Foundry Studio", page_icon="📊", layout="wide")
st.title("📊 Dataset")
st.caption("Przeglądaj i filtruj wygenerowane pary pytanie-odpowiedź.")

# ---------------------------------------------------------------------------
# Stats header
# ---------------------------------------------------------------------------

try:
    stats = requests.get(f"{API_URL}/api/samples/stats", timeout=5).json()
except Exception:
    stats = {}

total = stats.get("total", 0)
if total == 0:
    st.info("Brak danych. Uruchom pipeline na stronie 🚀 AutoPilot.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Łącznie Q&A", total)
col2.metric("Avg. score", f"{stats.get('avg_quality_score', 0):.3f}")
col3.metric("Pary DPO", stats.get("dpo_pairs", 0))
col4.metric("Perspektywy", len(stats.get("perspectives", {})))

st.divider()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

f1, f2, f3, f4 = st.columns(4)

perspectives = ["(wszystkie)"] + list(stats.get("perspectives", {}).keys())
difficulties = ["(wszystkie)", "easy", "medium", "hard"]

with f1:
    sel_persp = st.selectbox("Perspektywa", perspectives)
with f2:
    sel_diff = st.selectbox("Trudność", difficulties)
with f3:
    min_score = st.slider("Min. score", 0.0, 1.0, 0.0, 0.01)
with f4:
    page_size = st.selectbox("Na stronie", [10, 25, 50], index=1)

# ---------------------------------------------------------------------------
# Sample table
# ---------------------------------------------------------------------------

params: dict = {"limit": page_size, "min_score": min_score if min_score > 0 else None}
if sel_persp != "(wszystkie)":
    params["perspective"] = sel_persp
if sel_diff != "(wszystkie)":
    params["difficulty"] = sel_diff

page_num = st.session_state.get("sample_page", 0)
params["offset"] = page_num * page_size

try:
    resp = requests.get(f"{API_URL}/api/samples", params={k: v for k, v in params.items() if v is not None}, timeout=10)
    data = resp.json() if resp.ok else {"samples": [], "total": 0}
except Exception:
    data = {"samples": [], "total": 0}

samples = data.get("samples", [])
data_total = data.get("total", 0)

# Pagination
col_prev, col_info, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("◀ Poprzednia") and page_num > 0:
        st.session_state["sample_page"] = page_num - 1
        st.rerun()
with col_info:
    start = page_num * page_size + 1
    end = min(start + page_size - 1, data_total)
    st.caption(f"Wyniki {start}–{end} z {data_total}")
with col_next:
    if st.button("Następna ▶") and end < data_total:
        st.session_state["sample_page"] = page_num + 1
        st.rerun()

# Display samples
for s in samples:
    score = s.get("quality_score") or 0
    if score >= 0.88:
        badge = "🟢"
    elif score >= 0.70:
        badge = "🟡"
    else:
        badge = "🔴"

    dpo_tag = "📎 DPO" if s.get("has_dpo") else ""
    turns_tag = f"🔄 {s.get('turn_count', 1)} tury" if s.get("turn_count", 1) > 1 else ""
    header = (
        f"{badge} [{s.get('perspective', '?')}] [{s.get('difficulty', '?')}] "
        f"score={score:.2f} {dpo_tag} {turns_tag}"
    )

    with st.expander(f"{header}  —  {s['question'][:100]}..."):
        st.markdown(f"**Pytanie:**\n{s['question']}")
        st.markdown(f"**Odpowiedź:**\n{s['answer']}")
        c1, c2, c3 = st.columns(3)
        c1.write(f"Type: {s.get('question_type', '?')}")
        c2.write(f"Batch: {s.get('batch_id', '?')}")
        c3.write(f"Adversarial: {s.get('is_adversarial', False)}")
        st.caption(f"ID: {s['id']} | Created: {s.get('created_at', '?')}")
