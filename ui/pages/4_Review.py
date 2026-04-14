"""
ui/pages/4_Review.py — Human Review Interface (strefa szara).

Pokazuje TYLKO próbki w strefie szarej (score 0.70–0.88, nieprzetworzone
przez człowieka). Automatycznie uruchamia AutoReviewer przed wyświetleniem.

Operator może:
  ✓ Zatwierdzić próbkę (human_flag = 'human_approved')
  ✗ Odrzucić próbkę   (human_flag = 'human_rejected')
  ✏ Edytować odpowiedź i zatwierdzić

Strefa zielona (≥0.88) i czerwona (<0.70) są obsługiwane automatycznie
— NIE pojawiają się na tej stronie.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = st.session_state.get("api_url", os.getenv("API_URL", "http://localhost:8080"))

APPROVE_THRESHOLD = 0.88
REVIEW_THRESHOLD = 0.70

st.set_page_config(page_title="Review — Foundry Studio", page_icon="🔶", layout="wide")
st.title("🔶 Human Review")
st.caption(
    f"Strefa szara: próbki z score {REVIEW_THRESHOLD}–{APPROVE_THRESHOLD}. "
    "Poniżej i powyżej — obsługa automatyczna."
)

# ---------------------------------------------------------------------------
# Run AutoReviewer (idempotent — przetwarza tylko nieocenione próbki)
# ---------------------------------------------------------------------------

col_run, col_stats = st.columns([2, 3])

with col_run:
    if st.button("🤖 Uruchom AutoReviewer", help="Przetworzy automatycznie zielone i czerwone strefy"):
        with st.spinner("AutoReviewer w toku..."):
            try:
                r = requests.post(
                    f"{API_URL}/api/samples/auto-review",
                    params={
                        "approve_threshold": APPROVE_THRESHOLD,
                        "review_threshold": REVIEW_THRESHOLD,
                    },
                    timeout=30,
                )
                if r.ok:
                    d = r.json()
                    st.success(
                        f"✅ Przetworzone: {d['total']} | "
                        f"Auto-approved: {d['approved']} | "
                        f"Do przeglądu: {d['queued']} | "
                        f"Auto-rejected: {d['rejected']}"
                    )
                    st.rerun()
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

# ---------------------------------------------------------------------------
# Fetch grey-zone samples (not yet human reviewed)
# ---------------------------------------------------------------------------

try:
    resp = requests.get(
        f"{API_URL}/api/samples",
        params={
            "limit": 50,
            "min_score": REVIEW_THRESHOLD,
            "offset": 0,
        },
        timeout=10,
    )
    all_data = resp.json() if resp.ok else {"samples": [], "total": 0}
except Exception:
    all_data = {"samples": [], "total": 0}

# Filter to grey zone only (server-side filtering will be enhanced in future sprint)
grey_samples = [
    s for s in all_data.get("samples", [])
    if (s.get("quality_score") or 0) < APPROVE_THRESHOLD
]

with col_stats:
    st.metric("Próbki do przeglądu (strefa szara)", len(grey_samples))

st.divider()

if not grey_samples:
    st.success(
        "✅ Brak próbek do ręcznego przeglądu — "
        "AutoReviewer obsłużył wszystkie próbki automatycznie."
    )
    st.stop()

st.info(
    f"Poniżej {len(grey_samples)} próbek w strefie szarej. "
    "Możesz je przejrzeć, ale nie musisz — pipeline działa bez Twojej decyzji."
)

# ---------------------------------------------------------------------------
# Review interface — one sample at a time
# ---------------------------------------------------------------------------

for s in grey_samples:
    score = s.get("quality_score") or 0
    perspective = s.get("perspective", "?")
    difficulty = s.get("difficulty", "?")

    with st.expander(
        f"🟡 score={score:.2f} | {perspective} | {difficulty} | {s['question'][:80]}...",
        expanded=False,
    ):
        st.markdown(f"**Pytanie:**\n\n{s['question']}")
        st.divider()
        st.markdown(f"**Odpowiedź:**\n\n{s['answer']}")

        col_a, col_r, col_e, col_info = st.columns([1, 1, 2, 2])

        with col_a:
            if st.button("✓ Zatwierdź", key=f"approve_{s['id']}", type="primary"):
                try:
                    r = requests.patch(
                        f"{API_URL}/api/samples/{s['id']}/review",
                        params={"action": "approve"},
                        timeout=10,
                    )
                    if r.ok:
                        st.success("Zatwierdzone!")
                        st.rerun()
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(str(e))

        with col_r:
            if st.button("✗ Odrzuć", key=f"reject_{s['id']}", type="secondary"):
                try:
                    r = requests.patch(
                        f"{API_URL}/api/samples/{s['id']}/review",
                        params={"action": "reject"},
                        timeout=10,
                    )
                    if r.ok:
                        st.warning("Odrzucone.")
                        st.rerun()
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(str(e))

        with col_e:
            with st.popover("✏️ Edytuj odpowiedź"):
                edited = st.text_area(
                    "Poprawiona odpowiedź:",
                    value=s["answer"],
                    key=f"edit_{s['id']}",
                    height=150,
                )
                if st.button("Zapisz edycję", key=f"save_{s['id']}"):
                    try:
                        r = requests.patch(
                            f"{API_URL}/api/samples/{s['id']}/review",
                            params={"action": "edit", "edited_answer": edited},
                            timeout=10,
                        )
                        if r.ok:
                            st.success("Zapisano!")
                            st.rerun()
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(str(e))

        with col_info:
            st.caption(
                f"Type: {s.get('question_type', '?')} | "
                f"Adversarial: {s.get('is_adversarial', False)} | "
                f"DPO: {s.get('has_dpo', False)} | "
                f"ID: {s['id'][:8]}..."
            )
