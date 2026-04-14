"""
ui/pages/6_Chatbot.py — Chatbot Studio.

Tabs:
  1. Testuj chatbota  — interactive chat with deployed model
  2. Ewaluacja        — automated evaluation run + metrics dashboard
"""

from __future__ import annotations

import os
import time
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Chatbot Studio", page_icon="💬", layout="wide")
st.title("💬 Chatbot Studio")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(path: str, **params) -> dict | list | None:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _post(path: str, json: dict | None = None, params: dict | None = None) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{path}", json=json, params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Timeout — model nie odpowiedział w ciągu 120s. Sprawdź czy Ollama jest uruchomiony.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_model" not in st.session_state:
    st.session_state.chat_model = ""
if "eval_run_id" not in st.session_state:
    st.session_state.eval_run_id = None

# ---------------------------------------------------------------------------
# Sidebar — Ollama connection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Połączenie Ollama")
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")

    if st.button("Odśwież modele"):
        models_data = _get("/api/chatbot/models", ollama_url=ollama_url)
        if models_data:
            st.session_state["available_models"] = models_data.get("models", [])
            st.success(f"Znaleziono {models_data['count']} modeli")

    models = st.session_state.get("available_models", [])
    if models:
        model_names = [m["name"] for m in models]
        selected_model = st.selectbox("Wybierz model", model_names)
        st.session_state.chat_model = selected_model

        # Show model size
        for m in models:
            if m["name"] == selected_model:
                st.caption(f"Rozmiar: {m['size_gb']:.2f} GB")
                break
    else:
        st.info("Kliknij **Odśwież modele** aby załadować listę.")
        selected_model = st.text_input("Lub wpisz nazwę modelu ręcznie", "foundry-domain-model")
        st.session_state.chat_model = selected_model

    st.divider()
    st.caption(
        "Ollama musi być uruchomiony lokalnie lub w kontenerze.\n\n"
        "Uruchomienie przez Docker:\n"
        "`docker compose --profile chatbot up -d`"
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_eval = st.tabs(["💬 Testuj chatbota", "📊 Ewaluacja"])

# ===========================================================================
# TAB 1 — Chat
# ===========================================================================

with tab_chat:
    active_model = st.session_state.chat_model

    if not active_model:
        st.warning("Wybierz model w panelu bocznym.")
        st.stop()

    st.caption(f"Model: **{active_model}** | Ollama: `{ollama_url}`")

    # System prompt
    with st.expander("⚙️ System prompt (opcjonalnie)", expanded=False):
        system_prompt = st.text_area(
            "System prompt",
            value=(
                "Jesteś ekspertem ds. ESG i prawa korporacyjnego UE. "
                "Odpowiadasz wyłącznie na podstawie dokumentów, na których zostałeś wytrenowany. "
                "Jeśli pytanie wykracza poza Twoją wiedzę, odpowiedz: "
                "\"Brak danych w zbiorze wiedzy.\""
            ),
            height=80,
        )
        temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
        max_tokens = st.number_input("Max tokens", 64, 4096, 512, 64)

    # Chat container
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Clear chat button
    col_inp, col_clear = st.columns([6, 1])
    with col_clear:
        if st.button("Wyczyść", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Input
    user_input = st.chat_input("Napisz pytanie...", key="chat_input")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Build message list for API
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(st.session_state.chat_history)

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Myślę..."):
                    resp = _post(
                        "/api/chatbot/chat",
                        json={
                            "model": active_model,
                            "messages": api_messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "ollama_url": ollama_url,
                        },
                    )
                if resp:
                    answer = resp.get("content", "")
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    tokens = resp.get("usage", {})
                    if tokens:
                        st.caption(
                            f"Tokeny: {tokens.get('prompt_tokens', 0)} prompt + "
                            f"{tokens.get('completion_tokens', 0)} odpowiedź"
                        )
                else:
                    st.error("Brak odpowiedzi od modelu.")

        st.rerun()


# ===========================================================================
# TAB 2 — Evaluation
# ===========================================================================

with tab_eval:
    st.subheader("📊 Automatyczna ewaluacja modelu")
    st.markdown(
        "Ewaluacja pobiera próbki z datasetu JSONL, odpytuje model przez Ollama "
        "i ocenia odpowiedzi przez judge (GPT-4o-mini). Wynik: avg_score, pass_rate, refusal_pct."
    )

    with st.form("eval_form"):
        col_ev1, col_ev2 = st.columns(2)
        with col_ev1:
            eval_model = st.text_input(
                "Nazwa modelu Ollama",
                value=st.session_state.chat_model or "foundry-domain-model",
            )
            eval_jsonl = st.text_input("Ścieżka JSONL (auto)", "")
        with col_ev2:
            eval_samples = st.number_input("Liczba próbek", 5, 500, 50, 5)
            eval_ollama = st.text_input("Ollama URL", ollama_url)
            eval_seed = st.number_input("Seed", 0, 9999, 42, 1)

        eval_submitted = st.form_submit_button("▶️ URUCHOM EWALUACJĘ", use_container_width=True)

    if eval_submitted:
        payload = {
            "model_name": eval_model,
            "n_samples": eval_samples,
            "ollama_url": eval_ollama,
            "seed": eval_seed,
        }
        if eval_jsonl:
            payload["jsonl_path"] = eval_jsonl

        with st.spinner("Uruchamiam ewaluację..."):
            resp = _post("/api/chatbot/eval/run", json=payload)
        if resp:
            st.session_state.eval_run_id = resp["run_id"]
            st.success(f"Ewaluacja uruchomiona: `{resp['run_id']}`")
            st.rerun()

    # -------- Live eval status --------
    eval_run_id = st.session_state.eval_run_id
    if eval_run_id:
        status_data = _get(f"/api/chatbot/eval/status/{eval_run_id}")
        if status_data:
            status = status_data.get("status", "unknown")
            elapsed = status_data.get("elapsed_seconds", 0)

            col_e1, col_e2 = st.columns(2)
            col_e1.metric("Status", status.upper())
            col_e2.metric("Czas", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")

            if status_data.get("error"):
                st.error(f"Błąd: {status_data['error']}")

            metrics = status_data.get("metrics") or {}
            if metrics and "avg_score" in metrics:
                st.divider()
                st.subheader("Wyniki ewaluacji")

                # Key metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("avg_score", f"{metrics.get('avg_score', 0):.3f}")
                m2.metric("pass ≥0.88", f"{metrics.get('pass_rate_088', 0):.1f}%")
                m3.metric("fail <0.70", f"{metrics.get('fail_rate_070', 0):.1f}%")
                m4.metric("refusal %", f"{metrics.get('refusal_pct', 0):.1f}%")

                m5, m6, m7 = st.columns(3)
                m5.metric("median_score", f"{metrics.get('median_score', 0):.3f}")
                m6.metric("p25", f"{metrics.get('p25_score', 0):.3f}")
                m7.metric("p75", f"{metrics.get('p75_score', 0):.3f}")

                st.metric(
                    "Śr. długość odpowiedzi",
                    f"{metrics.get('avg_response_len', 0):.0f} znaków",
                )
                st.caption(f"Oceniono {metrics.get('n_evaluated', 0)} próbek.")

                # Low-score examples
                examples = metrics.get("low_score_examples", [])
                if examples:
                    st.markdown("**Przykłady niskich/wysokich ocen:**")
                    for ex in examples[:5]:
                        score = ex.get("score", 0)
                        color = "🟢" if score >= 0.88 else ("🟡" if score >= 0.70 else "🔴")
                        with st.expander(
                            f"{color} Score: {score:.3f} | Q: {ex.get('question', '')[:80]}..."
                        ):
                            st.markdown(f"**Pytanie:** {ex.get('question', '')}")
                            st.markdown(f"**Odpowiedź modelu:** {ex.get('model_answer', '')}")
                            st.markdown(f"**Odpowiedź referencyjna:** {ex.get('ref_answer', '')}")

            # Log
            log_data = _get(f"/api/chatbot/eval/log/{eval_run_id}", offset=0, limit=200)
            if log_data:
                lines = log_data.get("lines", [])
                if lines:
                    with st.expander("Log ewaluacji", expanded=(status == "running")):
                        st.code("\n".join(lines[-100:]), language=None)

            if status == "running":
                time.sleep(2)
                st.rerun()
            elif status in ("done", "error"):
                if st.button("🔄 Nowa ewaluacja", key="new_eval"):
                    st.session_state.eval_run_id = None
                    st.rerun()

    # -------- Previous eval runs --------
    st.divider()
    st.subheader("Historia ewaluacji")
    eval_runs = _get("/api/chatbot/eval/runs")
    if eval_runs:
        for r in eval_runs:
            status_icon = {"done": "✅", "error": "❌", "running": "⏳"}.get(r["status"], "⏸️")
            elapsed = r.get("elapsed_seconds", 0)
            metrics = r.get("metrics") or {}
            avg = metrics.get("avg_score")
            avg_str = f" | avg_score: {avg:.3f}" if avg is not None else ""
            st.markdown(
                f"- {status_icon} `{r['run_id']}` — {int(elapsed // 60)}m{avg_str}"
            )
    else:
        st.info("Brak wyników ewaluacji.")
