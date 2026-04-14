"""
ui/pages/5_Trening.py — Training pipeline UI.

Tabs:
  1. Hardware & Gate  — inspect GPU, run quality gate
  2. Uruchom trening  — one-button training start + live log
  3. Eksport          — merge LoRA → GGUF → client ZIP
"""

from __future__ import annotations

import time
import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Trening modelu", page_icon="🧠", layout="wide")
st.title("🧠 Trening modelu")

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
        r = requests.post(f"{API_URL}{path}", json=json, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _score_badge(passed: bool) -> str:
    return "✅" if passed else "❌"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "train_run_id" not in st.session_state:
    st.session_state.train_run_id = None
if "export_run_id" not in st.session_state:
    st.session_state.export_run_id = None
if "hw_data" not in st.session_state:
    st.session_state.hw_data = None
if "gate_data" not in st.session_state:
    st.session_state.gate_data = None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_hw, tab_train, tab_export = st.tabs([
    "🖥️ Sprzęt & Jakość",
    "🚀 Uruchom trening",
    "📦 Eksport modelu",
])

# ===========================================================================
# TAB 1 — Hardware & Quality Gate
# ===========================================================================

with tab_hw:
    col_hw, col_gate = st.columns(2)

    # -------- Hardware Inspector --------
    with col_hw:
        st.subheader("🖥️ Profil sprzętowy")
        if st.button("Sprawdź GPU", key="hw_inspect"):
            with st.spinner("Wykrywam GPU..."):
                hw = _get("/api/training/hardware")
                if hw:
                    st.session_state.hw_data = hw

        hw = st.session_state.hw_data
        if hw:
            gpu = hw.get("gpu_name", "CPU / brak GPU")
            vram = hw.get("vram_gb", 0)
            st.metric("GPU", gpu)
            st.metric("VRAM", f"{vram:.1f} GB" if vram else "brak")

            model = hw.get("recommended_model", {})
            if model:
                st.success(f"**Zalecany model:** {model.get('name', '—')}")
                c1, c2, c3 = st.columns(3)
                c1.metric("LoRA rank", model.get("lora_rank", "—"))
                c2.metric("Batch size", model.get("batch_size", "—"))
                c3.metric("Max seq len", model.get("max_seq_length", "—"))
                st.caption(
                    f"Szac. czas: {model.get('estimated_hours_per_1k', 0):.2f} h / 1k próbek"
                )
        else:
            st.info("Kliknij **Sprawdź GPU** aby wykryć konfigurację sprzętu.")

    # -------- Quality Gate --------
    with col_gate:
        st.subheader("🔍 Quality Gate")

        with st.expander("Ścieżki plików (opcjonalne)"):
            jsonl_path_gate = st.text_input("Ścieżka JSONL", "", key="gate_jsonl")
            dpo_path_gate = st.text_input("Ścieżka DPO JSONL", "", key="gate_dpo")

        if st.button("Uruchom Quality Gate", key="run_gate"):
            with st.spinner("Sprawdzam dataset..."):
                params: dict = {}
                if jsonl_path_gate:
                    params["jsonl_path"] = jsonl_path_gate
                if dpo_path_gate:
                    params["dpo_path"] = dpo_path_gate
                gate = _post("/api/training/gate", params=params)
                if gate:
                    st.session_state.gate_data = gate

        gate = st.session_state.gate_data
        if gate:
            passed = gate.get("passed", False)
            if passed:
                st.success("✅ Dataset przeszedł Quality Gate — gotowy do treningu!")
            else:
                st.error("❌ Dataset nie spełnia wymagań — popraw zaznaczone problemy.")

            checks = gate.get("checks", [])
            if checks:
                st.markdown("**Wyniki sprawdzeń:**")
                for c in checks:
                    icon = _score_badge(c["passed"])
                    msg = c.get("message") or f"Wartość: {c['value']} (próg: {c['threshold']})"
                    st.markdown(f"- {icon} **{c['name']}** — {msg}")

            warnings = gate.get("warnings", [])
            if warnings:
                st.markdown("**Ostrzeżenia:**")
                for w in warnings:
                    st.warning(w)
        else:
            st.info("Kliknij **Uruchom Quality Gate** aby sprawdzić gotowość datasetu.")


# ===========================================================================
# TAB 2 — Training
# ===========================================================================

with tab_train:
    st.subheader("🚀 Uruchom trening SFT → DPO")

    # Config overrides (collapsed by default)
    with st.expander("⚙️ Parametry zaawansowane (opcjonalne)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            run_name = st.text_input("Nazwa runu", "foundry-model")
            jsonl_path_train = st.text_input("Ścieżka JSONL (auto)", "")
            dpo_path_train = st.text_input("Ścieżka DPO JSONL (auto)", "")
            base_model_override = st.text_input("Model bazowy (auto)", "")
        with col_b:
            lora_rank_override = st.number_input(
                "LoRA rank (auto)", min_value=4, max_value=128, value=0, step=4
            )
            epochs_override = st.number_input(
                "Epoki (auto)", min_value=0, max_value=10, value=0, step=1
            )
            skip_dpo = st.checkbox("Pomiń DPO alignment", value=False)
            skip_eval = st.checkbox("Pomiń ewaluację", value=False)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_btn = st.button(
            "🚀 START TRENING",
            disabled=st.session_state.train_run_id is not None
            and (st.session_state.get("train_status") not in ("done", "error", None)),
            use_container_width=True,
        )

    if start_btn:
        payload: dict = {
            "run_name": run_name,
            "skip_dpo": skip_dpo,
            "skip_eval": skip_eval,
        }
        if jsonl_path_train:
            payload["jsonl_path"] = jsonl_path_train
        if dpo_path_train:
            payload["dpo_path"] = dpo_path_train
        if base_model_override:
            payload["base_model"] = base_model_override
        if lora_rank_override > 0:
            payload["lora_rank"] = lora_rank_override
        if epochs_override > 0:
            payload["epochs"] = epochs_override

        with st.spinner("Uruchamiam trening..."):
            resp = _post("/api/training/run", json=payload)
        if resp:
            st.session_state.train_run_id = resp["run_id"]
            st.session_state.train_status = "starting"
            st.success(
                f"Run **{resp['run_id']}** uruchomiony — model: {resp.get('base_model','?')}, "
                f"próbki: {resp.get('n_samples', 0)}, "
                f"epoki: {resp.get('epochs_sft', '?')}, "
                f"szac. czas: {resp.get('estimated_hours', '?')} h"
            )
            st.rerun()

    # -------- Live training status --------
    run_id = st.session_state.train_run_id
    if run_id:
        status_data = _get(f"/api/training/status/{run_id}")
        if status_data:
            status = status_data.get("status", "unknown")
            st.session_state.train_status = status
            elapsed = status_data.get("elapsed_seconds", 0)

            col_s1, col_s2 = st.columns(2)
            col_s1.metric("Status", status.upper())
            col_s2.metric("Czas", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")

            cfg = status_data.get("config") or {}
            if cfg:
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", cfg.get("base_model", "—"))
                c2.metric("Próbki", cfg.get("n_samples", "—"))
                c3.metric("GPU", cfg.get("gpu", "CPU"))

            if status_data.get("error"):
                st.error(f"Błąd: {status_data['error']}")

            # Log lines
            st.markdown("**Log treningu:**")
            log_data = _get(f"/api/training/log/{run_id}", offset=0, limit=500)
            if log_data:
                lines = log_data.get("lines", [])
                total = log_data.get("total_lines", 0)
                log_text = "\n".join(lines[-200:])  # show last 200 lines
                st.code(log_text or "(brak logów)", language=None)
                if total > 200:
                    st.caption(f"Pokazuję ostatnie 200 z {total} linii.")

            if status in ("running", "starting"):
                time.sleep(0.1)
                st.rerun()
            elif status == "done":
                st.success("✅ Trening zakończony! Przejdź do zakładki **Eksport** aby zbudować paczkę.")
                if st.button("🔄 Nowy run", key="new_run"):
                    st.session_state.train_run_id = None
                    st.session_state.train_status = None
                    st.rerun()
            elif status == "error":
                st.error("❌ Trening zakończony błędem.")
                if st.button("🔄 Spróbuj ponownie", key="retry_run"):
                    st.session_state.train_run_id = None
                    st.session_state.train_status = None
                    st.rerun()

    # -------- Previous training runs --------
    st.divider()
    st.subheader("Historia runów treningowych")
    runs_data = _get("/api/training/runs")
    if runs_data:
        for r in runs_data:
            status_icon = {"done": "✅", "error": "❌", "running": "⏳"}.get(r["status"], "⏸️")
            elapsed = r.get("elapsed_seconds", 0)
            time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            st.markdown(
                f"- {status_icon} `{r['run_id']}` — batch: {r.get('batch_id','?')} — {time_str}"
            )
    else:
        st.info("Brak runów treningowych.")


# ===========================================================================
# TAB 3 — Export
# ===========================================================================

with tab_export:
    st.subheader("📦 Eksport modelu → paczka dla klienta")

    st.markdown(
        "Eksport łączy adaptery LoRA z modelem bazowym, konwertuje do GGUF "
        "i pakuje wszystko do ZIP gotowego do wdrożenia przez klienta."
    )

    with st.form("export_form"):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            model_path = st.text_input(
                "Ścieżka do modelu LoRA",
                "/app/output/models/sft",
                help="Katalog z adapterami LoRA (wynik treningu SFT lub DPO)",
            )
            model_name = st.text_input("Nazwa modelu", "foundry-domain-model")
            domain_label = st.text_input("Etykieta domeny", "ESG / Prawo korporacyjne UE")
        with col_e2:
            base_model_export = st.text_input(
                "Model bazowy", "meta-llama/Llama-3.2-3B-Instruct"
            )
            quantization = st.selectbox(
                "Kwantyzacja GGUF",
                ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                index=0,
            )

        export_submitted = st.form_submit_button("📦 EKSPORTUJ MODEL", use_container_width=True)

    if export_submitted:
        payload = {
            "model_path": model_path,
            "base_model": base_model_export,
            "model_name": model_name,
            "domain_label": domain_label,
            "quantization": quantization,
        }
        with st.spinner("Uruchamiam eksport..."):
            resp = _post("/api/training/export", json=payload)
        if resp:
            st.session_state.export_run_id = resp.get("export_run_id")
            st.success(f"Eksport uruchomiony: `{st.session_state.export_run_id}`")
            st.rerun()

    # -------- Export progress --------
    export_run_id = st.session_state.export_run_id
    if export_run_id:
        status_data = _get(f"/api/training/status/{export_run_id}")
        if status_data:
            status = status_data.get("status", "unknown")
            elapsed = status_data.get("elapsed_seconds", 0)

            col_x1, col_x2 = st.columns(2)
            col_x1.metric("Status eksportu", status.upper())
            col_x2.metric("Czas", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")

            if status_data.get("error"):
                st.error(f"Błąd eksportu: {status_data['error']}")

            cfg = status_data.get("config") or {}
            if cfg.get("zip_path"):
                zip_name = cfg["zip_path"].split("/")[-1]
                st.success(f"✅ Paczka gotowa: `{zip_name}`")
                st.info(
                    "Zawartość paczki:\n"
                    "- `docker-compose.yml` — uruchomienie jedną komendą\n"
                    "- `Modelfile` — konfiguracja Ollama\n"
                    "- `*.gguf` — model w formacie GGUF\n"
                    "- `datacard.json` — statystyki datasetu\n"
                    "- `README.md` — instrukcja dla klienta"
                )

                # Download button via streaming API endpoint
                import requests as _req
                try:
                    dl_resp = _req.get(
                        f"{API_URL}/api/training/export/download/{export_run_id}",
                        timeout=30,
                        stream=True,
                    )
                    if dl_resp.ok:
                        st.download_button(
                            label=f"⬇️ Pobierz {zip_name}",
                            data=dl_resp.content,
                            file_name=zip_name,
                            mime="application/zip",
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Nie można pobrać pliku (kod {dl_resp.status_code}).")
                except Exception as e:
                    st.warning(f"Pobieranie niedostępne: {e}")

            # Log
            log_data = _get(f"/api/training/log/{export_run_id}", offset=0, limit=100)
            if log_data:
                lines = log_data.get("lines", [])
                if lines:
                    st.code("\n".join(lines[-50:]), language=None)

            if status in ("running", "starting"):
                time.sleep(0.5)
                st.rerun()
            elif status in ("done", "error"):
                if st.button("🔄 Nowy eksport", key="new_export"):
                    st.session_state.export_run_id = None
                    st.rerun()
