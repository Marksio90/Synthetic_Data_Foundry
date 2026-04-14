"""
ui/pages/1_Dokumenty.py — Upload i zarządzanie dokumentami PDF.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = st.session_state.get("api_url", os.getenv("API_URL", "http://localhost:8080"))

st.set_page_config(page_title="Dokumenty — Foundry Studio", page_icon="📄", layout="wide")
st.title("📄 Dokumenty")
st.caption("Wgraj dokumenty PDF, które posłużą jako źródło danych do pipeline'u.")

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------

st.subheader("Wgraj nowe dokumenty")
uploaded_files = st.file_uploader(
    "Przeciągnij pliki PDF tutaj lub kliknij Browse",
    type=["pdf"],
    accept_multiple_files=True,
    help="Obsługiwane formaty: PDF. Maksymalnie 10 plików na raz.",
)

if uploaded_files:
    if st.button("⬆️ Prześlij do platformy", type="primary"):
        files_payload = [
            ("files", (f.name, f.getvalue(), "application/pdf"))
            for f in uploaded_files
        ]
        with st.spinner("Przesyłanie..."):
            try:
                resp = requests.post(f"{API_URL}/api/documents/upload", files=files_payload, timeout=60)
                if resp.ok:
                    data = resp.json()
                    st.success(f"✅ Przesłano {data['count']} plik(ów).")
                    st.rerun()
                else:
                    st.error(f"Błąd: {resp.text}")
            except Exception as e:
                st.error(f"Połączenie z API nie powiodło się: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Document list
# ---------------------------------------------------------------------------

st.subheader("Wgrane dokumenty")

try:
    resp = requests.get(f"{API_URL}/api/documents", timeout=10)
    if not resp.ok:
        st.error(f"Nie można pobrać listy dokumentów: {resp.status_code}")
        st.stop()

    data = resp.json()
    docs = data.get("documents", [])

    if not docs:
        st.info("Brak dokumentów. Wgraj pierwsze PDFy powyżej.")
        st.stop()

    # Display as table with action buttons
    for doc in docs:
        col_name, col_size, col_chunks, col_samples, col_status, col_del = st.columns(
            [3, 1, 1, 1, 1, 1]
        )
        with col_name:
            st.write(f"📄 **{doc['filename']}**")
        with col_size:
            size_mb = doc["size_bytes"] / 1_048_576
            st.write(f"{size_mb:.1f} MB")
        with col_chunks:
            st.write(f"{doc['chunk_count']} chunki")
        with col_samples:
            st.write(f"{doc['sample_count']} Q&A")
        with col_status:
            if doc["in_db"]:
                st.write("✅ w bazie")
            else:
                st.write("⏳ nowy")
        with col_del:
            if st.button("🗑️", key=f"del_{doc['filename']}", help="Usuń plik"):
                try:
                    r = requests.delete(
                        f"{API_URL}/api/documents/{doc['filename']}",
                        timeout=10,
                    )
                    if r.ok:
                        st.success(f"Usunięto: {doc['filename']}")
                        st.rerun()
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(str(e))

    st.caption(f"Łącznie: {data['total']} dokumentów")

except requests.exceptions.ConnectionError:
    st.error("Nie można połączyć się z API. Sprawdź czy serwis foundry-api jest uruchomiony.")
