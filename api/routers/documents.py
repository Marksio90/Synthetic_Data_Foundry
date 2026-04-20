"""
api/routers/documents.py — Document upload, listing, deletion.

Endpoints:
  POST   /api/documents/upload      Upload one or more PDFs to data/
  GET    /api/documents             List all PDFs in data/ with DB stats
  DELETE /api/documents/{filename}  Remove a PDF and optionally its DB records
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.db import get_session
from api.schemas import DocumentInfo, DocumentListResponse
from api.security import require_admin_api_key
from config.settings import settings
from db.models import DirectiveChunk, GeneratedSample, SourceDocument

router = APIRouter(dependencies=[Depends(require_admin_api_key)])

DATA_DIR = Path(settings.data_dir)
_MAX_UPLOAD_BYTES = settings.max_upload_bytes


def _safe_data_path(name: str) -> Path:
    if not name or Path(name).name != name:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {name}")
    base = DATA_DIR.resolve()
    candidate = (DATA_DIR / name).resolve()
    if base not in candidate.parents:
        raise HTTPException(status_code=400, detail=f"Invalid path: {name}")
    return candidate


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _doc_info(filename: str, session: Session) -> DocumentInfo:
    path = _safe_data_path(filename)
    size = path.stat().st_size if path.exists() else 0
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else ""

    # Look up DB stats
    doc = session.scalar(
        select(SourceDocument).where(SourceDocument.filename == filename)
    )
    chunk_count = 0
    sample_count = 0
    if doc:
        chunk_count = session.scalar(
            select(func.count(DirectiveChunk.id)).where(
                DirectiveChunk.source_doc_id == doc.id
            )
        ) or 0
        sample_count = session.scalar(
            select(func.count(GeneratedSample.id))
            .join(DirectiveChunk, GeneratedSample.chunk_id == DirectiveChunk.id)
            .where(DirectiveChunk.source_doc_id == doc.id)
        ) or 0

    return DocumentInfo(
        filename=filename,
        size_bytes=size,
        uploaded_at=mtime,
        in_db=doc is not None,
        chunk_count=chunk_count,
        sample_count=sample_count,
    )


@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
) -> dict:
    """Upload one or more PDF files to the data/ directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for file in files:
        if not file.filename:
            continue
        safe_name = Path(file.filename).name
        dest = _safe_data_path(safe_name)

        total = 0
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large: {safe_name}. "
                            f"Limit is {_MAX_UPLOAD_BYTES} bytes."
                        ),
                    )
                out.write(chunk)

        uploaded.append(
            DocumentInfo(
                filename=dest.name,
                size_bytes=total,
                uploaded_at=datetime.now().isoformat(),
                in_db=False,
                chunk_count=0,
                sample_count=0,
            )
        )
    return {"uploaded": [d.model_dump() for d in uploaded], "count": len(uploaded)}


_SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".html", ".txt", ".md", ".mp3", ".wav", ".m4a", ".mp4"}


@router.get("", response_model=DocumentListResponse)
def list_documents(session: Session = Depends(get_session)) -> DocumentListResponse:
    """List all supported files in the data/ directory with chunk/sample counts from DB."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    docs = [_doc_info(p.name, session) for p in files]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{filename}")
def delete_document(
    filename: str,
    remove_db: bool = False,
    session: Session = Depends(get_session),
) -> dict:
    """
    Delete a PDF from data/.
    If remove_db=true, also removes the SourceDocument and all derived chunks/samples.
    """
    path = _safe_data_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    path.unlink()

    db_removed = False
    if remove_db:
        doc = session.scalar(
            select(SourceDocument).where(SourceDocument.filename == filename)
        )
        if doc:
            session.delete(doc)
            session.commit()
            db_removed = True

    return {
        "filename": filename,
        "file_deleted": True,
        "db_records_removed": db_removed,
    }
