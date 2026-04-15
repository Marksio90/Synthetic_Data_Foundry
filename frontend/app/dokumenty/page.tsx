'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { FileText, Upload, Trash2, RefreshCw, Database, File } from 'lucide-react';
import type { Document } from '@/lib/api';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

function fmt(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function DokumentyPage() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [dragging, setDragging] = useState(false);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/documents`);
      const data = await res.json();
      setDocs(data.documents ?? []);
    } catch {
      setError('Nie można pobrać listy dokumentów.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const notify = (msg: string, type: 'ok' | 'err') => {
    if (type === 'ok') { setSuccess(msg); setTimeout(() => setSuccess(''), 4000); }
    else { setError(msg); setTimeout(() => setError(''), 5000); }
  };

  const upload = async (files: FileList | File[]) => {
    const pdfs = Array.from(files).filter(f => f.name.endsWith('.pdf'));
    if (!pdfs.length) { notify('Wybierz pliki PDF.', 'err'); return; }
    setUploading(true);
    try {
      const fd = new FormData();
      pdfs.forEach(f => fd.append('files', f));
      const res = await fetch(`${API}/api/documents/upload`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Błąd uploadu');
      const data = await res.json();
      notify(`Przesłano ${data.count} plik(ów).`, 'ok');
      await load();
    } catch (e: unknown) {
      notify(e instanceof Error ? e.message : 'Błąd uploadu', 'err');
    } finally {
      setUploading(false);
    }
  };

  const deleteDoc = async (filename: string) => {
    if (!confirm(`Usunąć "${filename}" z dysku i bazy?`)) return;
    setDeletingFile(filename);
    try {
      const res = await fetch(`${API}/api/documents/${encodeURIComponent(filename)}?remove_db=true`, { method: 'DELETE' });
      if (!res.ok) throw new Error((await res.json()).detail ?? 'Błąd usuwania');
      notify(`Usunięto: ${filename}`, 'ok');
      await load();
    } catch (e: unknown) {
      notify(e instanceof Error ? e.message : 'Błąd usuwania', 'err');
    } finally {
      setDeletingFile(null);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    upload(e.dataTransfer.files);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <FileText className="w-6 h-6 text-accent" /> Dokumenty
        </h1>
        <button onClick={load} className="btn-secondary flex items-center gap-1.5 text-sm">
          <RefreshCw className="w-4 h-4" /> Odśwież
        </button>
      </div>

      {error && <div className="alert-error">{error}</div>}
      {success && <div className="alert-success">{success}</div>}

      {/* Upload zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors
          ${dragging ? 'border-accent bg-accent/10' : 'border-border hover:border-accent/60 hover:bg-bg-surface2'}
        `}
      >
        <Upload className="w-8 h-8 mx-auto mb-2 text-text-muted" />
        <p className="text-text font-medium">{uploading ? 'Przesyłanie...' : 'Przeciągnij pliki PDF lub kliknij'}</p>
        <p className="text-text-muted text-sm mt-1">Obsługuje wiele plików jednocześnie</p>
        <input ref={inputRef} type="file" accept=".pdf" multiple className="hidden"
          onChange={e => e.target.files && upload(e.target.files)} />
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-text-muted text-xs uppercase tracking-wider">
              <th className="text-left px-4 py-3">Plik</th>
              <th className="text-right px-4 py-3">Rozmiar</th>
              <th className="text-right px-4 py-3">Chunki</th>
              <th className="text-right px-4 py-3">Q&A</th>
              <th className="text-center px-4 py-3">Status</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 3 }).map((_, i) => (
                <tr key={i} className="border-b border-border">
                  {Array.from({ length: 6 }).map((_, j) => (
                    <td key={j} className="px-4 py-3">
                      <div className="h-4 bg-bg-surface2 rounded animate-pulse" />
                    </td>
                  ))}
                </tr>
              ))
            ) : docs.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-12 text-center text-text-muted">
                  <File className="w-10 h-10 mx-auto mb-2 opacity-30" />
                  Brak dokumentów. Wgraj pliki PDF powyżej.
                </td>
              </tr>
            ) : docs.map(doc => (
              <tr key={doc.filename} className="border-b border-border hover:bg-bg-surface2 transition-colors">
                <td className="px-4 py-3 font-medium text-text flex items-center gap-2">
                  <FileText className="w-4 h-4 text-accent flex-shrink-0" />
                  {doc.filename}
                </td>
                <td className="px-4 py-3 text-right text-text-muted">{fmt(doc.size_bytes)}</td>
                <td className="px-4 py-3 text-right">{doc.chunk_count}</td>
                <td className="px-4 py-3 text-right">{doc.sample_count}</td>
                <td className="px-4 py-3 text-center">
                  {doc.in_db
                    ? <span className="badge badge-success"><Database className="w-3 h-3" /> W bazie</span>
                    : <span className="badge badge-muted">Tylko plik</span>}
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={() => deleteDoc(doc.filename)}
                    disabled={deletingFile === doc.filename}
                    className="text-text-muted hover:text-error transition-colors p-1 rounded"
                    title="Usuń dokument"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
