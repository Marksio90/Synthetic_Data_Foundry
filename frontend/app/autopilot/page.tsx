'use client';

import { Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Loader2, RotateCcw, Square, Trash2, UploadCloud, Zap } from 'lucide-react';
import { apiFetch } from '@/lib/api';
import type { Document, PipelineRun } from '@/lib/api';
import LiveLog from '@/components/LiveLog';
import StatusBadge from '@/components/StatusBadge';
import ProgressBar from '@/components/ProgressBar';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

function ErrorBanner({ message, onClose }: { message: string; onClose: () => void }) {
  return (
    <div className="flex items-center justify-between alert-error">
      <span>{message}</span>
      <button onClick={onClose} className="ml-4 font-bold text-lg leading-none">×</button>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function AutopilotPageContent() {
  const searchParams = useSearchParams();
  const scoutPrefix = searchParams.get('scout_prefix') ?? '';

  const [docs, setDocs] = useState<Document[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [docsError, setDocsError] = useState('');

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [runError, setRunError] = useState('');
  const [cancelling, setCancelling] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Documents ──────────────────────────────────────────────────────────────

  const loadDocs = useCallback(async () => {
    setLoadingDocs(true);
    setDocsError('');
    try {
      const data = await apiFetch<{ documents?: Document[] } | Document[]>('/api/documents');
      const list: Document[] = Array.isArray((data as { documents?: Document[] }).documents)
        ? (data as { documents: Document[] }).documents
        : Array.isArray(data)
        ? (data as Document[])
        : [];
      setDocs(list);
      if (scoutPrefix) {
        setSelectedDocs(new Set(list.filter((d) => d.filename.startsWith(scoutPrefix)).map((d) => d.filename)));
      }
    } catch (e: unknown) {
      setDocs([]);
      setDocsError(e instanceof Error ? e.message : 'Nie udało się pobrać dokumentów');
    } finally {
      setLoadingDocs(false);
    }
  }, [scoutPrefix]);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  const toggleDoc = (filename: string) =>
    setSelectedDocs((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename); else next.add(filename);
      return next;
    });

  const toggleAll = () =>
    setSelectedDocs(selectedDocs.size === docs.length ? new Set() : new Set(docs.map((d) => d.filename)));

  // ── Upload ─────────────────────────────────────────────────────────────────

  const uploadFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (!arr.length) return;
    setUploading(true);
    setUploadError('');
    try {
      for (const file of arr) {
        const fd = new FormData();
        fd.append('files', file);
        const res = await fetch(`${API}/api/documents/upload`, { method: 'POST', body: fd });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail ?? `HTTP ${res.status}`);
        }
      }
      await loadDocs();
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Błąd uploadu');
    } finally {
      setUploading(false);
    }
  };

  const deleteDocument = async (filename: string) => {
    setDeletingDoc(filename);
    try {
      const res = await fetch(`${API}/api/documents/${encodeURIComponent(filename)}?remove_db=true`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setDocs((prev) => prev.filter((d) => d.filename !== filename));
      setSelectedDocs((prev) => {
        const next = new Set(prev);
        next.delete(filename);
        return next;
      });
    } catch {
      /* ignore — plik mógł już nie istnieć */
    } finally {
      setDeletingDoc(null);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) uploadFiles(e.target.files);
    e.target.value = '';
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
  };

  // ── Pipeline ───────────────────────────────────────────────────────────────

  const startPoll = (runId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/pipeline/status/${runId}`);
        if (res.ok) {
          const data: PipelineRun = await res.json();
          setRun(data);
          if (data.status === 'done' || data.status === 'error') {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setRunning(false);
          }
        }
      } catch { /* ignore */ }
    }, 2000);
  };

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const handleRun = async () => {
    if (!selectedDocs.size) { setRunError('Wybierz co najmniej jeden dokument.'); return; }
    setRunError('');
    setRunning(true);
    setRun(null);
    try {
      // AI calibrator handles quality_threshold, max_turns, adversarial_ratio automatically
      const data = await apiFetch<PipelineRun>('/api/pipeline/run', {
        method: 'POST',
        body: JSON.stringify({ filenames: Array.from(selectedDocs) }),
      });
      setRun(data);
      startPoll(data.run_id);
    } catch (e: unknown) {
      setRunError(e instanceof Error ? e.message : 'Błąd uruchamiania pipeline');
      setRunning(false);
    }
  };

  const handleCancel = async () => {
    if (!run) return;
    setCancelling(true);
    try {
      await fetch(`${API}/api/pipeline/cancel/${run.run_id}`, { method: 'POST' });
      if (pollRef.current) clearInterval(pollRef.current);
      setRunning(false);
      setRun(null);
    } catch { /* ignore */ } finally { setCancelling(false); }
  };

  const handleNewRun = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setRun(null);
    setRunning(false);
    setRunError('');
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-8 w-full">
      <div className="flex items-center gap-2">
        <Zap className="w-6 h-6 text-accent" />
        <h1 className="text-2xl font-bold text-text">AutoPilot</h1>
      </div>

      {docsError   && <ErrorBanner message={docsError}   onClose={() => setDocsError('')} />}
      {uploadError && <ErrorBanner message={uploadError} onClose={() => setUploadError('')} />}
      {runError    && <ErrorBanner message={runError}    onClose={() => setRunError('')} />}

      {/* ─── Section 1: Dokumenty ─── */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="section-title">
            <span className="w-6 h-6 rounded-full bg-accent text-bg-base text-xs flex items-center justify-center font-bold">1</span>
            Dokumenty do przetworzenia
          </h2>
          <span className="text-sm text-text-muted">{selectedDocs.size}/{docs.length} zaznaczonych</span>
        </div>

        {/* Drag-drop upload zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            flex flex-col items-center justify-center gap-2 py-5 rounded-lg border-2 border-dashed cursor-pointer
            transition-colors select-none
            ${dragOver
              ? 'border-accent bg-accent/5'
              : 'border-border hover:border-accent/50 hover:bg-bg-surface2'}
          `}
        >
          {uploading
            ? <Loader2 className="w-5 h-5 text-accent animate-spin" />
            : <UploadCloud className="w-5 h-5 text-text-muted" />}
          <p className="text-sm text-text-muted">
            {uploading ? 'Wgrywanie…' : 'Przeciągnij pliki PDF / DOCX / HTML lub kliknij'}
          </p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.doc,.html,.txt,.md,.mp3,.wav,.m4a,.mp4"
            onChange={onFileChange}
            className="hidden"
          />
        </div>

        {/* Document checklist */}
        <div className="card p-4 space-y-2">
          {loadingDocs ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => <div key={i} className="h-8 bg-bg-surface2 rounded animate-pulse" />)}
            </div>
          ) : docs.length === 0 ? (
            <p className="text-text-muted text-sm py-4 text-center">
              Brak dokumentów. Wgraj pliki powyżej lub użyj <strong>Gap Scout</strong> → Ingestuj.
            </p>
          ) : (
            <>
              <label className="flex items-center gap-2 cursor-pointer px-2 py-1.5 rounded hover:bg-bg-surface2 transition-colors">
                <input
                  type="checkbox"
                  checked={selectedDocs.size === docs.length && docs.length > 0}
                  onChange={toggleAll}
                  className="w-4 h-4"
                />
                <span className="text-sm font-medium text-text">Zaznacz wszystkie</span>
              </label>
              <div className="border-t border-border" />
              {docs.map((doc) => (
                <div key={doc.filename} className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-bg-surface2 transition-colors group">
                  <label className="flex items-center gap-2 flex-1 min-w-0 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedDocs.has(doc.filename)}
                      onChange={() => toggleDoc(doc.filename)}
                      className="w-4 h-4 shrink-0"
                    />
                    <span className="text-sm text-text flex-1 truncate">{doc.filename}</span>
                    <span className="text-xs text-text-muted shrink-0">{doc.chunk_count} ch · {doc.sample_count} Q&A</span>
                  </label>
                  <button
                    onClick={() => deleteDocument(doc.filename)}
                    disabled={deletingDoc === doc.filename}
                    title="Usuń dokument"
                    className="opacity-0 group-hover:opacity-100 p-1 rounded text-text-muted/40 hover:text-error transition-all disabled:opacity-50 shrink-0"
                  >
                    {deletingDoc === doc.filename
                      ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      : <Trash2 className="w-3.5 h-3.5" />}
                  </button>
                </div>
              ))}
            </>
          )}
        </div>
      </section>

      {/* ─── Section 2: Run ─── */}
      <section className="space-y-4">
        <h2 className="section-title">
          <span className="w-6 h-6 rounded-full bg-accent text-bg-base text-xs flex items-center justify-center font-bold">2</span>
          Uruchom pipeline
        </h2>

        <p className="text-sm text-text-muted">
          AI automatycznie analizuje dokumenty i dobiera próg jakości, liczbę tur dialogu oraz parametry kalibracji.
        </p>

        <div className="text-sm text-text-muted">
          Uruchomi: <strong className="text-text">{selectedDocs.size}</strong> dok. · limit <strong className="text-text">∞</strong> chunków
        </div>

        {!running && !run && (
          <button
            onClick={handleRun}
            disabled={selectedDocs.size === 0}
            className="
              w-full py-4 rounded-lg font-bold text-lg
              bg-accent text-bg-base
              hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-200
              flex items-center justify-center gap-3
            "
          >
            <Zap className="w-5 h-5" />
            FULL AUTO RUN
          </button>
        )}

        {(running || run) && (
          <div className="card p-5 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="font-semibold text-text">Status pipeline</span>
                {run && <StatusBadge status={run.status} />}
              </div>
              <div className="flex gap-2">
                {running && (
                  <button onClick={handleCancel} disabled={cancelling} className="btn-danger flex items-center gap-1.5 text-sm">
                    <Square className="w-3.5 h-3.5" />
                    {cancelling ? 'Anulowanie…' : 'Anuluj'}
                  </button>
                )}
                {!running && (
                  <button onClick={handleNewRun} className="btn-secondary flex items-center gap-1.5 text-sm">
                    <RotateCcw className="w-3.5 h-3.5" />
                    Nowy run
                  </button>
                )}
              </div>
            </div>

            {run && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Chunki" value={`${run.chunks_done}/${run.chunks_total}`} />
                <MetricCard label="Postęp" value={`${(run.progress_pct ?? 0).toFixed(1)}%`} />
                <MetricCard label="Q&A gotowych" value={run.records_written} />
                <MetricCard label="Pary DPO" value={run.dpo_pairs} />
              </div>
            )}

            {run && (() => {
              const elapsed = run.elapsed_seconds ?? 0;
              const done = run.chunks_done ?? 0;
              const total = run.chunks_total ?? 0;
              const speed = elapsed > 0 && done > 0 ? done / elapsed : null;
              const remaining = total - done;
              const etaSec = speed && remaining > 0 ? Math.round(remaining / speed) : null;
              const etaStr = etaSec == null ? null
                : etaSec > 3600 ? `${Math.floor(etaSec / 3600)}h ${Math.floor((etaSec % 3600) / 60)}min`
                : etaSec > 60 ? `${Math.floor(etaSec / 60)}min ${etaSec % 60}s`
                : `${etaSec}s`;
              const barStatus = run.status === 'error' ? 'error' : run.status === 'done' ? 'done' : 'running';
              return (
                <div className="space-y-1">
                  <ProgressBar
                    value={run.progress_pct ?? 0}
                    max={100}
                    label={`Chunk ${done} / ${total}`}
                    valueLabel={`${(run.progress_pct ?? 0).toFixed(1)}%${etaStr ? ` · ETA: ${etaStr}` : ''} · ${elapsed.toFixed(0)}s`}
                    status={barStatus}
                    size="md"
                  />
                  <p className="text-xs text-text-muted text-right">Batch: {run.batch_id}</p>
                </div>
              );
            })()}

            {run && <LiveLog runId={run.run_id} apiBase={API} />}
          </div>
        )}
      </section>
    </div>
  );
}

export default function AutopilotPage() {
  return (
    <Suspense fallback={<div className="w-full">Ładowanie…</div>}>
      <AutopilotPageContent />
    </Suspense>
  );
}
