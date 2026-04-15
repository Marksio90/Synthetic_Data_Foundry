'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Zap, ChevronDown, ChevronUp, Square, RotateCcw } from 'lucide-react';
import type { Document, AnalysisResult, PipelineRun } from '@/lib/api';
import LiveLog from '@/components/LiveLog';
import StatusBadge from '@/components/StatusBadge';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

function ErrorBanner({ message, onClose }: { message: string; onClose: () => void }) {
  return (
    <div className="flex items-center justify-between alert-error">
      <span>{message}</span>
      <button onClick={onClose} className="ml-4 font-bold text-lg leading-none">×</button>
    </div>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      {sub && <div className="text-xs text-text-muted mt-1">{sub}</div>}
    </div>
  );
}

export default function AutopilotPage() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [loadingDocs, setLoadingDocs] = useState(true);

  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisError, setAnalysisError] = useState('');
  const [showReasoning, setShowReasoning] = useState(false);

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [overrideQuality, setOverrideQuality] = useState(false);
  const [qualityThreshold, setQualityThreshold] = useState(0.75);
  const [overrideTurns, setOverrideTurns] = useState(false);
  const [maxTurns, setMaxTurns] = useState(3);
  const [overrideAdversarial, setOverrideAdversarial] = useState(false);
  const [adversarialRatio, setAdversarialRatio] = useState(0.1);
  const [chunkLimit, setChunkLimit] = useState(0);
  const [batchId, setBatchId] = useState('');

  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [runError, setRunError] = useState('');
  const [cancelling, setCancelling] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadDocs = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const res = await fetch(`${API}/api/documents`);
      const data = await res.json();
      const list: Document[] = data.documents ?? data ?? [];
      setDocs(list);
    } catch {
      // ignore
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  const toggleDoc = (filename: string) => {
    setSelectedDocs(prev => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
  };

  const toggleAll = () => {
    if (selectedDocs.size === docs.length) {
      setSelectedDocs(new Set());
    } else {
      setSelectedDocs(new Set(docs.map(d => d.filename)));
    }
  };

  const handleAnalyze = async () => {
    if (selectedDocs.size === 0) { setAnalysisError('Wybierz co najmniej jeden dokument.'); return; }
    setAnalyzing(true);
    setAnalysisError('');
    setAnalysis(null);
    try {
      const res = await fetch(`${API}/api/pipeline/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filenames: Array.from(selectedDocs) }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? `HTTP ${res.status}`);
      const data: AnalysisResult = await res.json();
      setAnalysis(data);
      // Pre-fill advanced settings from calibration
      setQualityThreshold(data.calibration.quality_threshold);
      setMaxTurns(data.calibration.max_turns);
      setAdversarialRatio(data.calibration.adversarial_ratio);
    } catch (e: unknown) {
      setAnalysisError(e instanceof Error ? e.message : 'Błąd analizy');
    } finally {
      setAnalyzing(false);
    }
  };

  const startPollRun = (runId: string) => {
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
    if (selectedDocs.size === 0) { setRunError('Wybierz dokumenty do przetworzenia.'); return; }
    setRunError('');
    setRunning(true);
    setRun(null);
    try {
      const payload: Record<string, unknown> = {
        filenames: Array.from(selectedDocs),
        chunk_limit: chunkLimit > 0 ? chunkLimit : null,
      };
      if (batchId.trim()) payload.batch_id = batchId.trim();
      if (overrideQuality) payload.quality_threshold = qualityThreshold;
      if (overrideTurns) payload.max_turns = maxTurns;
      if (overrideAdversarial) payload.adversarial_ratio = adversarialRatio;

      const res = await fetch(`${API}/api/pipeline/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? `HTTP ${res.status}`);
      const data: PipelineRun = await res.json();
      setRun(data);
      startPollRun(data.run_id);
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
    } catch { /* ignore */ } finally {
      setCancelling(false);
    }
  };

  const handleNewRun = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setRun(null);
    setRunning(false);
    setRunError('');
  };

  const domainColor = analysis
    ? analysis.domain_confidence >= 0.7 ? 'badge-success'
    : analysis.domain_confidence >= 0.4 ? 'badge-warning'
    : 'badge-error'
    : '';

  return (
    <div className="space-y-8 max-w-4xl">
      <div className="flex items-center gap-2">
        <Zap className="w-6 h-6 text-accent" />
        <h1 className="text-2xl font-bold text-text">AutoPilot</h1>
      </div>

      {analysisError && <ErrorBanner message={analysisError} onClose={() => setAnalysisError('')} />}
      {runError && <ErrorBanner message={runError} onClose={() => setRunError('')} />}

      {/* ─── Section 1: Dokumenty ─── */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="section-title">
            <span className="w-6 h-6 rounded-full bg-accent text-bg-base text-xs flex items-center justify-center font-bold">1</span>
            Wybierz dokumenty
          </h2>
          <span className="text-sm text-text-muted">
            {selectedDocs.size}/{docs.length} zaznaczonych
          </span>
        </div>

        <div className="card p-4 space-y-2">
          {loadingDocs ? (
            <div className="space-y-2">
              {[1,2,3].map(i => (
                <div key={i} className="h-8 bg-bg-surface2 rounded animate-pulse" />
              ))}
            </div>
          ) : docs.length === 0 ? (
            <p className="text-text-muted text-sm py-4 text-center">Brak dokumentów. Przejdź do zakładki Dokumenty, aby wgrać pliki.</p>
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
              {docs.map(doc => (
                <label key={doc.filename} className="flex items-center gap-2 cursor-pointer px-2 py-1.5 rounded hover:bg-bg-surface2 transition-colors">
                  <input
                    type="checkbox"
                    checked={selectedDocs.has(doc.filename)}
                    onChange={() => toggleDoc(doc.filename)}
                    className="w-4 h-4"
                  />
                  <span className="text-sm text-text flex-1">{doc.filename}</span>
                  <span className="text-xs text-text-muted">{doc.chunk_count} ch · {doc.sample_count} Q&A</span>
                </label>
              ))}
            </>
          )}
        </div>
      </section>

      {/* ─── Section 2: Analiza ─── */}
      <section className="space-y-3">
        <h2 className="section-title">
          <span className="w-6 h-6 rounded-full bg-accent text-bg-base text-xs flex items-center justify-center font-bold">2</span>
          Analiza dokumentów
        </h2>

        <button
          onClick={handleAnalyze}
          disabled={analyzing || selectedDocs.size === 0}
          className="btn-primary"
        >
          {analyzing ? (
            <>
              <div className="w-4 h-4 border-2 border-bg-base/30 border-t-bg-base rounded-full animate-spin" />
              Analizowanie...
            </>
          ) : '🔍 Analizuj dokumenty'}
        </button>

        {analysis && (
          <div className="card p-5 space-y-5">
            {/* Language + Domain */}
            <div className="flex flex-wrap gap-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">Język:</span>
                <span className={`badge ${analysis.language === 'pl' || analysis.language === 'PL' ? 'badge-success' : 'badge-warning'}`}>
                  {analysis.language.toUpperCase()}
                  {analysis.translation_required && ' (wymaga tłumaczenia)'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">Domena:</span>
                <span className={`badge ${domainColor}`}>
                  {analysis.domain_label} — {((analysis.domain_confidence ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* Perspectives */}
            {analysis.perspectives.length > 0 && (
              <div>
                <p className="text-sm text-text-muted mb-2">Perspektywy:</p>
                <div className="flex flex-wrap gap-2">
                  {analysis.perspectives.map(p => (
                    <span key={p} className="badge badge-accent">{p}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Auto-decisions */}
            {analysis.auto_decisions.length > 0 && (
              <div>
                <p className="text-sm text-text-muted mb-2">Automatyczne decyzje:</p>
                <div className="space-y-1">
                  {analysis.auto_decisions.map((d, i) => (
                    <div key={i} className={`text-sm px-3 py-1.5 rounded ${d.toLowerCase().includes('warn') || d.toLowerCase().includes('uwaga') ? 'alert-warning' : 'alert-info'}`}>
                      {d}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Calibration */}
            <div>
              <p className="text-sm text-text-muted mb-2">Kalibracja:</p>
              <div className="grid grid-cols-3 gap-3">
                <MetricCard label="Próg jakości" value={(analysis.calibration?.quality_threshold ?? 0).toFixed(2)} />
                <MetricCard label="Maks. tury" value={analysis.calibration?.max_turns ?? '—'} />
                <MetricCard label="Adwersarial" value={((analysis.calibration?.adversarial_ratio ?? 0) * 100).toFixed(0) + '%'} />
              </div>
            </div>

            {/* Reasoning collapsible */}
            {analysis.calibration.reasoning && analysis.calibration.reasoning.length > 0 && (
              <div>
                <button
                  onClick={() => setShowReasoning(v => !v)}
                  className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text transition-colors"
                >
                  {showReasoning ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  {showReasoning ? 'Ukryj' : 'Pokaż'} uzasadnienie
                </button>
                {showReasoning && (
                  <ul className="mt-2 space-y-1 pl-4">
                    {analysis.calibration.reasoning.map((r, i) => (
                      <li key={i} className="text-sm text-text-muted list-disc">{r}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        )}
      </section>

      {/* ─── Section 3: Uruchom ─── */}
      <section className="space-y-4">
        <h2 className="section-title">
          <span className="w-6 h-6 rounded-full bg-accent text-bg-base text-xs flex items-center justify-center font-bold">3</span>
          Uruchom pipeline
        </h2>

        {/* Advanced settings */}
        <div className="card overflow-hidden">
          <button
            onClick={() => setShowAdvanced(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-text hover:bg-bg-surface2 transition-colors"
          >
            Zaawansowane ustawienia
            {showAdvanced ? <ChevronUp className="w-4 h-4 text-text-muted" /> : <ChevronDown className="w-4 h-4 text-text-muted" />}
          </button>
          {showAdvanced && (
            <div className="px-4 pb-4 space-y-4 border-t border-border pt-4">
              {/* Quality threshold */}
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-text-muted w-48">
                  <input type="checkbox" checked={overrideQuality} onChange={e => setOverrideQuality(e.target.checked)} />
                  Próg jakości
                </label>
                <div className="flex-1 flex items-center gap-3">
                  <input
                    type="range"
                    min={0.5} max={1.0} step={0.01}
                    value={qualityThreshold}
                    onChange={e => setQualityThreshold(parseFloat(e.target.value))}
                    disabled={!overrideQuality}
                    className="flex-1"
                  />
                  <span className="text-sm text-text w-10 text-right">{qualityThreshold.toFixed(2)}</span>
                </div>
              </div>

              {/* Max turns */}
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-text-muted w-48">
                  <input type="checkbox" checked={overrideTurns} onChange={e => setOverrideTurns(e.target.checked)} />
                  Maks. tury dialogu
                </label>
                <select
                  value={maxTurns}
                  onChange={e => setMaxTurns(parseInt(e.target.value))}
                  disabled={!overrideTurns}
                  className="input-field w-24"
                >
                  {[1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>

              {/* Adversarial ratio */}
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-text-muted w-48">
                  <input type="checkbox" checked={overrideAdversarial} onChange={e => setOverrideAdversarial(e.target.checked)} />
                  Współczynnik adwers.
                </label>
                <div className="flex-1 flex items-center gap-3">
                  <input
                    type="range"
                    min={0} max={0.3} step={0.01}
                    value={adversarialRatio}
                    onChange={e => setAdversarialRatio(parseFloat(e.target.value))}
                    disabled={!overrideAdversarial}
                    className="flex-1"
                  />
                  <span className="text-sm text-text w-10 text-right">{(adversarialRatio * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* Chunk limit */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-text-muted w-48">Limit chunków</label>
                <input
                  type="number"
                  min={0}
                  value={chunkLimit}
                  onChange={e => setChunkLimit(parseInt(e.target.value) || 0)}
                  className="input-field w-32"
                  placeholder="0 = wszystkie"
                />
                {chunkLimit === 0 && <span className="text-xs text-text-muted">wszystkie</span>}
              </div>

              {/* Batch ID */}
              <div className="flex items-center gap-4">
                <label className="text-sm text-text-muted w-48">Batch ID</label>
                <input
                  type="text"
                  value={batchId}
                  onChange={e => setBatchId(e.target.value)}
                  className="input-field w-64"
                  placeholder="auto-generowany"
                />
              </div>
            </div>
          )}
        </div>

        {/* Summary */}
        <div className="text-sm text-text-muted">
          Uruchomi: <strong className="text-text">{selectedDocs.size}</strong> dok. ·
          limit <strong className="text-text">{chunkLimit > 0 ? chunkLimit : '∞'}</strong> chunków
          {batchId && <> · batch: <strong className="text-text">{batchId}</strong></>}
        </div>

        {/* Run button */}
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

        {/* Progress */}
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
                    {cancelling ? 'Anulowanie...' : 'Anuluj'}
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

            {/* Metrics */}
            {run && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard
                  label="Chunki"
                  value={`${run.chunks_done}/${run.chunks_total}`}
                />
                <MetricCard
                  label="Postęp"
                  value={`${(run.progress_pct ?? 0).toFixed(1)}%`}
                />
                <MetricCard
                  label="Q&A gotowych"
                  value={run.records_written}
                />
                <MetricCard
                  label="Pary DPO"
                  value={run.dpo_pairs}
                />
              </div>
            )}

            {/* Progress bar */}
            {run && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-text-muted">
                  <span>Chunks: {run.chunks_done} / {run.chunks_total}</span>
                  <span>{(run.elapsed_seconds ?? 0).toFixed(0)}s</span>
                </div>
                <div className="w-full bg-bg-surface2 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      run.status === 'error' ? 'bg-error' :
                      run.status === 'done' ? 'bg-success' :
                      'bg-accent'
                    }`}
                    style={{ width: `${Math.min(100, run.progress_pct)}%` }}
                  />
                </div>
                <p className="text-xs text-text-muted text-right">Batch: {run.batch_id}</p>
              </div>
            )}

            {/* Live log */}
            {run && (
              <LiveLog runId={run.run_id} apiBase={API} />
            )}
          </div>
        )}
      </section>
    </div>
  );
}
