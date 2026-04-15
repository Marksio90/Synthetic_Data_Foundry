'use client';

import { useCallback, useEffect, useState } from 'react';
import { ClipboardCheck, CheckCircle, X, Edit3, RefreshCw } from 'lucide-react';
import type { Sample } from '@/lib/api';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

interface ReviewSample extends Sample {
  editText?: string;
  editMode?: boolean;
  saving?: boolean;
}

interface AutoReviewResult {
  approved?: number;
  queued?: number;
  rejected?: number;
  processed?: number;
  message?: string;
}

function ScoreBadge({ score }: { score: number }) {
  const color = score >= 0.85 ? 'text-success' : score >= 0.7 ? 'text-warning' : 'text-error';
  return <span className={`font-mono text-sm font-bold ${color}`}>{score.toFixed(3)}</span>;
}

export default function ReviewPage() {
  const [samples, setSamples] = useState<ReviewSample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [autoReviewResult, setAutoReviewResult] = useState<AutoReviewResult | null>(null);
  const [autoReviewing, setAutoReviewing] = useState(false);

  const fetchSamples = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({ min_score: '0.70', max_score: '0.88', limit: '50' });
      const res = await fetch(`${API}/api/samples?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const list: Sample[] = data.samples ?? data ?? [];
      setSamples(list.map(s => ({ ...s })));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd pobierania próbek');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchSamples(); }, [fetchSamples]);

  const notify = (msg: string, type: 'ok' | 'err') => {
    if (type === 'ok') { setSuccess(msg); setTimeout(() => setSuccess(''), 4000); }
    else { setError(msg); setTimeout(() => setError(''), 5000); }
  };

  const patchSample = async (id: number, action: string, editedAnswer?: string) => {
    const params: Record<string, string> = { action };
    if (editedAnswer !== undefined) params.edited_answer = editedAnswer;
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`${API}/api/samples/${id}/review?${qs}`, { method: 'PATCH' });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ?? `HTTP ${res.status}`);
    }
  };

  const handleApprove = async (id: number) => {
    setSamples(prev => prev.map(s => s.id === id ? { ...s, saving: true } : s));
    try {
      await patchSample(id, 'approve');
      setSamples(prev => prev.filter(s => s.id !== id));
      notify('Zatwierdzono próbkę.', 'ok');
    } catch (e: unknown) {
      notify(e instanceof Error ? e.message : 'Błąd zatwierdzania', 'err');
      setSamples(prev => prev.map(s => s.id === id ? { ...s, saving: false } : s));
    }
  };

  const handleReject = async (id: number) => {
    if (!confirm('Odrzucić tę próbkę?')) return;
    setSamples(prev => prev.map(s => s.id === id ? { ...s, saving: true } : s));
    try {
      await patchSample(id, 'reject');
      setSamples(prev => prev.filter(s => s.id !== id));
      notify('Odrzucono próbkę.', 'ok');
    } catch (e: unknown) {
      notify(e instanceof Error ? e.message : 'Błąd odrzucania', 'err');
      setSamples(prev => prev.map(s => s.id === id ? { ...s, saving: false } : s));
    }
  };

  const handleEditToggle = (id: number) => {
    setSamples(prev => prev.map(s =>
      s.id === id
        ? { ...s, editMode: !s.editMode, editText: s.editMode ? s.editText : s.answer }
        : s
    ));
  };

  const handleEditSave = async (id: number) => {
    const s = samples.find(x => x.id === id);
    if (!s || !s.editText) return;
    setSamples(prev => prev.map(x => x.id === id ? { ...x, saving: true } : x));
    try {
      await patchSample(id, 'edit', s.editText);
      setSamples(prev => prev.filter(x => x.id !== id));
      notify('Zapisano edycję i zatwierdzono próbkę.', 'ok');
    } catch (e: unknown) {
      notify(e instanceof Error ? e.message : 'Błąd zapisywania', 'err');
      setSamples(prev => prev.map(x => x.id === id ? { ...x, saving: false } : x));
    }
  };

  const handleAutoReview = async () => {
    setAutoReviewing(true);
    setAutoReviewResult(null);
    try {
      const res = await fetch(`${API}/api/samples/auto-review`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setAutoReviewResult(data);
      await fetchSamples();
    } catch (e: unknown) {
      notify(e instanceof Error ? e.message : 'Błąd auto-review', 'err');
    } finally {
      setAutoReviewing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ClipboardCheck className="w-6 h-6 text-accent" />
          <h1 className="text-2xl font-bold text-text">Review</h1>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchSamples}
            disabled={loading}
            className="btn-secondary flex items-center gap-1.5 text-sm"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Odśwież
          </button>
          <button
            onClick={handleAutoReview}
            disabled={autoReviewing}
            className="btn-primary text-sm"
          >
            {autoReviewing ? (
              <>
                <div className="w-4 h-4 border-2 border-bg-base/30 border-t-bg-base rounded-full animate-spin" />
                Auto-review...
              </>
            ) : '🤖 Auto-Review'}
          </button>
        </div>
      </div>

      {/* Banners */}
      {error && (
        <div className="alert-error flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError('')} className="font-bold text-lg">×</button>
        </div>
      )}
      {success && (
        <div className="alert-success flex items-center justify-between">
          <span>{success}</span>
          <button onClick={() => setSuccess('')} className="font-bold text-lg">×</button>
        </div>
      )}

      {/* Auto-review result */}
      {autoReviewResult && (
        <div className="card p-4">
          <p className="text-sm font-medium text-text mb-3">Wyniki Auto-Review:</p>
          <div className="flex flex-wrap gap-4 text-sm">
            {autoReviewResult.approved != null && (
              <span className="text-success">
                ✅ Zatwierdzono: <strong>{autoReviewResult.approved}</strong>
              </span>
            )}
            {autoReviewResult.queued != null && (
              <span className="text-warning">
                🕐 W kolejce: <strong>{autoReviewResult.queued}</strong>
              </span>
            )}
            {autoReviewResult.rejected != null && (
              <span className="text-error">
                ❌ Odrzucono: <strong>{autoReviewResult.rejected}</strong>
              </span>
            )}
            {autoReviewResult.processed != null && (
              <span className="text-text-muted">
                Przetworzono: <strong className="text-text">{autoReviewResult.processed}</strong>
              </span>
            )}
            {autoReviewResult.message && (
              <span className="text-text-muted">{autoReviewResult.message}</span>
            )}
          </div>
          <button onClick={() => setAutoReviewResult(null)} className="mt-2 text-xs text-text-muted hover:text-text">
            Zamknij
          </button>
        </div>
      )}

      {/* Info */}
      <p className="text-sm text-text-muted">
        Wyświetlane próbki z wynikiem 0.70–0.88 wymagające ludzkiej weryfikacji.
        Razem: <strong className="text-text">{samples.length}</strong>
      </p>

      {/* Cards */}
      {loading ? (
        <div className="space-y-4">
          {[1,2,3].map(i => (
            <div key={i} className="card p-5 space-y-3">
              <div className="h-5 bg-bg-surface2 rounded animate-pulse w-3/4" />
              <div className="h-4 bg-bg-surface2 rounded animate-pulse w-full" />
              <div className="h-4 bg-bg-surface2 rounded animate-pulse w-5/6" />
            </div>
          ))}
        </div>
      ) : samples.length === 0 ? (
        <div className="card p-12 text-center">
          <ClipboardCheck className="w-12 h-12 mx-auto mb-3 text-success opacity-60" />
          <p className="text-lg font-medium text-text">Brak próbek do przeglądu</p>
          <p className="text-text-muted text-sm mt-1">Wszystkie próbki zostały przejrzane lub nie ma próbek w zakresie 0.70–0.88.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {samples.map(sample => (
            <div
              key={sample.id}
              className={`card p-5 space-y-4 transition-opacity ${sample.saving ? 'opacity-50 pointer-events-none' : ''}`}
            >
              {/* Header */}
              <div className="flex items-start justify-between gap-4">
                <div className="flex flex-wrap gap-2">
                  <span className="badge badge-accent text-xs">{sample.perspective}</span>
                  <span className="badge badge-muted text-xs">{sample.difficulty}</span>
                  {sample.is_adversarial && <span className="badge badge-warning text-xs">⚠ Adwersarialny</span>}
                </div>
                <ScoreBadge score={sample.quality_score} />
              </div>

              {/* Question */}
              <div>
                <p className="text-xs text-text-muted uppercase tracking-wider mb-1.5">Pytanie</p>
                <p className="text-text text-sm leading-relaxed line-clamp-4">
                  {sample.question.slice(0, 200)}{sample.question.length > 200 ? '…' : ''}
                </p>
              </div>

              {/* Answer */}
              <div>
                <p className="text-xs text-text-muted uppercase tracking-wider mb-1.5">Odpowiedź</p>
                {sample.editMode ? (
                  <div className="space-y-2">
                    <textarea
                      value={sample.editText ?? sample.answer}
                      onChange={e => setSamples(prev => prev.map(s => s.id === sample.id ? { ...s, editText: e.target.value } : s))}
                      className="input-field h-32 font-mono text-xs resize-y"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleEditSave(sample.id)}
                        className="btn-primary text-xs py-1.5"
                      >
                        Zapisz i zatwierdź
                      </button>
                      <button
                        onClick={() => handleEditToggle(sample.id)}
                        className="btn-secondary text-xs py-1.5"
                      >
                        Anuluj
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className="text-text-muted text-sm leading-relaxed">
                    {sample.answer.slice(0, 300)}{sample.answer.length > 300 ? '…' : ''}
                  </p>
                )}
              </div>

              {/* Actions */}
              {!sample.editMode && (
                <div className="flex items-center gap-2 pt-1 border-t border-border">
                  <button
                    onClick={() => handleApprove(sample.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-success/10 border border-success/30 text-success text-sm hover:bg-success/20 transition-colors"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Zatwierdź
                  </button>
                  <button
                    onClick={() => handleEditToggle(sample.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-accent/10 border border-accent/30 text-accent text-sm hover:bg-accent/20 transition-colors"
                  >
                    <Edit3 className="w-4 h-4" />
                    Edytuj
                  </button>
                  <button
                    onClick={() => handleReject(sample.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-error/10 border border-error/30 text-error text-sm hover:bg-error/20 transition-colors"
                  >
                    <X className="w-4 h-4" />
                    Odrzuć
                  </button>
                  <span className="ml-auto text-xs text-text-muted">ID: {sample.id} · Batch: {sample.batch_id}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
