'use client';

import { useCallback, useEffect, useState } from 'react';
import { Database, X, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react';
import type { Sample, SampleDetail, SampleStats } from '@/lib/api';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

const PAGE_SIZE = 20;

function formatScore(score: number | null | undefined): string {
  if (score == null) return '—';
  return score.toFixed(3);
}

function ScoreBadge({ score }: { score: number }) {
  const color = score >= 0.85 ? 'badge-success' : score >= 0.7 ? 'badge-warning' : 'badge-error';
  return <span className={`badge ${color}`}>{score.toFixed(2)}</span>;
}

function formatDate(iso: string): string {
  try { return new Date(iso).toLocaleString('pl-PL', { dateStyle: 'short', timeStyle: 'short' }); }
  catch { return iso; }
}

function truncate(str: string, len: number): string {
  return str.length > len ? str.slice(0, len) + '…' : str;
}

interface FiltersState {
  perspective: string;
  difficulty: string;
  min_score: string;
  batch_id: string;
}

export default function DatasetPage() {
  const [stats, setStats] = useState<SampleStats | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState<FiltersState>({ perspective: '', difficulty: '', min_score: '', batch_id: '' });
  const [selectedSample, setSelectedSample] = useState<SampleDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/samples/stats`);
      if (res.ok) setStats(await res.json());
    } catch { /* ignore */ }
  }, []);

  const fetchSamples = useCallback(async (pageNum: number, f: FiltersState) => {
    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({ limit: String(PAGE_SIZE), offset: String(pageNum * PAGE_SIZE) });
      if (f.perspective) params.set('perspective', f.perspective);
      if (f.difficulty) params.set('difficulty', f.difficulty);
      if (f.min_score) params.set('min_score', f.min_score);
      if (f.batch_id) params.set('batch_id', f.batch_id);
      const res = await fetch(`${API}/api/samples?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSamples(data.samples ?? data ?? []);
      setTotal(data.total ?? (data.samples ?? data ?? []).length);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd pobierania próbek');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchStats(); }, [fetchStats]);
  useEffect(() => { fetchSamples(page, filters); }, [page, filters, fetchSamples]);

  const handleFilterChange = (key: keyof FiltersState, value: string) => {
    setPage(0);
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const resetFilters = () => {
    setPage(0);
    setFilters({ perspective: '', difficulty: '', min_score: '', batch_id: '' });
  };

  const openDetail = async (sample: Sample) => {
    setLoadingDetail(true);
    try {
      const res = await fetch(`${API}/api/samples/${sample.id}`);
      if (res.ok) {
        setSelectedSample(await res.json());
      } else {
        setSelectedSample({ ...sample, rejected_answer: null, judge_model: null, judge_reasoning: null, record_index: null, conversation_json: null });
      }
    } catch {
      setSelectedSample({ ...sample, rejected_answer: null, judge_model: null, judge_reasoning: null, record_index: null, conversation_json: null });
    } finally {
      setLoadingDetail(false);
    }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const startItem = page * PAGE_SIZE + 1;
  const endItem = Math.min((page + 1) * PAGE_SIZE, total);

  const perspectives = stats ? Object.keys(stats.perspectives) : [];
  const difficulties = stats ? Object.keys(stats.difficulties) : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Database className="w-6 h-6 text-accent" />
        <h1 className="text-2xl font-bold text-text">Dataset</h1>
      </div>

      {error && (
        <div className="alert-error flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError('')} className="font-bold text-lg">×</button>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="metric-card">
              <div className="metric-value">{stats.total.toLocaleString('pl-PL')}</div>
              <div className="metric-label">Total próbek</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{formatScore(stats.avg_quality_score)}</div>
              <div className="metric-label">Średni wynik</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{stats.dpo_pairs.toLocaleString('pl-PL')}</div>
              <div className="metric-label">Pary DPO</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{perspectives.length}</div>
              <div className="metric-label">Perspektywy</div>
            </div>
          </div>
          {/* Perspective breakdown */}
          {perspectives.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {perspectives.map(p => (
                <span key={p} className="badge badge-accent text-xs">
                  {p}: {stats.perspectives[p]}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-text-muted">Perspektywa</label>
          <select
            value={filters.perspective}
            onChange={e => handleFilterChange('perspective', e.target.value)}
            className="input-field w-40"
          >
            <option value="">Wszystkie</option>
            {perspectives.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-text-muted">Trudność</label>
          <select
            value={filters.difficulty}
            onChange={e => handleFilterChange('difficulty', e.target.value)}
            className="input-field w-36"
          >
            <option value="">Wszystkie</option>
            {difficulties.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-text-muted">Min. wynik</label>
          <input
            type="number"
            min={0} max={1} step={0.01}
            value={filters.min_score}
            onChange={e => handleFilterChange('min_score', e.target.value)}
            className="input-field w-28"
            placeholder="0.0"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-text-muted">Batch ID</label>
          <input
            type="text"
            value={filters.batch_id}
            onChange={e => handleFilterChange('batch_id', e.target.value)}
            className="input-field w-40"
            placeholder="Wszystkie"
          />
        </div>
        <button onClick={resetFilters} className="btn-secondary h-9">
          <RefreshCw className="w-3.5 h-3.5" />
          Reset
        </button>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-bg-surface2 text-text-muted text-xs uppercase tracking-wider">
                <th className="px-4 py-3 text-left w-10">#</th>
                <th className="px-4 py-3 text-left">Pytanie</th>
                <th className="px-4 py-3 text-left">Perspektywa</th>
                <th className="px-4 py-3 text-left">Trudność</th>
                <th className="px-4 py-3 text-right">Wynik</th>
                <th className="px-4 py-3 text-center">Adwers.</th>
                <th className="px-4 py-3 text-left">Data</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="border-b border-border">
                    {Array.from({ length: 7 }).map((_, j) => (
                      <td key={j} className="px-4 py-3">
                        <div className="h-4 bg-bg-surface2 rounded animate-pulse" />
                      </td>
                    ))}
                  </tr>
                ))
              ) : samples.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-text-muted">
                    Brak próbek spełniających kryteria filtrów.
                  </td>
                </tr>
              ) : samples.map((s, idx) => (
                <tr
                  key={s.id}
                  className="border-b border-border hover:bg-bg-surface2 cursor-pointer transition-colors"
                  onClick={() => openDetail(s)}
                >
                  <td className="px-4 py-3 text-text-muted">{page * PAGE_SIZE + idx + 1}</td>
                  <td className="px-4 py-3 text-text max-w-xs">
                    <span title={s.question}>{truncate(s.question, 80)}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="badge badge-accent text-xs">{s.perspective}</span>
                  </td>
                  <td className="px-4 py-3 text-text-muted text-xs">{s.difficulty}</td>
                  <td className="px-4 py-3 text-right">
                    <ScoreBadge score={s.quality_score} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    {s.is_adversarial ? (
                      <span className="text-warning text-xs">⚠ Tak</span>
                    ) : (
                      <span className="text-text-muted text-xs">Nie</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-text-muted text-xs">{formatDate(s.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {total > 0 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-border bg-bg-surface2 text-sm">
            <span className="text-text-muted">
              {startItem}–{endItem} z {total.toLocaleString('pl-PL')}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="btn-secondary py-1 px-2"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-text-muted text-xs">
                {page + 1} / {totalPages}
              </span>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="btn-secondary py-1 px-2"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Detail slide-out panel */}
      {(selectedSample || loadingDetail) && (
        <div className="fixed inset-0 z-50 flex">
          {/* Backdrop */}
          <div
            className="flex-1 bg-black/50"
            onClick={() => setSelectedSample(null)}
          />
          {/* Panel */}
          <div className="w-full max-w-xl bg-bg-surface border-l border-border overflow-y-auto flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-border flex-shrink-0">
              <h2 className="font-semibold text-text">Szczegóły próbki</h2>
              <button
                onClick={() => setSelectedSample(null)}
                className="text-text-muted hover:text-text transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {loadingDetail ? (
              <div className="flex-1 flex items-center justify-center">
                <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
              </div>
            ) : selectedSample && (
              <div className="flex-1 p-6 space-y-5 text-sm">
                {/* Metadata badges */}
                <div className="flex flex-wrap gap-2">
                  <span className="badge badge-accent">{selectedSample.perspective}</span>
                  <span className="badge badge-muted">{selectedSample.difficulty}</span>
                  <span className="badge badge-muted">{selectedSample.question_type}</span>
                  <ScoreBadge score={selectedSample.quality_score} />
                  {selectedSample.is_adversarial && <span className="badge badge-warning">Adwersarialny</span>}
                  {selectedSample.has_dpo && <span className="badge badge-success">DPO</span>}
                </div>

                {/* Question */}
                <div>
                  <p className="text-xs text-text-muted uppercase tracking-wider mb-2">Pytanie</p>
                  <div className="bg-bg-surface2 rounded-md p-3 text-text leading-relaxed whitespace-pre-wrap">
                    {selectedSample.question}
                  </div>
                </div>

                {/* Answer */}
                <div>
                  <p className="text-xs text-text-muted uppercase tracking-wider mb-2">Odpowiedź</p>
                  <div className="bg-bg-surface2 rounded-md p-3 text-text leading-relaxed whitespace-pre-wrap">
                    {selectedSample.answer}
                  </div>
                </div>

                {/* Rejected answer (DPO) */}
                {selectedSample.rejected_answer && (
                  <div>
                    <p className="text-xs text-error uppercase tracking-wider mb-2">Odrzucona odpowiedź (DPO)</p>
                    <div className="bg-error/5 border border-error/20 rounded-md p-3 text-error/80 leading-relaxed whitespace-pre-wrap line-through decoration-error/40">
                      {selectedSample.rejected_answer}
                    </div>
                  </div>
                )}

                {/* Judge */}
                {selectedSample.judge_model && (
                  <div>
                    <p className="text-xs text-text-muted uppercase tracking-wider mb-1">Model sędziego</p>
                    <span className="badge badge-muted">{selectedSample.judge_model}</span>
                  </div>
                )}

                {selectedSample.judge_reasoning && (
                  <div>
                    <p className="text-xs text-text-muted uppercase tracking-wider mb-2">Uzasadnienie sędziego</p>
                    <div className="bg-bg-surface2 rounded-md p-3 text-text-muted text-xs leading-relaxed whitespace-pre-wrap">
                      {selectedSample.judge_reasoning}
                    </div>
                  </div>
                )}

                {/* Extra metadata */}
                <div className="grid grid-cols-2 gap-3 text-xs text-text-muted">
                  <div><span className="text-text-muted">ID:</span> <span className="text-text">{selectedSample.id}</span></div>
                  <div><span className="text-text-muted">Batch:</span> <span className="text-text">{selectedSample.batch_id}</span></div>
                  <div><span className="text-text-muted">Tury:</span> <span className="text-text">{selectedSample.turn_count}</span></div>
                  {selectedSample.record_index != null && (
                    <div><span className="text-text-muted">Record idx:</span> <span className="text-text">{selectedSample.record_index}</span></div>
                  )}
                  <div className="col-span-2"><span className="text-text-muted">Utworzono:</span> <span className="text-text">{formatDate(selectedSample.created_at)}</span></div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
