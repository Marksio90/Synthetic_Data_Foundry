'use client';

/**
 * ModelGapChart — wizualizacja luk wiedzy z podziałem na modele LLM
 *
 * Pobiera GET /api/scout/gaps/models i wyświetla siatkę kart modeli.
 * Każda karta pokazuje: nazwę modelu, liczbę tematów, avg knowledge_gap_score
 * oraz 3 najważniejsze tematy do ingestowania.
 */

import { useCallback, useEffect, useState } from 'react';
import { getGapsByModel } from '@/lib/api';
import type { ModelGapsResponse, ScoutTopic } from '@/lib/api';
import { Brain, ChevronRight, Loader2, RefreshCw } from 'lucide-react';

// ─── Stałe ────────────────────────────────────────────────────────────────────

const MODEL_META: Record<string, { label: string; color: string; ring: string }> = {
  'gpt-4o':           { label: 'GPT-4o',        color: 'from-emerald-500/15 to-emerald-500/5', ring: 'border-emerald-500/35' },
  'claude-3.5-sonnet':{ label: 'Claude 3.5',     color: 'from-purple-500/15 to-purple-500/5',  ring: 'border-purple-500/35' },
  'llama-3':          { label: 'Llama 3',        color: 'from-orange-500/15 to-orange-500/5',  ring: 'border-orange-500/35' },
  'gemini-1.5':       { label: 'Gemini 1.5',     color: 'from-blue-500/15 to-blue-500/5',      ring: 'border-blue-500/35' },
  'mistral':          { label: 'Mistral',         color: 'from-cyan-500/15 to-cyan-500/5',      ring: 'border-cyan-500/35' },
  'all':              { label: 'Wszystkie modele', color: 'from-accent/10 to-accent/5',         ring: 'border-accent/25' },
};

const DEFAULT_META = {
  label: '',
  color: 'from-text-muted/10 to-text-muted/5',
  ring: 'border-border',
};

function getMeta(model: string) {
  return MODEL_META[model] ?? { ...DEFAULT_META, label: model };
}

// ─── Pasek procentowy ──────────────────────────────────────────────────────────

function GapBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  const barCls =
    pct >= 70 ? 'bg-success' :
    pct >= 40 ? 'bg-warning' :
    'bg-error/70';

  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[10px] text-text-muted">
        <span>{label}</span>
        <span className="tabular-nums font-medium">{pct}%</span>
      </div>
      <div className="w-full h-1 rounded-full bg-bg-surface2 overflow-hidden">
        <div
          className={`h-full rounded-full ${barCls} transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ─── Karta jednego modelu ──────────────────────────────────────────────────────

function ModelCard({
  model,
  topics,
  onClick,
}: {
  model: string;
  topics: ScoutTopic[];
  onClick: () => void;
}) {
  const meta = getMeta(model);
  const avgGap =
    topics.length > 0
      ? topics.reduce((s, t) => s + (t.knowledge_gap_score ?? t.score), 0) / topics.length
      : 0;
  const avgUncertainty =
    topics.length > 0
      ? topics.reduce((s, t) => s + t.llm_uncertainty, 0) / topics.length
      : 0;

  const topThree = topics.slice(0, 3);

  return (
    <button
      onClick={onClick}
      className={`
        text-left w-full rounded-xl border bg-gradient-to-br p-4 flex flex-col gap-3
        hover:brightness-110 transition-all duration-200 cursor-pointer
        ${meta.color} ${meta.ring}
      `}
    >
      {/* Nagłówek karty */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-text-muted shrink-0" />
          <span className="text-sm font-semibold text-text">{meta.label}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[11px] tabular-nums font-medium text-text-muted">
            {topics.length} temat{topics.length === 1 ? '' : topics.length < 5 ? 'y' : 'ów'}
          </span>
          <ChevronRight className="w-3.5 h-3.5 text-text-muted/50" />
        </div>
      </div>

      {/* Paski wyników */}
      <div className="space-y-1.5">
        <GapBar value={avgGap} label="Śr. luka wiedzy" />
        <GapBar value={avgUncertainty} label="Śr. niepewność LLM" />
      </div>

      {/* Top 3 tematy */}
      {topThree.length > 0 && (
        <ul className="space-y-1">
          {topThree.map((t) => (
            <li
              key={t.topic_id}
              className="text-[11px] text-text-muted line-clamp-1 flex items-center gap-1"
            >
              <span className="w-1 h-1 rounded-full bg-text-muted/40 shrink-0" />
              {t.title}
            </li>
          ))}
        </ul>
      )}

      {topics.length === 0 && (
        <p className="text-[11px] text-text-muted/50 italic">Brak tematów dla tego modelu</p>
      )}
    </button>
  );
}

// ─── Panel szczegółów modelu ───────────────────────────────────────────────────

function ModelDetail({
  model,
  topics,
  onClose,
}: {
  model: string;
  topics: ScoutTopic[];
  onClose: () => void;
}) {
  const meta = getMeta(model);

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-2 pb-3 border-b border-border shrink-0">
        <button
          onClick={onClose}
          className="text-[11px] text-accent hover:underline flex items-center gap-0.5"
        >
          ← Powrót
        </button>
        <span className="text-text-muted/40">/</span>
        <Brain className="w-3.5 h-3.5 text-text-muted" />
        <span className="text-sm font-semibold text-text">{meta.label}</span>
        <span className="text-[11px] text-text-muted ml-1">
          {topics.length} temat{topics.length === 1 ? '' : topics.length < 5 ? 'y' : 'ów'}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 pt-3 space-y-2">
        {topics.length === 0 ? (
          <p className="text-sm text-text-muted italic py-8 text-center">
            Brak tematów dla modelu {meta.label}
          </p>
        ) : (
          topics.map((t) => {
            const gapScore = t.knowledge_gap_score ?? t.score;
            const pct = Math.round(gapScore * 100);
            const scoreCls =
              pct >= 70 ? 'text-success border-success/40 bg-success/10' :
              pct >= 40 ? 'text-warning border-warning/40 bg-warning/10' :
              'text-error border-error/40 bg-error/10';

            return (
              <div
                key={t.topic_id}
                className="rounded-lg border border-border bg-bg-surface p-3 flex items-start gap-3"
              >
                <span className={`shrink-0 w-8 h-8 rounded border flex items-center justify-center text-[11px] font-bold tabular-nums ${scoreCls}`}>
                  {pct}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-[12px] font-medium text-text line-clamp-2">{t.title}</p>
                  <p className="text-[11px] text-text-muted mt-0.5 line-clamp-1">{t.summary}</p>
                  <div className="flex items-center gap-2 mt-1 text-[10px] text-text-muted/60">
                    <span>Źródła: {t.source_count}</span>
                    {t.citation_velocity != null && t.citation_velocity > 0 && (
                      <span>Prędkość: {t.citation_velocity.toFixed(2)}/dzień</span>
                    )}
                    {t.source_tier && <span>Tier {t.source_tier}</span>}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ─── Główny komponent ──────────────────────────────────────────────────────────

export default function ModelGapChart({ className = '' }: { className?: string }) {
  const [data, setData] = useState<ModelGapsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getGapsByModel(undefined, 15);
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd pobierania danych');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Widok szczegółowy jednego modelu
  if (selected && data) {
    return (
      <div className={`flex flex-col h-full min-h-0 ${className}`}>
        <ModelDetail
          model={selected}
          topics={data.models[selected] ?? []}
          onClose={() => setSelected(null)}
        />
      </div>
    );
  }

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Nagłówek sekcji */}
      <div className="flex items-center gap-2">
        <Brain className="w-4 h-4 text-accent" />
        <h2 className="text-sm font-semibold text-text">Luki wiedzy per model</h2>
        {data && (
          <span className="text-[11px] text-text-muted">
            — {data.total_topics} temat{data.total_topics === 1 ? '' : data.total_topics < 5 ? 'y' : 'ów'} · {data.model_count} modeli
          </span>
        )}
        <div className="flex-1" />
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-1 text-[11px] text-text-muted hover:text-accent transition-colors disabled:opacity-50"
        >
          {loading
            ? <Loader2 className="w-3 h-3 animate-spin" />
            : <RefreshCw className="w-3 h-3" />}
          Odśwież
        </button>
      </div>

      {/* Błąd */}
      {error && (
        <div className="rounded-lg bg-error/10 border border-error/30 text-error text-sm px-4 py-3">
          {error}
        </div>
      )}

      {/* Stan ładowania */}
      {loading && !data && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 text-accent animate-spin" />
        </div>
      )}

      {/* Siatka kart modeli */}
      {data && !loading && Object.keys(data.models).length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
          <Brain className="w-12 h-12 text-text-muted/20" />
          <div>
            <p className="text-text font-medium">Brak danych o lukach modeli</p>
            <p className="text-sm text-text-muted mt-1">
              Uruchom skan — tematy pojawią się z przypisanymi celami modelowymi
            </p>
          </div>
        </div>
      )}

      {data && Object.keys(data.models).length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(data.models)
            .sort(([, a], [, b]) => {
              const avgA = a.reduce((s, t) => s + (t.knowledge_gap_score ?? t.score), 0) / (a.length || 1);
              const avgB = b.reduce((s, t) => s + (t.knowledge_gap_score ?? t.score), 0) / (b.length || 1);
              return avgB - avgA;
            })
            .map(([model, topics]) => (
              <ModelCard
                key={model}
                model={model}
                topics={topics}
                onClick={() => setSelected(model)}
              />
            ))}
        </div>
      )}
    </div>
  );
}
