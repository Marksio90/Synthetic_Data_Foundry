'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  getScoutTopics,
  getWsBase,
  ingestTopic,
  startScoutRun,
  type ScoutSource,
  type ScoutTopic,
} from '@/lib/api';
import {
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  FileSearch,
  Loader2,
  MessageCircleQuestion,
  RefreshCw,
  Telescope,
  TrendingUp,
  Zap,
} from 'lucide-react';

// ─── Source-type helpers ──────────────────────────────────────────────────────

const SOURCE_LABELS: Record<string, string> = {
  arxiv: 'arXiv',
  openalex: 'OpenAlex',
  hackernews: 'HackerNews',
  eurlex: 'EUR-Lex',
};

const SOURCE_COLOURS: Record<string, string> = {
  arxiv: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
  openalex: 'bg-teal-500/15 text-teal-300 border-teal-500/30',
  hackernews: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
  eurlex: 'bg-blue-500/15 text-blue-300 border-blue-500/30',
};

function SourceBadge({ type }: { type: string }) {
  const cls = SOURCE_COLOURS[type] ?? 'bg-text-muted/15 text-text-muted border-border';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded border text-[10px] font-medium ${cls}`}>
      {SOURCE_LABELS[type] ?? type}
    </span>
  );
}

// ─── Score bar ────────────────────────────────────────────────────────────────

function ScoreBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  const colour =
    pct >= 70 ? 'bg-success' : pct >= 40 ? 'bg-warning' : 'bg-error';
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 text-text-muted shrink-0">{label}</span>
      <div className="flex-1 bg-bg-surface2 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${colour}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 text-right tabular-nums text-text-muted">{pct}%</span>
    </div>
  );
}

// ─── Topic card ───────────────────────────────────────────────────────────────

function TopicCard({ topic, onIngest }: { topic: ScoutTopic; onIngest: (id: string) => void }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestDone, setIngestDone] = useState(false);
  const [ingestMsg, setIngestMsg] = useState('');

  const score = Math.round(topic.score * 100);
  const scoreCls =
    score >= 70
      ? 'text-success border-success/40 bg-success/10'
      : score >= 40
      ? 'text-warning border-warning/40 bg-warning/10'
      : 'text-error border-error/40 bg-error/10';

  async function handleIngest() {
    setIngesting(true);
    try {
      const res = await ingestTopic(topic.topic_id);
      setIngestMsg(res.message);
      setIngestDone(true);
      onIngest(topic.topic_id);
      // Navigate to AutoPilot — downloaded files are ready
      setTimeout(() => router.push('/autopilot'), 1800);
    } catch (e: unknown) {
      setIngestMsg(e instanceof Error ? e.message : 'Błąd ingestion');
    } finally {
      setIngesting(false);
    }
  }

  return (
    <div className="rounded-lg border border-border bg-bg-surface p-4 flex flex-col gap-3">
      {/* Header row */}
      <div className="flex items-start gap-3">
        <span
          className={`shrink-0 w-10 h-10 rounded-lg border flex items-center justify-center text-sm font-bold tabular-nums ${scoreCls}`}
        >
          {score}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-text leading-snug line-clamp-2">{topic.title}</p>
          <p className="text-xs text-text-muted mt-0.5 line-clamp-2">{topic.summary}</p>
        </div>
      </div>

      {/* Score bars */}
      <div className="space-y-1.5">
        <ScoreBar value={topic.recency_score} label="Świeżość" />
        <ScoreBar value={topic.llm_uncertainty} label="Luka wiedzy" />
        <ScoreBar value={Math.min(1, topic.source_count / 10)} label="Źródła" />
      </div>

      {/* Source type pills */}
      <div className="flex flex-wrap gap-1">
        {Array.from(new Set(topic.sources.map((s) => s.source_type))).map((t) => (
          <SourceBadge key={t} type={t} />
        ))}
        <span className="text-[10px] text-text-muted ml-1 self-center">
          {topic.source_count} źródeł
        </span>
      </div>

      {/* Expand / actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setOpen((v) => !v)}
          className="text-xs text-accent hover:underline"
        >
          {open ? 'Zwiń' : 'Zobacz źródła'}
        </button>
        <div className="flex-1" />
        {ingestDone ? (
          <span className="flex items-center gap-1 text-xs text-success font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Przechodzę do AutoPilota…
          </span>
        ) : (
          <button
            onClick={handleIngest}
            disabled={ingesting}
            className="
              flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium
              bg-accent/10 text-accent border border-accent/30
              hover:bg-accent/20 disabled:opacity-50 disabled:cursor-not-allowed
              transition-colors
            "
          >
            {ingesting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
            Ingestuj → AutoPilot
          </button>
        )}
      </div>

      {ingestMsg && !ingestDone && (
        <p className="text-xs text-error bg-error/10 rounded px-2 py-1">{ingestMsg}</p>
      )}

      {/* Source list */}
      {open && (
        <ul className="space-y-1.5 border-t border-border pt-3">
          {topic.sources.slice(0, 8).map((src: ScoutSource, i: number) => (
            <li key={i} className="flex items-start gap-2 text-xs">
              <SourceBadge type={src.source_type} />
              <a
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-text-muted hover:text-accent flex-1 min-w-0 line-clamp-1 flex items-center gap-1"
              >
                <span className="truncate">{src.title || src.url}</span>
                <ArrowUpRight className="w-3 h-3 shrink-0" />
              </a>
              {src.published_at && (
                <span className="text-text-muted/60 shrink-0 tabular-nums">
                  {src.published_at.slice(0, 10)}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ─── Live log panel ───────────────────────────────────────────────────────────

function LiveLog({ scoutId, onDone }: { scoutId: string; onDone: () => void }) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState('running');
  const bottomRef = useRef<HTMLDivElement>(null);
  // Stable ref avoids re-creating the WebSocket when parent re-renders
  const onDoneRef = useRef(onDone);
  useEffect(() => { onDoneRef.current = onDone; });

  useEffect(() => {
    const ws = new WebSocket(`${getWsBase()}/api/scout/ws/${scoutId}`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.line && msg.line !== '__EOF__') {
        setLines((prev) => [...prev, msg.line]);
      }
      if (msg.status) setStatus(msg.status);
      if (msg.line === '__EOF__') {
        ws.close();
        onDoneRef.current();
      }
    };
    ws.onerror = () => setStatus('error');
    return () => ws.close();
  }, [scoutId]); // scoutId only — onDone stabilised via ref

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines]);

  return (
    <div className="rounded-lg border border-border bg-bg-surface overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-bg-surface2">
        {status === 'running' && <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />}
        {status === 'done' && <span className="w-3.5 h-3.5 rounded-full bg-success" />}
        {status === 'error' && <span className="w-3.5 h-3.5 rounded-full bg-error" />}
        <span className="text-xs font-medium text-text-muted">
          Skanowanie w toku — {status}
        </span>
      </div>
      <div className="h-48 overflow-y-auto font-mono text-[11px] text-text-muted p-3 space-y-0.5">
        {lines.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function ScoutPage() {
  const [topics, setTopics] = useState<ScoutTopic[]>([]);
  const [loading, setLoading] = useState(false);
  const [scoutId, setScoutId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<string>('all');

  const loadTopics = useCallback(async () => {
    try {
      const data = await getScoutTopics(50);
      setTopics(data);
    } catch {
      // Topics may be empty on first load — that's fine
    }
  }, []);

  useEffect(() => {
    loadTopics();
  }, [loadTopics]);

  async function handleRun() {
    setLoading(true);
    setError('');
    try {
      const run = await startScoutRun();
      setScoutId(run.scout_id);
      setTimeout(() => {
        setLoading((prev) => {
          if (prev) setError('Skanowanie zajęło zbyt długo — spróbuj ponownie');
          return false;
        });
      }, 5 * 60 * 1000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd uruchamiania skanu');
      setLoading(false);
    }
  }

  const handleRunDone = useCallback(() => {
    setLoading(false);
    loadTopics();
  }, [loadTopics]);

  const allTypes = useMemo(
    () => Array.from(new Set(topics.flatMap((t) => t.sources.map((s) => s.source_type)))).sort(),
    [topics],
  );

  const filtered = useMemo(
    () =>
      filter === 'all'
        ? topics
        : topics.filter((t) => t.sources.some((s) => s.source_type === filter)),
    [topics, filter],
  );

  const stats = useMemo(() => {
    if (!topics.length) return null;
    return {
      avgScore: Math.round((topics.reduce((a, t) => a + t.score, 0) / topics.length) * 100),
      avgUncertainty: Math.round((topics.reduce((a, t) => a + t.llm_uncertainty, 0) / topics.length) * 100),
      totalSources: topics.reduce((a, t) => a + t.source_count, 0),
    };
  }, [topics]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-border bg-bg-surface shrink-0">
        <Telescope className="w-5 h-5 text-accent" />
        <div>
          <h1 className="text-base font-semibold text-text">Gap Scout</h1>
          <p className="text-xs text-text-muted">
            Automatyczne wykrywanie luk wiedzy LLM — kliknij &quot;Ingestuj&quot; aby wysłać temat do AutoPilota
          </p>
        </div>
        <div className="flex-1" />
        <button
          onClick={handleRun}
          disabled={loading}
          className="
            flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
            bg-accent text-white hover:bg-accent/90
            disabled:opacity-60 disabled:cursor-not-allowed
            transition-colors
          "
        >
          {loading
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : <RefreshCw className="w-4 h-4" />
          }
          {loading ? 'Skanowanie...' : 'Uruchom skan'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {error && (
          <div className="rounded-lg bg-error/10 border border-error/30 text-error text-sm px-4 py-3">
            {error}
          </div>
        )}

        {/* Live log */}
        {scoutId && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-text-muted uppercase tracking-wider">
              Postęp skanowania
            </p>
            <LiveLog scoutId={scoutId} onDone={handleRunDone} />
          </div>
        )}

        {/* Stats row */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { icon: FileSearch, label: 'Tematów', value: topics.length },
              { icon: TrendingUp,             label: 'Śr. score',    value: `${stats.avgScore}%` },
              { icon: MessageCircleQuestion,  label: 'Śr. luka LLM', value: `${stats.avgUncertainty}%` },
              { icon: BookOpen,               label: 'Źródeł łącznie', value: stats.totalSources },
            ].map(({ icon: Icon, label, value }) => (
              <div
                key={label}
                className="rounded-lg border border-border bg-bg-surface px-4 py-3 flex items-center gap-3"
              >
                <Icon className="w-4 h-4 text-accent shrink-0" />
                <div>
                  <p className="text-lg font-semibold text-text tabular-nums">{value}</p>
                  <p className="text-xs text-text-muted">{label}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Filter tabs */}
        {allTypes.length > 0 && (
          <div className="flex gap-2 flex-wrap">
            {['all', ...allTypes].map((t) => (
              <button
                key={t}
                onClick={() => setFilter(t)}
                className={`
                  px-3 py-1 rounded-full text-xs font-medium border transition-colors
                  ${filter === t
                    ? 'bg-accent/15 text-accent border-accent/40'
                    : 'bg-bg-surface text-text-muted border-border hover:border-accent/30'}
                `}
              >
                {t === 'all' ? 'Wszystkie' : SOURCE_LABELS[t] ?? t}
              </button>
            ))}
          </div>
        )}

        {/* Topic grid */}
        {filtered.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filtered.map((topic) => (
              <TopicCard
                key={topic.topic_id}
                topic={topic}
                onIngest={loadTopics}
              />
            ))}
          </div>
        ) : !loading ? (
          <div className="flex flex-col items-center justify-center py-20 text-center gap-4">
            <Telescope className="w-12 h-12 text-text-muted/40" />
            <div>
              <p className="text-text font-medium">Brak wykrytych tematów</p>
              <p className="text-sm text-text-muted mt-1">
                Kliknij &quot;Uruchom skan&quot; aby odkryć luki wiedzy LLM
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
