'use client';

/**
 * scout/page.tsx — Strona Gap Scout
 *
 * Zakładki:
 *   Tematy     — siatka odkrytych luk wiedzy + ingestowanie
 *   Na żywo    — SSE strumień nowych tematów w czasie rzeczywistym
 *   Modele     — luki wiedzy pogrupowane per model LLM
 *   Źródła     — panel zdrowia 46+ crawlerów + WebSub
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ElementType } from 'react';
import { useRouter } from 'next/navigation';
import {
  getScoutTopics,
  getWsBase,
  ingestTopic,
  startScoutRun,
  submitFeedback,
  type ScoutSource,
  type ScoutTopic,
} from '@/lib/api';
import LiveSourceStream from '@/components/LiveSourceStream';
import ModelGapChart from '@/components/ModelGapChart';
import SourceHealthDashboard from '@/components/SourceHealthDashboard';
import {
  Activity,
  ArrowUpRight,
  Brain,
  CheckCircle2,
  Loader2,
  Radio,
  RefreshCw,
  Server,
  Telescope,
  ThumbsDown,
  ThumbsUp,
  Zap,
} from 'lucide-react';

// ─── Pomocnicze ───────────────────────────────────────────────────────────────

const SOURCE_LABELS: Record<string, string> = {
  arxiv: 'arXiv', openalex: 'OpenAlex', hackernews: 'HN', eurlex: 'EUR-Lex',
  curia: 'Curia', federalregister: 'Fed.Reg', secedgar: 'SEC', esma: 'ESMA',
  eba: 'EBA', oecd: 'OECD', wto: 'WTO', wipo: 'WIPO', epo: 'EPO',
  imf: 'IMF', worldbank: 'WorldBank', fred: 'FRED', ecb: 'ECB', bis: 'BIS',
  eurostat: 'Eurostat', irena: 'IRENA', owid: 'OurWorldInData', hdx: 'HDX',
  github: 'GitHub', reddit: 'Reddit', stackexchange: 'StackEx',
  producthunt: 'PH', paperswithcode: 'PwC', mastodon: 'Mastodon',
  youtube: 'YouTube', podcastindex: 'Podcast', ted: 'TED',
  archive: 'Archive', jstor: 'JSTOR', europeana: 'Europeana',
};

const SOURCE_COLOURS: Record<string, string> = {
  arxiv: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
  openalex: 'bg-teal-500/15 text-teal-300 border-teal-500/30',
  hackernews: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
  eurlex: 'bg-blue-500/15 text-blue-300 border-blue-500/30',
  github: 'bg-gray-500/15 text-gray-300 border-gray-500/30',
  reddit: 'bg-red-500/15 text-red-300 border-red-500/30',
  youtube: 'bg-red-600/15 text-red-400 border-red-600/30',
};

function SourceBadge({ type }: { type: string }) {
  const cls = SOURCE_COLOURS[type] ?? 'bg-text-muted/15 text-text-muted border-border';
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded border text-[10px] font-medium ${cls}`}>
      {SOURCE_LABELS[type] ?? type}
    </span>
  );
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(value * 100);
  const colour = pct >= 70 ? 'bg-success' : pct >= 40 ? 'bg-warning' : 'bg-error';
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-20 text-text-muted shrink-0">{label}</span>
      <div className="flex-1 bg-bg-surface2 rounded-full h-1 overflow-hidden">
        <div className={`h-full rounded-full ${colour} transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-7 text-right tabular-nums text-text-muted">{pct}%</span>
    </div>
  );
}

// ─── Panel oceny ─────────────────────────────────────────────────────────────

function FeedbackPanel({ topicId }: { topicId: string }) {
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);
  const [avgRating, setAvgRating] = useState<number | null>(null);

  async function vote(helpful: boolean) {
    setLoading(true);
    try {
      const res = await submitFeedback(topicId, helpful ? 5 : 2, helpful);
      setAvgRating(res.avg_rating);
      setDone(true);
    } catch {
      /* cicha obsługa błędu */
    } finally {
      setLoading(false);
    }
  }

  if (done) {
    return (
      <span className="text-[10px] text-text-muted/60 flex items-center gap-1">
        <CheckCircle2 className="w-3 h-3 text-success" />
        Dziękujemy! Śr. ocena: {avgRating?.toFixed(1)}
      </span>
    );
  }

  return (
    <div className="flex items-center gap-1">
      <span className="text-[10px] text-text-muted/60">Ocena:</span>
      <button
        onClick={() => vote(true)}
        disabled={loading}
        className="p-0.5 rounded hover:text-success transition-colors text-text-muted/40 disabled:opacity-50"
        title="Przydatny temat"
      >
        <ThumbsUp className="w-3.5 h-3.5" />
      </button>
      <button
        onClick={() => vote(false)}
        disabled={loading}
        className="p-0.5 rounded hover:text-error transition-colors text-text-muted/40 disabled:opacity-50"
        title="Nieprzydatny temat"
      >
        <ThumbsDown className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ─── Karta tematu ─────────────────────────────────────────────────────────────

function TopicCard({
  topic,
  isNew,
  onIngest,
}: {
  topic: ScoutTopic;
  isNew: boolean;
  onIngest: () => void;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestDone, setIngestDone] = useState(false);
  const [ingestErr, setIngestErr] = useState('');

  const gapScore = topic.knowledge_gap_score ?? topic.score;
  const displayScore = Math.round(gapScore * 100);
  const scoreCls =
    displayScore >= 70
      ? 'text-success border-success/40 bg-success/10'
      : displayScore >= 40
      ? 'text-warning border-warning/40 bg-warning/10'
      : 'text-error border-error/40 bg-error/10';

  async function handleIngest() {
    setIngesting(true);
    setIngestErr('');
    try {
      await ingestTopic(topic.topic_id);
      setIngestDone(true);
      onIngest();
      setTimeout(() => router.push('/autopilot'), 1600);
    } catch (e: unknown) {
      setIngestErr(e instanceof Error ? e.message : 'Błąd ingestion');
    } finally {
      setIngesting(false);
    }
  }

  const sourceTypes = Array.from(new Set(topic.sources.map((s) => s.source_type)));

  return (
    <div className={`rounded-lg border border-border bg-bg-surface p-4 flex flex-col gap-2.5 ${isNew ? 'topic-appear' : ''}`}>
      <div className="flex items-start gap-3">
        <span className={`shrink-0 w-9 h-9 rounded-md border flex items-center justify-center text-xs font-bold tabular-nums ${scoreCls}`}>
          {displayScore}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-text leading-snug line-clamp-2">{topic.title}</p>
          <p className="text-[11px] text-text-muted mt-0.5 line-clamp-1">{topic.summary}</p>
        </div>
      </div>

      <div className="space-y-1">
        <ScoreBar value={topic.recency_score} label="Świeżość" />
        <ScoreBar value={topic.llm_uncertainty} label="Luka LLM" />
        <ScoreBar value={Math.min(1, topic.source_count / 10)} label="Źródła" />
        {(topic.citation_velocity ?? 0) > 0 && (
          <ScoreBar value={Math.min(1, (topic.citation_velocity ?? 0) / 5)} label="Dynamika" />
        )}
      </div>

      {/* Etykiety źródeł */}
      <div className="flex items-center gap-1 flex-wrap">
        {sourceTypes.slice(0, 4).map((t) => <SourceBadge key={t} type={t} />)}
        {sourceTypes.length > 4 && (
          <span className="text-[10px] text-text-muted">+{sourceTypes.length - 4}</span>
        )}
        <span className="text-[10px] text-text-muted ml-0.5">{topic.source_count} src</span>
        {topic.source_tier && (
          <span className="text-[10px] text-text-muted/50 ml-0.5">Tier {topic.source_tier}</span>
        )}
      </div>

      {/* Modele docelowe */}
      {topic.cutoff_model_targets && topic.cutoff_model_targets.length > 0 && (
        <div className="flex items-center gap-1 flex-wrap">
          {topic.cutoff_model_targets.slice(0, 3).map((m) => (
            <span key={m} className="text-[9px] px-1.5 py-0.5 rounded-full border border-accent/25 bg-accent/5 text-accent/70">
              {m}
            </span>
          ))}
        </div>
      )}

      {/* Akcje */}
      <div className="flex items-center gap-2 pt-0.5">
        <button onClick={() => setOpen((v) => !v)} className="text-[11px] text-accent hover:underline">
          {open ? 'Zwiń' : 'Źródła'}
        </button>
        <FeedbackPanel topicId={topic.topic_id} />
        <div className="flex-1" />
        {ingestDone ? (
          <span className="flex items-center gap-1 text-[11px] text-success font-medium">
            <CheckCircle2 className="w-3 h-3" /> AutoPilot…
          </span>
        ) : (
          <button
            onClick={handleIngest}
            disabled={ingesting}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium
              bg-accent/10 text-accent border border-accent/30
              hover:bg-accent/20 disabled:opacity-50 transition-colors"
          >
            {ingesting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
            Ingestuj
          </button>
        )}
      </div>

      {ingestErr && (
        <p className="text-[11px] text-error bg-error/10 rounded px-2 py-1">{ingestErr}</p>
      )}

      {open && (
        <ul className="space-y-1.5 border-t border-border pt-2.5">
          {topic.sources.slice(0, 6).map((src: ScoutSource, i: number) => (
            <li key={i} className="flex items-start gap-1.5 text-[11px]">
              <SourceBadge type={src.source_type} />
              <a
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-text-muted hover:text-accent flex-1 min-w-0 flex items-center gap-0.5 line-clamp-1"
              >
                <span className="truncate">{src.title || src.url}</span>
                <ArrowUpRight className="w-2.5 h-2.5 shrink-0" />
              </a>
              {src.published_at && (
                <span className="text-text-muted/50 shrink-0 tabular-nums">
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

// ─── Panel postępu skanowania (WebSocket log) ─────────────────────────────────

function LiveLog({ scoutId, onDone }: { scoutId: string; onDone: () => void }) {
  const [lines, setLines] = useState<string[]>([]);
  const [status, setStatus] = useState<string>('running');
  const bottomRef = useRef<HTMLDivElement>(null);
  const onDoneRef = useRef(onDone);
  useEffect(() => { onDoneRef.current = onDone; });

  useEffect(() => {
    const ws = new WebSocket(`${getWsBase()}/api/scout/ws/${scoutId}`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data as string) as { line?: string; status?: string };
      if (msg.line && msg.line !== '__EOF__') setLines((prev) => [...prev, msg.line!]);
      if (msg.status) setStatus(msg.status);
      if (msg.line === '__EOF__') { ws.close(); onDoneRef.current(); }
    };
    ws.onerror = () => setStatus('error');
    return () => ws.close();
  }, [scoutId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines]);

  return (
    <div className="flex flex-col rounded-lg border border-border bg-bg-surface overflow-hidden flex-1 min-h-0">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-bg-surface2 shrink-0">
        {status === 'running' && <Loader2 className="w-3 h-3 animate-spin text-accent" />}
        {status === 'done' && <span className="w-2.5 h-2.5 rounded-full bg-success" />}
        {status === 'error' && <span className="w-2.5 h-2.5 rounded-full bg-error" />}
        <span className="text-[11px] font-medium text-text-muted tracking-wide uppercase">
          Live scan — {status}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto font-mono text-[10px] text-text-muted p-3 space-y-0.5 min-h-0">
        {lines.map((l, i) => (
          <div key={i} className={l.includes('🎯') ? 'text-accent' : l.includes('❌') ? 'text-error' : ''}>
            {l}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ─── Typy zakładek ─────────────────────────────────────────────────────────────

type Tab = 'tematy' | 'live' | 'modele' | 'zrodla';

const TABS: { id: Tab; label: string; icon: ElementType }[] = [
  { id: 'tematy',  label: 'Tematy',    icon: Telescope },
  { id: 'live',    label: 'Na żywo',   icon: Radio },
  { id: 'modele',  label: 'Modele',    icon: Brain },
  { id: 'zrodla',  label: 'Źródła',    icon: Server },
];

// ─── Strona główna ─────────────────────────────────────────────────────────────

export default function ScoutPage() {
  const [activeTab, setActiveTab] = useState<Tab>('tematy');

  // Stan tematów
  const [topics, setTopics] = useState<ScoutTopic[]>([]);
  const [newTopicIds, setNewTopicIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [scoutId, setScoutId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<string>('all');
  const knownIdsRef = useRef<Set<string>>(new Set());

  const loadTopics = useCallback(async (markNew = false) => {
    try {
      const data = await getScoutTopics(50);
      if (markNew) {
        const fresh = data.filter((t) => !knownIdsRef.current.has(t.topic_id));
        if (fresh.length > 0) {
          const freshIds = new Set(fresh.map((t) => t.topic_id));
          setNewTopicIds((prev) => new Set([...Array.from(prev), ...Array.from(freshIds)]));
          fresh.forEach((t) => knownIdsRef.current.add(t.topic_id));
          setTimeout(() => {
            setNewTopicIds((prev) => {
              const next = new Set(prev);
              freshIds.forEach((id) => next.delete(id));
              return next;
            });
          }, 600);
        }
      } else {
        data.forEach((t) => knownIdsRef.current.add(t.topic_id));
      }
      setTopics(data);
    } catch { /* cicha obsługa błędu */ }
  }, []);

  useEffect(() => { loadTopics(false); }, [loadTopics]);

  // Odpytuj tematy co 2.5s podczas aktywnego skanu
  useEffect(() => {
    if (!loading || !scoutId) return;
    const interval = setInterval(() => loadTopics(true), 2500);
    return () => clearInterval(interval);
  }, [loading, scoutId, loadTopics]);

  async function handleRun() {
    setLoading(true);
    setError('');
    try {
      const run = await startScoutRun();
      setScoutId(run.scout_id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd uruchamiania skanu');
      setLoading(false);
    }
  }

  const handleRunDone = useCallback(() => {
    setLoading(false);
    loadTopics(true);
  }, [loadTopics]);

  const allTypes = useMemo(
    () => Array.from(new Set(topics.flatMap((t) => t.sources.map((s) => s.source_type)))).sort(),
    [topics],
  );

  const filtered = useMemo(
    () => filter === 'all' ? topics : topics.filter((t) => t.sources.some((s) => s.source_type === filter)),
    [topics, filter],
  );

  const handleIngest = useCallback(() => { loadTopics(false); }, [loadTopics]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Nagłówek ── */}
      <div className="flex items-center gap-3 px-5 py-3 border-b border-border bg-bg-surface shrink-0">
        <div className="relative">
          <Telescope className={`w-5 h-5 ${loading ? 'text-accent radar-pulse' : 'text-accent'}`} />
          {loading && (
            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-accent animate-ping" />
          )}
        </div>
        <div className="flex-1">
          <h1 className="text-sm font-semibold text-text flex items-center gap-2">
            Gap Scout
            {loading && (
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-accent bg-accent/10 border border-accent/25 rounded-full px-2 py-0.5">
                <Radio className="w-2.5 h-2.5" /> LIVE
              </span>
            )}
          </h1>
          <p className="text-[11px] text-text-muted">
            {loading
              ? `Skanowanie — ${topics.length} ${topics.length === 1 ? 'temat' : topics.length < 5 ? 'tematy' : 'tematów'} odkrytych…`
              : 'Wykrywanie luk wiedzy LLM · kliknij Ingestuj → AutoPilot'}
          </p>
        </div>
        <button
          onClick={handleRun}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
            bg-accent text-bg-base hover:bg-accent/90
            disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
          {loading ? 'Skanowanie…' : 'Uruchom skan'}
        </button>
      </div>

      {/* ── Pasek zakładek ── */}
      <div className="flex items-center gap-0.5 px-5 py-0 border-b border-border bg-bg-surface shrink-0">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`
              flex items-center gap-1.5 px-3 py-2.5 text-[12px] font-medium border-b-2 transition-colors
              ${activeTab === id
                ? 'border-accent text-accent'
                : 'border-transparent text-text-muted hover:text-text'}
            `}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
            {id === 'tematy' && topics.length > 0 && (
              <span className="text-[10px] tabular-nums text-text-muted/60">({topics.length})</span>
            )}
          </button>
        ))}
      </div>

      {/* ── Treść zakładki ── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* ── ZAKŁADKA: Tematy ── */}
        {activeTab === 'tematy' && (
          <>
            {/* Lewy panel — postęp skanowania */}
            {scoutId && (
              <div className="w-72 shrink-0 flex flex-col border-r border-border p-3 min-h-0">
                <p className="text-[10px] font-semibold text-text-muted uppercase tracking-widest mb-2 shrink-0">
                  Postęp skanowania
                </p>
                <LiveLog scoutId={scoutId} onDone={handleRunDone} />
              </div>
            )}

            {/* Prawy panel — tematy */}
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              {error && (
                <div className="mx-4 mt-4 rounded-lg bg-error/10 border border-error/30 text-error text-sm px-4 py-3 shrink-0">
                  {error}
                </div>
              )}

              {/* Filtry zakładek */}
              {allTypes.length > 0 && (
                <div className="flex gap-1.5 flex-wrap px-4 pt-3 pb-2 shrink-0">
                  {['all', ...allTypes].map((t) => (
                    <button
                      key={t}
                      onClick={() => setFilter(t)}
                      className={`px-2.5 py-0.5 rounded-full text-[11px] font-medium border transition-colors
                        ${filter === t
                          ? 'bg-accent/15 text-accent border-accent/40'
                          : 'bg-bg-surface text-text-muted border-border hover:border-accent/30'}`}
                    >
                      {t === 'all' ? `Wszystkie (${topics.length})` : SOURCE_LABELS[t] ?? t}
                    </button>
                  ))}
                </div>
              )}

              {/* Siatka tematów */}
              <div className="flex-1 overflow-y-auto px-4 pb-4">
                {filtered.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 pt-2">
                    {filtered.map((topic) => (
                      <TopicCard
                        key={topic.topic_id}
                        topic={topic}
                        isNew={newTopicIds.has(topic.topic_id)}
                        onIngest={handleIngest}
                      />
                    ))}
                  </div>
                ) : !loading ? (
                  <div className="flex flex-col items-center justify-center py-24 text-center gap-4">
                    <Telescope className="w-14 h-14 text-text-muted/20" />
                    <div>
                      <p className="text-text font-medium">Brak wykrytych tematów</p>
                      <p className="text-sm text-text-muted mt-1">
                        Kliknij <strong>Uruchom skan</strong> — AI przeskanuje 46 źródeł w 5 warstwach
                      </p>
                    </div>
                    <button
                      onClick={handleRun}
                      className="mt-2 flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium
                        bg-accent text-bg-base hover:bg-accent/90 transition-colors"
                    >
                      <Telescope className="w-4 h-4" />
                      Uruchom pierwszy skan
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-24 gap-3">
                    <Loader2 className="w-8 h-8 text-accent animate-spin" />
                    <p className="text-sm text-text-muted">Odkrywanie tematów w czasie rzeczywistym…</p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ── ZAKŁADKA: Na żywo ── */}
        {activeTab === 'live' && (
          <div className="flex-1 flex flex-col min-h-0 p-4 overflow-hidden">
            <div className="mb-3 shrink-0">
              <p className="text-[11px] text-text-muted">
                Tematy odkrywane przez Gap Scout pojawiają się tutaj natychmiast — bez konieczności odświeżania.
                Strumień SSE odtwarza 20 ostatnich tematów przy połączeniu.
              </p>
            </div>
            <LiveSourceStream className="flex-1 min-h-0" />
          </div>
        )}

        {/* ── ZAKŁADKA: Modele ── */}
        {activeTab === 'modele' && (
          <div className="flex-1 overflow-y-auto p-4 min-h-0">
            <div className="mb-3">
              <p className="text-[11px] text-text-muted">
                Tematy pogrupowane według docelowych modeli LLM na podstawie dat granicnych danych treningowych.
                Kliknij kartę modelu, aby zobaczyć szczegóły.
              </p>
            </div>
            <ModelGapChart />
          </div>
        )}

        {/* ── ZAKŁADKA: Źródła ── */}
        {activeTab === 'zrodla' && (
          <div className="flex-1 overflow-y-auto p-4 min-h-0">
            <div className="mb-3">
              <p className="text-[11px] text-text-muted">
                Stan wszystkich crawlerów (warstwy A–E) oraz subskrypcji WebSub Tier 1.
                Odświeżany automatycznie co 30 sekund.
              </p>
            </div>
            <SourceHealthDashboard />
          </div>
        )}
      </div>
    </div>
  );
}
