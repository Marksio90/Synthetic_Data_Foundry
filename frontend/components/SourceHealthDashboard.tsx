'use client';

/**
 * SourceHealthDashboard — panel zdrowia wszystkich 46+ crawlerów + WebSub
 *
 * Pobiera GET /api/scout/sources co 30 sekund.
 * Crawlery są pogrupowane według warstw (A–E).
 * Sekcja WebSub pokazuje aktywne subskrypcje i statystyki dostaw.
 */

import { useCallback, useEffect, useState } from 'react';
import type { ElementType } from 'react';
import { getScoutSources } from '@/lib/api';
import type { CrawlerStatus, ScoutSourcesResponse, WebSubSubscription } from '@/lib/api';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Loader2,
  PauseCircle,
  Radio,
  RefreshCw,
  Server,
  XCircle,
} from 'lucide-react';

// ─── Przypisanie crawlerów do warstw ──────────────────────────────────────────

const LAYER_MAP: Record<string, string> = {
  // Warstwa A — nauka i badania
  arxiv: 'A', openalex: 'A', semanticscholar: 'A', pubmed: 'A',
  ieee: 'A', core: 'A', hackernews: 'A', biorxiv: 'A', medrxiv: 'A', ssrn: 'A',
  // Warstwa B — prawo i regulacje
  eurlex: 'B', curia: 'B', federalregister: 'B', secedgar: 'B',
  esma: 'B', eba: 'B', oecd: 'B', wto: 'B', wipo: 'B', epo: 'B',
  // Warstwa C — finanse i statystyki
  imf: 'C', worldbank: 'C', fred: 'C', ecb: 'C', bis: 'C',
  eurostat: 'C', undl: 'C', irena: 'C', owid: 'C', hdx: 'C',
  // Warstwa D — tech i społeczności
  github: 'D', reddit: 'D', stackexchange: 'D', producthunt: 'D',
  paperswithcode: 'D', mastodon: 'D',
  // Warstwa E — multimedia
  youtube: 'E', podcastindex: 'E', ted: 'E', archive: 'E', jstor: 'E', europeana: 'E',
};

const LAYER_LABELS: Record<string, string> = {
  A: 'Warstwa A — Nauka i badania',
  B: 'Warstwa B — Prawo i regulacje',
  C: 'Warstwa C — Finanse i statystyki',
  D: 'Warstwa D — Tech i społeczności',
  E: 'Warstwa E — Multimedia',
};

const LAYER_COLORS: Record<string, string> = {
  A: 'text-purple-400',
  B: 'text-blue-400',
  C: 'text-emerald-400',
  D: 'text-orange-400',
  E: 'text-pink-400',
};

function getLayer(sourceId: string): string {
  return LAYER_MAP[sourceId.toLowerCase()] ?? '?';
}

// ─── Status crawlera ─────────────────────────────────────────────────────────

type CrawlerHealth = 'ok' | 'degraded' | 'error' | 'paused';

function getCrawlerHealth(c: CrawlerStatus): CrawlerHealth {
  if (c.is_paused) return 'paused';
  if (c.consecutive_errors >= 5) return 'error';
  if (c.consecutive_errors >= 2) return 'degraded';
  return 'ok';
}

const HEALTH_CONFIG: Record<CrawlerHealth, { icon: ElementType; cls: string; label: string }> = {
  ok:       { icon: CheckCircle2, cls: 'text-success',      label: 'OK' },
  degraded: { icon: AlertCircle,  cls: 'text-warning',      label: 'Degradacja' },
  error:    { icon: XCircle,      cls: 'text-error',        label: 'Błąd' },
  paused:   { icon: PauseCircle,  cls: 'text-text-muted/50', label: 'Wstrzymany' },
};

// ─── Wiersz crawlera ─────────────────────────────────────────────────────────

function CrawlerRow({ crawler }: { crawler: CrawlerStatus }) {
  const health = getCrawlerHealth(crawler);
  const { icon: Icon, cls, label } = HEALTH_CONFIG[health];

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 hover:bg-bg-surface2 transition-colors rounded">
      <Icon className={`w-3.5 h-3.5 shrink-0 ${cls}`} />
      <span className="text-[12px] font-medium text-text w-28 truncate">
        {crawler.source_id}
      </span>
      <span className={`text-[10px] ${cls}`}>{label}</span>
      {crawler.consecutive_errors > 0 && (
        <span className="text-[10px] text-error/70 tabular-nums">
          {crawler.consecutive_errors} błąd{crawler.consecutive_errors === 1 ? '' : 'ów'}
        </span>
      )}
      <div className="flex-1" />
      <span className="text-[10px] text-text-muted/40 tabular-nums">
        co {crawler.poll_interval}s
      </span>
    </div>
  );
}

// ─── Sekcja warstwy ───────────────────────────────────────────────────────────

function LayerSection({
  layer,
  crawlers,
}: {
  layer: string;
  crawlers: CrawlerStatus[];
}) {
  const [expanded, setExpanded] = useState(true);
  const errorCount = crawlers.filter((c) => getCrawlerHealth(c) !== 'ok').length;
  const colorCls = LAYER_COLORS[layer] ?? 'text-text-muted';

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-bg-surface2 hover:bg-bg-surface2/80 transition-colors"
      >
        <Server className={`w-3.5 h-3.5 ${colorCls}`} />
        <span className={`text-[11px] font-semibold ${colorCls}`}>
          {LAYER_LABELS[layer] ?? `Warstwa ${layer}`}
        </span>
        <span className="text-[10px] text-text-muted ml-1">
          {crawlers.length} crawler{crawlers.length !== 1 ? 'ów' : ''}
        </span>
        {errorCount > 0 && (
          <span className="text-[10px] text-warning bg-warning/10 border border-warning/25 rounded-full px-1.5 py-0.5">
            {errorCount} ≠ OK
          </span>
        )}
        <div className="flex-1" />
        <span className="text-[10px] text-text-muted/40">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="divide-y divide-border/40">
          {crawlers.map((c) => (
            <CrawlerRow key={c.source_id} crawler={c} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Wiersz WebSub ─────────────────────────────────────────────────────────────

function WebSubRow({ sub }: { sub: WebSubSubscription }) {
  const expiresHours = Math.floor(sub.expires_in_s / 3600);
  const expiresMin = Math.floor((sub.expires_in_s % 3600) / 60);
  const isExpiring = sub.expires_in_s < 3600;

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 hover:bg-bg-surface2 transition-colors rounded">
      {sub.verified ? (
        <CheckCircle2 className="w-3.5 h-3.5 text-success shrink-0" />
      ) : (
        <Loader2 className="w-3.5 h-3.5 text-text-muted/50 animate-spin shrink-0" />
      )}
      <span className="text-[11px] font-medium text-text w-24 shrink-0">{sub.source_type}</span>
      <span className="text-[10px] text-text-muted truncate flex-1">
        {sub.topic_url.replace(/^https?:\/\//, '').slice(0, 55)}
      </span>
      <span className="text-[10px] text-text-muted/50 shrink-0">
        Tier {sub.tier}
      </span>
      <span className={`text-[10px] tabular-nums shrink-0 ${isExpiring ? 'text-warning' : 'text-text-muted/40'}`}>
        {expiresHours > 0 ? `${expiresHours}h` : `${expiresMin}min`}
      </span>
      <span className="text-[10px] tabular-nums text-accent/70 shrink-0">
        {sub.delivery_count} dostaw
      </span>
    </div>
  );
}

// ─── Główny komponent ──────────────────────────────────────────────────────────

const REFRESH_INTERVAL_MS = 30_000;

export default function SourceHealthDashboard({ className = '' }: { className?: string }) {
  const [data, setData] = useState<ScoutSourcesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    setError('');
    try {
      const res = await getScoutSources();
      setData(res);
      setLastRefresh(new Date());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd pobierania statusów crawlerów');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(() => load(true), REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [load]);

  // Grupuj crawlery według warstw
  const crawlersByLayer: Record<string, CrawlerStatus[]> = {};
  if (data) {
    for (const c of data.crawlers) {
      const layer = getLayer(c.source_id);
      (crawlersByLayer[layer] ??= []).push(c);
    }
  }

  const totalErrors = data?.crawlers.filter((c) => getCrawlerHealth(c) !== 'ok').length ?? 0;

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Nagłówek */}
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-accent" />
        <h2 className="text-sm font-semibold text-text">Zdrowie crawlerów</h2>
        <div className="flex-1" />
        {lastRefresh && (
          <span className="text-[10px] text-text-muted/50">
            Odświeżono: {lastRefresh.toLocaleTimeString('pl-PL')}
          </span>
        )}
        <button
          onClick={() => load()}
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

      {/* Kafelki sumaryczne */}
      {data && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Łącznie crawlerów', value: data.total_crawlers, cls: 'text-text' },
            { label: 'Aktywnych', value: data.active_crawlers, cls: 'text-success' },
            { label: 'Z problemami', value: totalErrors, cls: totalErrors > 0 ? 'text-warning' : 'text-text-muted' },
            { label: 'Subskrypcje WebSub', value: data.websub_stats.total_subscriptions, cls: 'text-accent' },
          ].map(({ label, value, cls }) => (
            <div key={label} className="rounded-lg border border-border bg-bg-surface p-3 text-center">
              <p className={`text-2xl font-bold tabular-nums ${cls}`}>{value}</p>
              <p className="text-[11px] text-text-muted mt-0.5">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Stan ładowania (pierwsza wizyta) */}
      {loading && !data && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 text-accent animate-spin" />
        </div>
      )}

      {/* Sekcje warstw A–E */}
      {data && (
        <div className="space-y-3">
          {['A', 'B', 'C', 'D', 'E'].map((layer) =>
            crawlersByLayer[layer]?.length ? (
              <LayerSection
                key={layer}
                layer={layer}
                crawlers={crawlersByLayer[layer]}
              />
            ) : null,
          )}

          {/* Nieprzypisane crawlery */}
          {crawlersByLayer['?']?.length > 0 && (
            <LayerSection layer="?" crawlers={crawlersByLayer['?']} />
          )}
        </div>
      )}

      {/* Sekcja WebSub */}
      {data && data.websub_subscriptions.length > 0 && (
        <div className="rounded-lg border border-border overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 bg-bg-surface2">
            <Radio className="w-3.5 h-3.5 text-accent" />
            <span className="text-[11px] font-semibold text-accent">WebSub — Tier 1 (real-time)</span>
            <span className="text-[10px] text-text-muted ml-1">
              {data.websub_stats.verified}/{data.websub_stats.total_subscriptions} zweryfikowanych
            </span>
            <div className="flex-1" />
            <span className="text-[10px] text-text-muted/50">
              {data.websub_stats.feeds} znanych feedów
            </span>
          </div>
          <div className="divide-y divide-border/40">
            {data.websub_subscriptions.map((sub, i) => (
              <WebSubRow key={i} sub={sub} />
            ))}
          </div>
        </div>
      )}

      {data && data.websub_subscriptions.length === 0 && (
        <div className="rounded-lg border border-border bg-bg-surface p-4 text-center">
          <Radio className="w-6 h-6 text-text-muted/30 mx-auto mb-2" />
          <p className="text-sm text-text-muted">WebSub Tier 1 wyłączony</p>
          <p className="text-[11px] text-text-muted/50 mt-0.5">
            Ustaw <code className="bg-bg-surface2 px-1 rounded">SCOUT_WEBHOOK_CALLBACK_URL</code> aby włączyć
          </p>
        </div>
      )}
    </div>
  );
}
