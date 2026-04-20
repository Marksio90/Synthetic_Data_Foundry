'use client';

/**
 * LiveSourceStream — komponent SSE podłączony do GET /api/scout/live
 *
 * Wyświetla tematy odkryte w czasie rzeczywistym przez Gap Scout.
 * Automatycznie ponawia połączenie po rozłączeniu (max 5 prób).
 * Przy połączeniu odtwarza 20 ostatnich tematów jako bufor historyczny.
 */

import { useEffect, useRef, useState } from 'react';
import { getSseUrl } from '@/lib/api';
import type { ScoutTopic } from '@/lib/api';
import { Activity, ArrowUpRight, Radio, WifiOff } from 'lucide-react';

// ─── Typy lokalne ─────────────────────────────────────────────────────────────

interface LiveEvent {
  id: string;
  kind: 'topic' | 'heartbeat' | 'replay';
  topic?: ScoutTopic;
  ts: number;
}

// ─── Pomocnicze ───────────────────────────────────────────────────────────────

function gapColor(score: number): string {
  if (score >= 0.70) return 'text-success';
  if (score >= 0.40) return 'text-warning';
  return 'text-text-muted';
}

function relTime(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return `${s}s temu`;
  if (s < 3600) return `${Math.floor(s / 60)}min temu`;
  return `${Math.floor(s / 3600)}h temu`;
}

// ─── Mini-karta tematu ────────────────────────────────────────────────────────

function EventRow({ ev }: { ev: LiveEvent }) {
  if (ev.kind === 'heartbeat') {
    return (
      <div className="flex items-center gap-2 px-3 py-1 text-[10px] text-text-muted/40 select-none">
        <span className="w-1.5 h-1.5 rounded-full bg-text-muted/20" />
        <span>heartbeat</span>
        <span className="ml-auto tabular-nums">{relTime(ev.ts)}</span>
      </div>
    );
  }

  if (!ev.topic) return null;
  const t = ev.topic;
  const gapScore = t.knowledge_gap_score ?? t.score;
  const pct = Math.round(gapScore * 100);
  const sourceTypes = Array.from(new Set(t.sources.map((s) => s.source_type)));

  return (
    <div className={`px-3 py-2.5 border-b border-border last:border-0 ${ev.kind === 'replay' ? 'opacity-50' : 'event-appear'}`}>
      <div className="flex items-start gap-2">
        <span className={`shrink-0 tabular-nums text-[11px] font-bold w-7 text-right ${gapColor(gapScore)}`}>
          {pct}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-[12px] font-medium text-text leading-snug line-clamp-2">{t.title}</p>
          <div className="flex items-center gap-1.5 mt-1 flex-wrap">
            {sourceTypes.slice(0, 3).map((st) => (
              <span
                key={st}
                className="text-[9px] px-1 py-0.5 rounded bg-bg-surface2 text-text-muted border border-border"
              >
                {st}
              </span>
            ))}
            <span className="text-[10px] text-text-muted ml-auto tabular-nums">{relTime(ev.ts)}</span>
          </div>
        </div>
        {t.sources[0]?.url && (
          <a
            href={t.sources[0].url}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 text-text-muted/40 hover:text-accent transition-colors"
          >
            <ArrowUpRight className="w-3.5 h-3.5" />
          </a>
        )}
      </div>
    </div>
  );
}

// ─── Główny komponent ─────────────────────────────────────────────────────────

const MAX_EVENTS = 100;
const RECONNECT_DELAY_MS = [1000, 2000, 4000, 8000, 16000];

export default function LiveSourceStream({ className = '' }: { className?: string }) {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [topicCount, setTopicCount] = useState(0);

  const esRef = useRef<EventSource | null>(null);
  const attemptsRef = useRef(0);
  const mountedRef = useRef(true);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    mountedRef.current = true;

    function connect(attempt: number) {
      if (!mountedRef.current) return;

      const es = new EventSource(getSseUrl());
      esRef.current = es;
      let isFirstBatch = true;

      es.onopen = () => {
        if (!mountedRef.current) return;
        attemptsRef.current = 0;
        setConnected(true);
        setReconnectAttempt(0);
      };

      es.onmessage = (e: MessageEvent) => {
        if (!mountedRef.current) return;
        try {
          const payload = JSON.parse(e.data as string) as {
            event: string;
            data?: ScoutTopic;
          };

          if (payload.event === 'replay_end') {
            isFirstBatch = false;
            return;
          }

          if (payload.event === 'heartbeat') {
            setEvents((prev) =>
              [{
                id: `hb-${Date.now()}`,
                kind: 'heartbeat' as const,
                ts: Date.now(),
              }, ...prev].slice(0, MAX_EVENTS),
            );
            return;
          }

          if (payload.event === 'topic' && payload.data) {
            const kind = isFirstBatch ? ('replay' as const) : ('topic' as const);
            setEvents((prev) =>
              [{
                id: `t-${payload.data!.topic_id}-${Date.now()}`,
                kind,
                topic: payload.data,
                ts: Date.now(),
              }, ...prev].slice(0, MAX_EVENTS),
            );
            if (!isFirstBatch) {
              setTopicCount((c) => c + 1);
            }
          }
        } catch {
          /* ignoruj błędy parsowania */
        }
      };

      es.onerror = () => {
        es.close();
        esRef.current = null;
        if (!mountedRef.current) return;
        setConnected(false);
        const next = Math.min(attempt, RECONNECT_DELAY_MS.length - 1);
        setReconnectAttempt(attempt + 1);
        setTimeout(() => connect(attempt + 1), RECONNECT_DELAY_MS[next]);
      };
    }

    connect(0);

    return () => {
      mountedRef.current = false;
      esRef.current?.close();
    };
  }, []);

  const topics = events.filter((e) => e.kind !== 'heartbeat' && e.topic);
  const heartbeats = events.filter((e) => e.kind === 'heartbeat');
  const showHeartbeats = events.length > 0 && topics.length === 0;

  return (
    <div className={`flex flex-col rounded-lg border border-border bg-bg-surface overflow-hidden ${className}`}>
      {/* Nagłówek */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-bg-surface2 shrink-0">
        <div className="relative flex items-center">
          {connected ? (
            <>
              <Radio className="w-3.5 h-3.5 text-success" />
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-success animate-ping" />
            </>
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-error" />
          )}
        </div>
        <span className="text-[11px] font-semibold text-text-muted uppercase tracking-wide">
          Strumień na żywo
        </span>
        {connected && topicCount > 0 && (
          <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded-full bg-accent/15 text-accent border border-accent/25 tabular-nums">
            +{topicCount} nowych
          </span>
        )}
        <div className="flex-1" />
        <Activity className="w-3 h-3 text-text-muted/40" />
        {!connected && reconnectAttempt > 0 && (
          <span className="text-[10px] text-warning">
            Ponawiam #{reconnectAttempt}…
          </span>
        )}
      </div>

      {/* Lista zdarzeń */}
      <div ref={listRef} className="flex-1 overflow-y-auto min-h-0">
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
            <Radio className="w-8 h-8 text-text-muted/20" />
            <div>
              <p className="text-sm font-medium text-text-muted">
                {connected ? 'Oczekuję na tematy…' : 'Łączenie ze strumieniem…'}
              </p>
              <p className="text-[11px] text-text-muted/50 mt-0.5">
                Uruchom skan — nowe tematy pojawią się tutaj automatycznie
              </p>
            </div>
          </div>
        ) : (
          <div>
            {(showHeartbeats ? heartbeats.slice(0, 5) : topics).map((ev) => (
              <EventRow key={ev.id} ev={ev} />
            ))}
          </div>
        )}
      </div>

      {/* Stopka */}
      {events.length > 0 && (
        <div className="px-3 py-1.5 border-t border-border bg-bg-surface2 shrink-0 flex items-center gap-2">
          <span className="text-[10px] text-text-muted/50 tabular-nums">
            {topics.length} temat{topics.length === 1 ? '' : topics.length < 5 ? 'y' : 'ów'}
            {' · '}
            {heartbeats.length} ping{heartbeats.length !== 1 ? 'ów' : ''}
          </span>
          <button
            onClick={() => { setEvents([]); setTopicCount(0); }}
            className="ml-auto text-[10px] text-text-muted/40 hover:text-text-muted transition-colors"
          >
            Wyczyść
          </button>
        </div>
      )}
    </div>
  );
}
