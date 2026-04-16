'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import ProgressBar from './ProgressBar';

interface LiveLogProps {
  runId: string;
  apiBase?: string;
  onStatusChange?: (status: string) => void;
}

// Colour-code log lines by keywords so operators can scan at a glance
function classifyLine(line: string): string {
  const l = line.toLowerCase();
  if (l.includes('error') || l.includes('failed') || l.includes('błąd') || l.includes('exception'))
    return 'text-red-400';
  if (l.includes('warn') || l.includes('warning') || l.includes('uwaga') || l.includes('skip'))
    return 'text-yellow-400';
  if (
    l.includes('done') || l.includes('finished') || l.includes('gotow') ||
    l.includes('written') || l.includes('zapisano') || l.includes('complete') ||
    l.includes('sukces') || l.includes('ok ✓') || l.includes('✓')
  )
    return 'text-green-400';
  if (
    l.includes('phase') || l.includes('faza') || l.includes('chunk') ||
    l.includes('pipeline') || l.includes('start') || l.includes('running') ||
    l.includes('ingest') || l.includes('embedd')
  )
    return 'text-blue-300';
  if (l.includes('judge') || l.includes('score') || l.includes('quality') || l.includes('sędzia'))
    return 'text-purple-300';
  if (l.includes('[ws]'))
    return 'text-gray-500 italic';
  return 'text-green-400';
}

// Extract a progress-like summary line (chunk N/M, records, etc.)
function extractPhaseLabel(lines: string[]): string | null {
  for (let i = lines.length - 1; i >= Math.max(0, lines.length - 30); i--) {
    const l = lines[i];
    // "Chunk 12/80", "Przetworzono 12 chunków"
    const mChunk = l.match(/chunk[i\s]+(\d+)\s*\/\s*(\d+)/i);
    if (mChunk) return `Chunk ${mChunk[1]} / ${mChunk[2]}`;
    // "Phase 1: Ingest"
    const mPhase = l.match(/faz[ae]\s*(\d+)[:\s]+([^•\n]{3,40})/i) ||
                   l.match(/phase\s*(\d+)[:\s]+([^•\n]{3,40})/i);
    if (mPhase) return `Faza ${mPhase[1]}: ${mPhase[2].trim()}`;
    // "records_written: 45"
    const mRec = l.match(/records?[_\s]written[:\s]+(\d+)/i);
    if (mRec) return `Zapisano ${mRec[1]} rekordów`;
  }
  return null;
}

export default function LiveLog({ runId, apiBase, onStatusChange }: LiveLogProps) {
  const [lines, setLines] = useState<string[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'polling' | 'disconnected'>('connecting');
  const [phaseLabel, setPhaseLabel] = useState<string | null>(null);
  const [wsProgress, setWsProgress] = useState<{ pct: number; done: number; total: number } | null>(null);
  const [wsStatus, setWsStatus] = useState<string>('');
  const logRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const offsetRef = useRef(0);
  const mountedRef = useRef(true);

  const scrollToBottom = useCallback(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, []);

  const addLines = useCallback((newLines: string[]) => {
    if (!mountedRef.current) return;
    setLines((prev) => {
      const combined = [...prev, ...newLines];
      const trimmed = combined.length > 2000 ? combined.slice(combined.length - 2000) : combined;
      // Update phase label from latest lines
      const label = extractPhaseLabel(trimmed);
      if (label) setPhaseLabel(label);
      return trimmed;
    });
    setTimeout(scrollToBottom, 50);
  }, [scrollToBottom]);

  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;
    setConnectionStatus('polling');

    const poll = async () => {
      if (!mountedRef.current) return;
      try {
        const base = apiBase ?? '';
        const res = await fetch(`${base}/api/pipeline/log/${runId}?offset=${offsetRef.current}`);
        if (res.ok) {
          const data = await res.json();
          if (data.lines && data.lines.length > 0) {
            addLines(data.lines);
            offsetRef.current += data.lines.length;
          }
          if (data.status && onStatusChange) {
            onStatusChange(data.status);
          }
        }
      } catch {
        // ignore poll errors
      }
    };

    poll();
    pollIntervalRef.current = setInterval(poll, 2000);
  }, [runId, apiBase, addLines, onStatusChange]);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    setLines([]);
    setPhaseLabel(null);
    offsetRef.current = 0;
    setConnectionStatus('connecting');

    const base = apiBase ?? '';
    let wsUrl: string;
    if (base.startsWith('http://') || base.startsWith('https://')) {
      wsUrl = base.replace(/^http/, 'ws') + `/api/pipeline/ws/${runId}`;
    } else {
      const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = typeof window !== 'undefined' ? window.location.host : 'localhost:3000';
      wsUrl = `${protocol}//${host}/api/pipeline/ws/${runId}`;
    }

    let wsConnected = false;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      const connectTimeout = setTimeout(() => {
        if (!wsConnected) {
          ws.close();
          startPolling();
        }
      }, 5000);

      ws.onopen = () => {
        if (!mountedRef.current) return;
        wsConnected = true;
        clearTimeout(connectTimeout);
        setConnectionStatus('connected');
        addLines(['[WS] Połączono z serwerem logów...']);
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const data = JSON.parse(event.data);
          if (data.line) {
            addLines([data.line]);
          } else if (data.lines) {
            addLines(data.lines);
          } else if (typeof event.data === 'string') {
            addLines([event.data]);
          }
          if (data.status && onStatusChange) {
            onStatusChange(data.status);
          }
          // Track progress from WS payload
          if (data.progress_pct !== undefined && data.chunks_total > 0) {
            setWsProgress({
              pct: data.progress_pct,
              done: data.chunks_done ?? 0,
              total: data.chunks_total ?? 0,
            });
          }
          if (data.status) setWsStatus(data.status);
        } catch {
          if (typeof event.data === 'string') {
            addLines([event.data]);
          }
        }
      };

      ws.onerror = () => {
        clearTimeout(connectTimeout);
        if (!wsConnected && mountedRef.current) {
          startPolling();
        }
      };

      ws.onclose = () => {
        clearTimeout(connectTimeout);
        if (!mountedRef.current) return;
        if (wsConnected) {
          setConnectionStatus('disconnected');
          addLines(['[WS] Rozłączono.']);
        } else {
          startPolling();
        }
      };
    } catch {
      startPolling();
    }

    return () => {
      mountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      stopPolling();
    };
  }, [runId, apiBase, addLines, startPolling, stopPolling, onStatusChange]);

  const barStatus = wsStatus === 'done' ? 'done' : wsStatus === 'error' ? 'error' : 'running';

  return (
    <div className="space-y-2">
      {/* Header row: connection status + live phase label */}
      <div className="flex items-center justify-between text-xs text-text-muted">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${
              connectionStatus === 'connected'
                ? 'bg-success'
                : connectionStatus === 'polling'
                ? 'bg-warning animate-pulse'
                : connectionStatus === 'connecting'
                ? 'bg-accent animate-pulse'
                : 'bg-text-muted'
            }`}
          />
          {connectionStatus === 'connected' && 'WebSocket — na żywo'}
          {connectionStatus === 'polling' && 'Polling co 2s...'}
          {connectionStatus === 'connecting' && 'Łączenie...'}
          {connectionStatus === 'disconnected' && 'Rozłączono'}
        </div>
        {phaseLabel && (
          <span className="text-blue-300 font-mono truncate max-w-[60%]">{phaseLabel}</span>
        )}
        <span className="text-text-muted">{lines.length} linii</span>
      </div>

      {/* Inline mini progress bar (populated from WS chunk data) */}
      {wsProgress && (
        <ProgressBar
          value={wsProgress.pct}
          max={100}
          label={`Chunk ${wsProgress.done} / ${wsProgress.total}`}
          valueLabel={`${wsProgress.pct.toFixed(1)}%`}
          status={barStatus}
          size="xs"
        />
      )}

      {/* Log terminal — taller, colour-coded lines */}
      <div
        ref={logRef}
        className="log-box bg-[#0a0e0a] rounded-md border border-border h-96 overflow-y-auto p-3 font-mono text-xs leading-5 select-text"
      >
        {lines.length === 0 ? (
          <span className="text-text-muted animate-pulse">Oczekiwanie na logi pipeline...</span>
        ) : (
          lines.map((line, i) => (
            <div key={i} className={`whitespace-pre-wrap break-all ${classifyLine(line)}`}>
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
