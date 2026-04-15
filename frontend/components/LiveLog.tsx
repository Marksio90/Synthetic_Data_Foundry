'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

interface LiveLogProps {
  runId: string;
  apiBase?: string;
  onStatusChange?: (status: string) => void;
}

export default function LiveLog({ runId, apiBase, onStatusChange }: LiveLogProps) {
  const [lines, setLines] = useState<string[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'polling' | 'disconnected'>('connecting');
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
      // Keep last 2000 lines to avoid memory issues
      return combined.length > 2000 ? combined.slice(combined.length - 2000) : combined;
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
    offsetRef.current = 0;
    setConnectionStatus('connecting');

    // Determine WebSocket URL
    const base = apiBase ?? '';
    let wsUrl: string;
    if (base.startsWith('http://') || base.startsWith('https://')) {
      wsUrl = base.replace(/^http/, 'ws') + `/api/pipeline/ws/${runId}`;
    } else {
      // Relative URL: use current host
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

  return (
    <div className="space-y-1">
      {/* Connection status bar */}
      <div className="flex items-center gap-2 text-xs text-text-muted">
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
        {connectionStatus === 'connected' && 'Połączono (WebSocket)'}
        {connectionStatus === 'polling' && 'Polling...'}
        {connectionStatus === 'connecting' && 'Łączenie...'}
        {connectionStatus === 'disconnected' && 'Rozłączono'}
      </div>

      {/* Log box */}
      <div
        ref={logRef}
        className="log-box bg-black rounded-md border border-border h-64 overflow-y-auto p-3 font-mono text-xs text-green-400 leading-5"
      >
        {lines.length === 0 ? (
          <span className="text-text-muted">Oczekiwanie na logi...</span>
        ) : (
          lines.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
