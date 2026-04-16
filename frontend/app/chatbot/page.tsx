'use client';

import { useCallback, useEffect, useRef, useState, KeyboardEvent } from 'react';
import { MessageSquare, Send, Trash2, RefreshCw } from 'lucide-react';
import type { ChatMessage, OllamaModel } from '@/lib/api';
import ProgressBar from '@/components/ProgressBar';

function parseEvalProgress(logs: string[], total: number): { done: number; pct: number } | null {
  for (let i = logs.length - 1; i >= Math.max(0, logs.length - 30); i--) {
    const line = logs[i];
    // "Sample 5/10", "Próbka 5 z 10", "5/10"
    const m = line.match(/\b(\d+)\s*[\/z]\s*(\d+)\b/);
    if (m) {
      const done = parseInt(m[1]);
      const max = parseInt(m[2]);
      if (done <= max && max > 0)
        return { done, pct: (done / max) * 100 };
    }
    // "Evaluated: 5" or "evaluated 5"
    const m2 = line.match(/evaluat(?:ed|ing)[:\s]+(\d+)/i);
    if (m2) {
      const done = parseInt(m2[1]);
      if (total > 0)
        return { done, pct: (done / total) * 100 };
    }
  }
  return null;
}

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === 'system') {
    return (
      <div className="text-center my-2">
        <span className="text-xs text-text-muted bg-bg-surface2 px-3 py-1 rounded-full border border-border">
          {msg.content}
        </span>
      </div>
    );
  }
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`
          max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap
          ${isUser
            ? 'bg-accent text-bg-base rounded-br-sm'
            : 'bg-bg-surface2 text-text border border-border rounded-bl-sm'
          }
        `}
      >
        {msg.content}
      </div>
    </div>
  );
}

export default function ChatbotPage() {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [temperature, setTemperature] = useState(0.2);
  const [maxTokens, setMaxTokens] = useState(512);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [error, setError] = useState('');

  // Eval
  const [evalSamples, setEvalSamples] = useState(10);
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalLogs, setEvalLogs] = useState<string[]>([]);
  const [evalStatus, setEvalStatus] = useState('');
  const evalPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const evalOffsetRef = useRef(0);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  const fetchModels = useCallback(async () => {
    setLoadingModels(true);
    try {
      const res = await fetch(`${API}/api/chatbot/models`);
      if (res.ok) {
        const data = await res.json();
        const list: OllamaModel[] = data.models ?? data ?? [];
        setModels(list);
        if (list.length > 0 && !selectedModel) setSelectedModel(list[0].name);
      }
    } catch { /* ignore */ } finally {
      setLoadingModels(false);
    }
  }, [selectedModel]);

  useEffect(() => { fetchModels(); }, [fetchModels]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;
    if (!selectedModel) { setError('Wybierz model przed wysłaniem wiadomości.'); return; }

    const userMsg: ChatMessage = { role: 'user', content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setSending(true);
    setError('');

    try {
      const res = await fetch(`${API}/api/chatbot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          messages: newMessages,
          temperature,
          max_tokens: maxTokens,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      const assistantContent: string = data.message ?? data.content ?? data.response ?? '';
      setMessages(prev => [...prev, { role: 'assistant', content: assistantContent }]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd wysyłania wiadomości');
      setMessages(prev => prev.slice(0, -1)); // remove optimistic user message on error
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearHistory = () => {
    if (messages.length > 0 && !confirm('Wyczyścić historię czatu?')) return;
    setMessages([]);
    setError('');
  };

  const stopEvalPoll = useCallback(() => {
    if (evalPollRef.current) {
      clearInterval(evalPollRef.current);
      evalPollRef.current = null;
    }
  }, []);

  const pollEvalLogs = useCallback(async (runId: string) => {
    try {
      const res = await fetch(`${API}/api/chatbot/eval/log/${runId}?offset=${evalOffsetRef.current}`);
      if (res.ok) {
        const data = await res.json();
        if (data.lines?.length) {
          setEvalLogs(prev => [...prev, ...data.lines]);
          evalOffsetRef.current += data.lines.length;
        }
        if (data.status) setEvalStatus(data.status);
        if (data.status === 'done' || data.status === 'error') {
          stopEvalPoll();
          setEvalRunning(false);
        }
      }
    } catch { /* ignore */ }
  }, [stopEvalPoll]);

  useEffect(() => () => stopEvalPoll(), [stopEvalPoll]);

  const handleRunEval = async () => {
    if (!selectedModel) { setError('Wybierz model do ewaluacji.'); return; }
    setEvalRunning(true);
    setEvalLogs([]);
    setEvalStatus('starting');
    evalOffsetRef.current = 0;
    try {
      const res = await fetch(`${API}/api/chatbot/eval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, n_samples: evalSamples }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const runId: string = data.run_id ?? data.id ?? 'eval';
      if (evalPollRef.current) clearInterval(evalPollRef.current);
      evalPollRef.current = setInterval(() => pollEvalLogs(runId), 2000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Błąd ewaluacji');
      setEvalRunning(false);
      setEvalStatus('error');
    }
  };

  const evalLogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (evalLogRef.current) evalLogRef.current.scrollTop = evalLogRef.current.scrollHeight;
  }, [evalLogs]);

  return (
    <div className="flex gap-6 h-[calc(100vh-8rem)]">
      {/* ─── Left column: settings ─── */}
      <div className="w-72 flex-shrink-0 flex flex-col gap-5 overflow-y-auto">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-accent" />
          <h1 className="text-xl font-bold text-text">Chatbot</h1>
        </div>

        {/* Model selector */}
        <div className="card p-4 space-y-4">
          <h3 className="text-sm font-medium text-text">Ustawienia modelu</h3>
          <div>
            <label className="block text-xs text-text-muted mb-1.5">Model</label>
            {loadingModels ? (
              <div className="h-9 bg-bg-surface2 rounded animate-pulse" />
            ) : (
              <select
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value)}
                className="input-field"
              >
                {models.length === 0
                  ? <option value="">Brak dostępnych modeli</option>
                  : models.map(m => (
                    <option key={m.name} value={m.name}>
                      {m.name} ({(m.size_gb ?? 0).toFixed(1)} GB)
                    </option>
                  ))
                }
              </select>
            )}
            <button
              onClick={fetchModels}
              className="mt-1.5 text-xs text-text-muted hover:text-text flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" /> Odśwież modele
            </button>
          </div>

          {/* Temperature */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs text-text-muted">Temperatura</label>
              <span className="text-xs text-text font-mono">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0} max={1} step={0.01}
              value={temperature}
              onChange={e => setTemperature(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Max tokens */}
          <div>
            <label className="block text-xs text-text-muted mb-1.5">Maks. tokenów</label>
            <select
              value={maxTokens}
              onChange={e => setMaxTokens(parseInt(e.target.value))}
              className="input-field"
            >
              {[128, 256, 512, 1024, 2048].map(n => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Clear button */}
        <button onClick={clearHistory} className="btn-danger w-full justify-center">
          <Trash2 className="w-4 h-4" />
          Wyczyść historię
        </button>

        <div className="border-t border-border" />

        {/* Eval section */}
        <div className="card p-4 space-y-3">
          <h3 className="text-sm font-medium text-text">Ewaluacja</h3>
          <div>
            <label className="block text-xs text-text-muted mb-1.5">Liczba próbek</label>
            <input
              type="number"
              min={1} max={100}
              value={evalSamples}
              onChange={e => setEvalSamples(parseInt(e.target.value) || 10)}
              className="input-field"
            />
          </div>
          <button
            onClick={handleRunEval}
            disabled={evalRunning || !selectedModel}
            className="btn-primary w-full justify-center text-sm"
          >
            {evalRunning ? (
              <>
                <div className="w-4 h-4 border-2 border-bg-base/30 border-t-bg-base rounded-full animate-spin" />
                Ewaluacja...
              </>
            ) : '▶ Uruchom ewaluację'}
          </button>

          {(evalStatus || evalLogs.length > 0) && (() => {
            const ep = parseEvalProgress(evalLogs, evalSamples);
            const barStatus = evalStatus === 'done' ? 'done' : evalStatus === 'error' ? 'error' : 'running';
            return (
              <div className="space-y-2">
                {evalStatus && (
                  <div className="flex items-center gap-2 text-xs">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        evalStatus === 'done' ? 'bg-success' :
                        evalStatus === 'error' ? 'bg-error' :
                        'bg-accent animate-pulse'
                      }`}
                    />
                    <span className="text-text-muted">{evalStatus}</span>
                  </div>
                )}
                <ProgressBar
                  value={ep ? ep.pct : (evalStatus === 'done' ? 100 : 0)}
                  max={100}
                  label={ep ? `Próbka ${ep.done} / ${evalSamples}` : `0 / ${evalSamples}`}
                  status={barStatus}
                  size="sm"
                  indeterminate={!ep && evalStatus !== 'done' && evalStatus !== 'error' && evalRunning}
                />
                {evalLogs.length > 0 && (
                  <div
                    ref={evalLogRef}
                    className="log-box bg-black rounded border border-border h-32 overflow-y-auto p-2 font-mono text-xs text-green-400"
                  >
                    {evalLogs.map((l, i) => (
                      <div key={i} className="whitespace-pre-wrap break-all">{l}</div>
                    ))}
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      </div>

      {/* ─── Right column: chat ─── */}
      <div className="flex-1 flex flex-col min-w-0 card overflow-hidden">
        {/* Error banner */}
        {error && (
          <div className="alert-error flex items-center justify-between mx-4 mt-4 flex-shrink-0">
            <span className="text-sm">{error}</span>
            <button onClick={() => setError('')} className="font-bold text-lg ml-4">×</button>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
              <MessageSquare className="w-12 h-12 text-text-muted opacity-30" />
              <p className="text-text-muted text-lg">Wybierz model i zacznij rozmawiać</p>
              <p className="text-text-muted text-sm">
                {selectedModel
                  ? `Aktywny model: ${selectedModel}`
                  : 'Żaden model nie jest wybrany'}
              </p>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} />
              ))}
              {sending && (
                <div className="flex justify-start mb-3">
                  <div className="bg-bg-surface2 border border-border rounded-2xl rounded-bl-sm px-4 py-2.5">
                    <div className="flex gap-1 items-center">
                      <span className="text-text-muted text-sm">Pisanie</span>
                      <span className="flex gap-0.5">
                        {[0, 1, 2].map(i => (
                          <span
                            key={i}
                            className="w-1.5 h-1.5 rounded-full bg-text-muted animate-bounce"
                            style={{ animationDelay: `${i * 0.15}s` }}
                          />
                        ))}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-border p-4 flex-shrink-0">
          <div className="flex items-end gap-3">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                !selectedModel
                  ? 'Wybierz model...'
                  : sending
                  ? 'Czekam na odpowiedź...'
                  : 'Wpisz wiadomość... (Enter = wyślij, Shift+Enter = nowa linia)'
              }
              disabled={!selectedModel || sending}
              rows={2}
              className="
                flex-1 bg-bg-base border border-border rounded-lg px-3 py-2
                text-text text-sm resize-none
                focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/20
                disabled:opacity-50 disabled:cursor-not-allowed
                placeholder:text-text-muted
              "
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || !selectedModel || sending}
              className="
                flex-shrink-0 w-10 h-10 rounded-lg bg-accent text-bg-base
                flex items-center justify-center
                hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed
                transition-colors
              "
            >
              {sending
                ? <div className="w-4 h-4 border-2 border-bg-base/30 border-t-bg-base rounded-full animate-spin" />
                : <Send className="w-4 h-4" />
              }
            </button>
          </div>
          <p className="text-xs text-text-muted mt-1.5">
            Enter — wyślij · Shift+Enter — nowa linia
          </p>
        </div>
      </div>
    </div>
  );
}
