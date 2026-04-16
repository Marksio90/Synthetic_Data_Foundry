'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Cpu, Download, RefreshCw } from 'lucide-react';
import type { HardwareInfo, GateResult, TrainingRun } from '@/lib/api';
import StatusBadge from '@/components/StatusBadge';
import ProgressBar from '@/components/ProgressBar';

interface TrainProgress {
  step?: number;
  totalSteps?: number;
  epoch?: number;
  totalEpochs?: number;
  loss?: number;
  pct?: number;
}

function parseTrainProgress(logs: string[], configEpochs?: number): TrainProgress {
  for (let i = logs.length - 1; i >= Math.max(0, logs.length - 80); i--) {
    const line = logs[i];

    // tqdm: "50/500 [01:30<00:30"  — step-level
    const tqdm = line.match(/\b(\d+)\s*\/\s*(\d+)\s+\[/);
    if (tqdm) {
      const step = parseInt(tqdm[1]);
      const total = parseInt(tqdm[2]);
      if (total > 0 && step <= total)
        return { step, totalSteps: total, pct: (step / total) * 100 };
    }

    // HF Trainer JSON dict: "{'loss': 1.23, ..., 'epoch': 1.5}"
    const epochF = line.match(/'epoch':\s*([\d.]+)/);
    if (epochF) {
      const epoch = parseFloat(epochF[1]);
      const lossM = line.match(/'loss':\s*([\d.]+)/);
      const loss = lossM ? parseFloat(lossM[1]) : undefined;
      const total = configEpochs ?? undefined;
      return { epoch, totalEpochs: total, loss, pct: total ? Math.min(100, (epoch / total) * 100) : undefined };
    }

    // "Epoch 2 / 3" or "epoch 2/3"
    const epochNM = line.match(/epoch\s+(\d+)\s*[\/of]\s*(\d+)/i);
    if (epochNM) {
      const epoch = parseInt(epochNM[1]);
      const total = parseInt(epochNM[2]);
      return { epoch, totalEpochs: total, pct: total > 0 ? (epoch / total) * 100 : undefined };
    }

    // "Step 50 / 500" or "step 50/500"
    const stepM = line.match(/steps?\s+(\d+)\s*[\/of]\s*(\d+)/i);
    if (stepM) {
      const step = parseInt(stepM[1]);
      const total = parseInt(stepM[2]);
      return { step, totalSteps: total, pct: total > 0 ? (step / total) * 100 : undefined };
    }
  }
  return {};
}

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

type Tab = 'hardware' | 'gate' | 'trening' | 'eksport';

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 text-sm font-medium rounded-t-md border-b-2 transition-colors
        ${active
          ? 'border-accent text-accent bg-bg-surface2'
          : 'border-transparent text-text-muted hover:text-text hover:border-border'}
      `}
    >
      {label}
    </button>
  );
}

function KVTable({ data }: { data: Record<string, unknown> }) {
  return (
    <div className="space-y-1">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="flex gap-3 text-sm py-1 border-b border-border/50 last:border-0">
          <span className="text-text-muted w-48 flex-shrink-0 font-mono text-xs">{k}</span>
          <span className="text-text font-mono text-xs break-all">
            {v === null ? 'null' : typeof v === 'object' ? JSON.stringify(v) : String(v)}
          </span>
        </div>
      ))}
    </div>
  );
}

function LogBox({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines]);
  return (
    <div ref={ref} className="log-box bg-black rounded-md border border-border h-48 overflow-y-auto p-3 font-mono text-xs text-green-400 leading-5">
      {lines.length === 0
        ? <span className="text-text-muted">Oczekiwanie na logi...</span>
        : lines.map((l, i) => <div key={i} className="whitespace-pre-wrap break-all">{l}</div>)
      }
    </div>
  );
}

export default function TreningPage() {
  const [tab, setTab] = useState<Tab>('hardware');

  // Hardware tab
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [loadingHardware, setLoadingHardware] = useState(false);

  // Gate tab
  const [gate, setGate] = useState<GateResult | null>(null);
  const [loadingGate, setLoadingGate] = useState(false);
  const [gateError, setGateError] = useState('');

  // Training tab
  const [baseModel, setBaseModel] = useState('');
  const [loraRank, setLoraRank] = useState(16);
  const [epochs, setEpochs] = useState(3);
  const [runName, setRunName] = useState('');
  const [skipDpo, setSkipDpo] = useState(false);
  const [skipEval, setSkipEval] = useState(false);
  const [trainRun, setTrainRun] = useState<TrainingRun | null>(null);
  const [trainError, setTrainError] = useState('');
  const [trainLogs, setTrainLogs] = useState<string[]>([]);
  const [trainRunning, setTrainRunning] = useState(false);
  const trainPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logOffsetRef = useRef(0);

  // Export tab
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [exportStatus, setExportStatus] = useState<Record<string, string>>({});

  const fetchHardware = async () => {
    setLoadingHardware(true);
    try {
      const res = await fetch(`${API}/api/training/hardware`);
      if (res.ok) setHardware(await res.json());
      else throw new Error(`HTTP ${res.status}`);
    } catch (e: unknown) {
      setHardware({ error: e instanceof Error ? e.message : 'Błąd' });
    } finally {
      setLoadingHardware(false);
    }
  };

  const runGate = async () => {
    setLoadingGate(true);
    setGateError('');
    setGate(null);
    try {
      const res = await fetch(`${API}/api/training/gate`, { method: 'POST' });
      if (!res.ok) throw new Error((await res.json()).detail ?? `HTTP ${res.status}`);
      setGate(await res.json());
    } catch (e: unknown) {
      setGateError(e instanceof Error ? e.message : 'Błąd quality gate');
    } finally {
      setLoadingGate(false);
    }
  };

  const pollTrainLogs = useCallback(async (runId: string) => {
    try {
      const res = await fetch(`${API}/api/training/log/${runId}?offset=${logOffsetRef.current}`);
      if (res.ok) {
        const data = await res.json();
        if (data.lines?.length) {
          setTrainLogs(prev => [...prev, ...data.lines]);
          logOffsetRef.current += data.lines.length;
        }
      }
    } catch { /* ignore */ }
  }, []);

  const startTrainPoll = useCallback((runId: string) => {
    if (trainPollRef.current) clearInterval(trainPollRef.current);
    trainPollRef.current = setInterval(async () => {
      await pollTrainLogs(runId);
      try {
        const res = await fetch(`${API}/api/training/status/${runId}`);
        if (res.ok) {
          const data: TrainingRun = await res.json();
          setTrainRun(data);
          if (data.status === 'done' || data.status === 'error') {
            clearInterval(trainPollRef.current!);
            trainPollRef.current = null;
            setTrainRunning(false);
          }
        }
      } catch { /* ignore */ }
    }, 3000);
  }, [pollTrainLogs]);

  useEffect(() => () => { if (trainPollRef.current) clearInterval(trainPollRef.current); }, []);

  const handleStartTrain = async () => {
    if (!baseModel.trim()) { setTrainError('Podaj model bazowy.'); return; }
    setTrainError('');
    setTrainRunning(true);
    setTrainRun(null);
    setTrainLogs([]);
    logOffsetRef.current = 0;
    try {
      const payload: Record<string, unknown> = {
        base_model: baseModel.trim(),
        lora_rank: loraRank,
        epochs,
        skip_dpo: skipDpo,
        skip_eval: skipEval,
      };
      if (runName.trim()) payload.run_name = runName.trim();
      const res = await fetch(`${API}/api/training/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? `HTTP ${res.status}`);
      const data: TrainingRun = await res.json();
      setTrainRun(data);
      startTrainPoll(data.run_id);
    } catch (e: unknown) {
      setTrainError(e instanceof Error ? e.message : 'Błąd uruchamiania treningu');
      setTrainRunning(false);
    }
  };

  const fetchRuns = useCallback(async () => {
    setLoadingRuns(true);
    try {
      const res = await fetch(`${API}/api/training/runs`);
      if (res.ok) {
        const data = await res.json();
        setTrainingRuns(data.runs ?? data ?? []);
      }
    } catch { /* ignore */ } finally {
      setLoadingRuns(false);
    }
  }, []);

  useEffect(() => { if (tab === 'eksport') fetchRuns(); }, [tab, fetchRuns]);

  const handleExport = async (runId: string) => {
    setExportStatus(prev => ({ ...prev, [runId]: 'exporting' }));
    try {
      const res = await fetch(`${API}/api/training/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setExportStatus(prev => ({ ...prev, [runId]: 'done' }));
    } catch {
      setExportStatus(prev => ({ ...prev, [runId]: 'error' }));
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Cpu className="w-6 h-6 text-accent" />
        <h1 className="text-2xl font-bold text-text">Trening</h1>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        {(['hardware', 'gate', 'trening', 'eksport'] as Tab[]).map(t => (
          <TabButton
            key={t}
            label={t === 'hardware' ? 'Hardware' : t === 'gate' ? 'Quality Gate' : t === 'trening' ? 'Trening' : 'Eksport'}
            active={tab === t}
            onClick={() => setTab(t)}
          />
        ))}
      </div>

      {/* ─── Hardware tab ─── */}
      {tab === 'hardware' && (
        <div className="space-y-4">
          <button onClick={fetchHardware} disabled={loadingHardware} className="btn-primary">
            {loadingHardware ? (
              <>
                <div className="w-4 h-4 border-2 border-bg-base/30 border-t-bg-base rounded-full animate-spin" />
                Sprawdzanie...
              </>
            ) : '🖥️ Sprawdź sprzęt'}
          </button>
          {hardware && (
            <div className="card p-5">
              <KVTable data={hardware as Record<string, unknown>} />
            </div>
          )}
        </div>
      )}

      {/* ─── Quality Gate tab ─── */}
      {tab === 'gate' && (
        <div className="space-y-4">
          <button onClick={runGate} disabled={loadingGate} className="btn-primary">
            {loadingGate ? (
              <>
                <div className="w-4 h-4 border-2 border-bg-base/30 border-t-bg-base rounded-full animate-spin" />
                Uruchamianie...
              </>
            ) : '🔒 Uruchom Quality Gate'}
          </button>

          {gateError && <div className="alert-error">{gateError}</div>}

          {gate && (
            <div className="space-y-4">
              {/* Pass/Fail header */}
              <div className={`card p-4 flex items-center gap-3 ${gate.passed ? 'border-success/40' : 'border-error/40'}`}>
                <span className={`text-2xl ${gate.passed ? 'text-success' : 'text-error'}`}>
                  {gate.passed ? '✅' : '❌'}
                </span>
                <div>
                  <p className={`font-bold text-lg ${gate.passed ? 'text-success' : 'text-error'}`}>
                    {gate.passed ? 'Quality Gate PRZESZEDŁ' : 'Quality Gate NIE PRZESZEDŁ'}
                  </p>
                  <p className="text-text-muted text-sm">{gate.checks.length} sprawdzeń</p>
                </div>
              </div>

              {/* Checks table */}
              <div className="card overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-bg-surface2 text-text-muted text-xs uppercase">
                      <th className="px-4 py-2 text-left">Sprawdzenie</th>
                      <th className="px-4 py-2 text-right">Wartość</th>
                      <th className="px-4 py-2 text-right">Próg</th>
                      <th className="px-4 py-2 text-center">Status</th>
                      <th className="px-4 py-2 text-left">Wiadomość</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gate.checks.map((c, i) => (
                      <tr key={i} className="border-b border-border hover:bg-bg-surface2">
                        <td className="px-4 py-2 text-text font-medium">{c.name}</td>
                        <td className="px-4 py-2 text-right font-mono text-xs">{String(c.value)}</td>
                        <td className="px-4 py-2 text-right font-mono text-xs text-text-muted">{String(c.threshold)}</td>
                        <td className="px-4 py-2 text-center">
                          {c.passed
                            ? <span className="text-success text-base">✓</span>
                            : <span className="text-error text-base">✗</span>}
                        </td>
                        <td className="px-4 py-2 text-text-muted text-xs">{c.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Warnings */}
              {gate.warnings.length > 0 && (
                <div className="space-y-1">
                  <p className="text-sm font-medium text-warning">Ostrzeżenia:</p>
                  {gate.warnings.map((w, i) => (
                    <div key={i} className="alert-warning text-sm">{w}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ─── Trening tab ─── */}
      {tab === 'trening' && (
        <div className="space-y-5">
          {!trainRunning && !trainRun ? (
            <div className="card p-6 space-y-5 max-w-2xl">
              <h3 className="font-semibold text-text">Konfiguracja treningu LoRA</h3>

              {trainError && <div className="alert-error">{trainError}</div>}

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-text-muted mb-1.5">Model bazowy *</label>
                  <input
                    type="text"
                    value={baseModel}
                    onChange={e => setBaseModel(e.target.value)}
                    className="input-field"
                    placeholder="np. unsloth/llama-3-8b-bnb-4bit"
                  />
                </div>

                <div>
                  <label className="block text-sm text-text-muted mb-1.5">LoRA Rank</label>
                  <select value={loraRank} onChange={e => setLoraRank(parseInt(e.target.value))} className="input-field">
                    {[8, 16, 32, 64].map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-text-muted mb-1.5">Liczba epok</label>
                  <select value={epochs} onChange={e => setEpochs(parseInt(e.target.value))} className="input-field">
                    {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-text-muted mb-1.5">Nazwa runu (opcjonalne)</label>
                  <input
                    type="text"
                    value={runName}
                    onChange={e => setRunName(e.target.value)}
                    className="input-field"
                    placeholder="auto-generowana"
                  />
                </div>

                <div className="flex gap-6">
                  <label className="flex items-center gap-2 cursor-pointer text-sm text-text-muted">
                    <input type="checkbox" checked={skipDpo} onChange={e => setSkipDpo(e.target.checked)} />
                    Pomiń DPO
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-sm text-text-muted">
                    <input type="checkbox" checked={skipEval} onChange={e => setSkipEval(e.target.checked)} />
                    Pomiń ewaluację
                  </label>
                </div>
              </div>

              <button onClick={handleStartTrain} className="btn-primary w-full justify-center py-3">
                ▶ Uruchom trening
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="font-semibold text-text">Status treningu</span>
                  {trainRun && <StatusBadge status={trainRun.status} />}
                </div>
                {!trainRunning && (
                  <button
                    onClick={() => { setTrainRun(null); setTrainRunning(false); setTrainLogs([]); setTrainError(''); }}
                    className="btn-secondary text-sm"
                  >
                    Nowy trening
                  </button>
                )}
              </div>

              {trainRun && (() => {
                const tp = parseTrainProgress(trainLogs, trainRun.config?.epochs ?? epochs);
                const barStatus = trainRun.status === 'error' ? 'error' : trainRun.status === 'done' ? 'done' : 'running';
                const hasProgress = tp.pct !== undefined;
                const progressLabel = tp.totalSteps
                  ? `Krok ${tp.step} / ${tp.totalSteps}`
                  : tp.totalEpochs
                  ? `Epoka ${tp.epoch?.toFixed(1)} / ${tp.totalEpochs}`
                  : tp.epoch !== undefined
                  ? `Epoka ${tp.epoch?.toFixed(2)}`
                  : 'Trenowanie…';
                const progressRight = tp.loss !== undefined
                  ? `loss: ${tp.loss.toFixed(4)} · ${hasProgress ? (tp.pct!).toFixed(1) + '%' : ''}`
                  : hasProgress ? `${tp.pct!.toFixed(1)}%` : undefined;

                return (
                  <div className="space-y-3">
                    <div className="grid grid-cols-3 gap-3">
                      <div className="metric-card">
                        <div className="metric-value">{(trainRun.elapsed_seconds ?? 0).toFixed(0)}s</div>
                        <div className="metric-label">Czas</div>
                      </div>
                      {tp.loss !== undefined && (
                        <div className="metric-card">
                          <div className="metric-value text-lg">{tp.loss.toFixed(4)}</div>
                          <div className="metric-label">Loss</div>
                        </div>
                      )}
                      {tp.epoch !== undefined && (
                        <div className="metric-card">
                          <div className="metric-value text-lg">{typeof tp.epoch === 'number' ? tp.epoch.toFixed(2) : tp.epoch}</div>
                          <div className="metric-label">Epoka</div>
                        </div>
                      )}
                    </div>

                    <ProgressBar
                      value={tp.pct ?? 0}
                      max={100}
                      label={progressLabel}
                      valueLabel={progressRight}
                      status={barStatus}
                      size="md"
                      indeterminate={!hasProgress && trainRun.status === 'running'}
                    />
                  </div>
                );
              })()}

              {trainError && <div className="alert-error">{trainError}</div>}

              <div>
                <p className="text-sm text-text-muted mb-1.5">Logi:</p>
                <LogBox lines={trainLogs} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─── Eksport tab ─── */}
      {tab === 'eksport' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-text">Runy treningu</h3>
            <button onClick={fetchRuns} disabled={loadingRuns} className="btn-secondary text-sm">
              <RefreshCw className={`w-4 h-4 ${loadingRuns ? 'animate-spin' : ''}`} />
              Odśwież
            </button>
          </div>

          {loadingRuns ? (
            <div className="space-y-2">
              {[1,2,3].map(i => <div key={i} className="h-12 bg-bg-surface2 rounded animate-pulse" />)}
            </div>
          ) : trainingRuns.length === 0 ? (
            <div className="card p-8 text-center text-text-muted">
              Brak ukończonych runów treningowych.
            </div>
          ) : (
            <div className="card overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-bg-surface2 text-text-muted text-xs uppercase">
                    <th className="px-4 py-3 text-left">Run ID</th>
                    <th className="px-4 py-3 text-left">Batch ID</th>
                    <th className="px-4 py-3 text-center">Status</th>
                    <th className="px-4 py-3 text-right">Czas</th>
                    <th className="px-4 py-3 text-center">Akcje</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingRuns.map(run => (
                    <tr key={run.run_id} className="border-b border-border hover:bg-bg-surface2">
                      <td className="px-4 py-3 font-mono text-xs text-text">{run.run_id}</td>
                      <td className="px-4 py-3 text-text-muted text-xs">{run.batch_id}</td>
                      <td className="px-4 py-3 text-center">
                        <StatusBadge status={run.status} />
                      </td>
                      <td className="px-4 py-3 text-right text-text-muted text-xs">
                        {(run.elapsed_seconds ?? 0).toFixed(0)}s
                      </td>
                      <td className="px-4 py-3 text-center">
                        {run.status === 'done' && (
                          <div className="flex items-center justify-center gap-2">
                            {exportStatus[run.run_id] === 'done' ? (
                              <a
                                href={`${API}/api/training/export/download/${run.run_id}`}
                                className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-success/10 border border-success/30 text-success text-xs hover:bg-success/20 transition-colors"
                                download
                              >
                                <Download className="w-3.5 h-3.5" />
                                Pobierz ZIP
                              </a>
                            ) : (
                              <button
                                onClick={() => handleExport(run.run_id)}
                                disabled={exportStatus[run.run_id] === 'exporting'}
                                className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-accent/10 border border-accent/30 text-accent text-xs hover:bg-accent/20 disabled:opacity-50 transition-colors"
                              >
                                {exportStatus[run.run_id] === 'exporting' ? (
                                  <>
                                    <div className="w-3 h-3 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                                    Eksportowanie...
                                  </>
                                ) : exportStatus[run.run_id] === 'error' ? (
                                  '❌ Błąd — spróbuj ponownie'
                                ) : (
                                  'Eksportuj'
                                )}
                              </button>
                            )}
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
