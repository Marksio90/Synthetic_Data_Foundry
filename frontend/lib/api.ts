const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';

// ─── Response Types ───────────────────────────────────────────────────────────

export interface Document {
  filename: string;
  size_bytes: number;
  uploaded_at: string;
  in_db: boolean;
  chunk_count: number;
  sample_count: number;
}

export interface Sample {
  id: number;
  question: string;
  answer: string;
  perspective: string;
  difficulty: string;
  question_type: string;
  quality_score: number;
  is_adversarial: boolean;
  batch_id: string;
  has_dpo: boolean;
  turn_count: number;
  created_at: string;
}

export interface SampleDetail extends Sample {
  rejected_answer: string | null;
  judge_model: string | null;
  judge_reasoning: string | null;
  record_index: number | null;
  conversation_json: unknown | null;
}

export interface SampleStats {
  total: number;
  avg_quality_score: number;
  dpo_pairs: number;
  perspectives: Record<string, number>;
  difficulties: Record<string, number>;
}

export interface PipelineRun {
  run_id: string;
  batch_id: string;
  status: string;
  progress_pct: number;
  chunks_done: number;
  chunks_total: number;
  records_written: number;
  dpo_pairs: number;
  elapsed_seconds: number;
}

export interface AnalysisResult {
  language: string;
  translation_required: boolean;
  domain_label: string;
  domain_confidence: number;
  perspectives: string[];
  auto_decisions: string[];
  calibration: {
    quality_threshold: number;
    max_turns: number;
    adversarial_ratio: number;
    reasoning: string[];
  };
}

export interface TrainingRun {
  run_id: string;
  batch_id: string;
  status: string;
  elapsed_seconds: number;
}

export type HardwareInfo = Record<string, unknown>;

export interface GateCheck {
  name: string;
  passed: boolean;
  value: number | string;
  threshold: number | string;
  message: string;
}

export interface GateResult {
  passed: boolean;
  checks: GateCheck[];
  warnings: string[];
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface OllamaModel {
  name: string;
  size_gb: number;
  modified_at: string;
}

// ─── API Fetch Wrapper ────────────────────────────────────────────────────────

export async function apiFetch<T = unknown>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
  });

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body.detail ?? body.message ?? body.error ?? message;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(message);
  }

  // Handle 204 No Content
  if (res.status === 204) {
    return undefined as T;
  }

  return res.json() as Promise<T>;
}

// ─── Gap Scout Types ─────────────────────────────────────────────────────────

export interface ScoutSource {
  url: string;
  title: string;
  published_at: string;
  source_type: 'arxiv' | 'openalex' | 'hackernews' | 'eurlex' | string;
  verified: boolean;
}

export interface ScoutTopic {
  topic_id: string;
  title: string;
  summary: string;
  score: number;
  recency_score: number;
  llm_uncertainty: number;
  source_count: number;
  social_signal: number;
  sources: ScoutSource[];
  domains: string[];
  discovered_at: string;
}

export interface ScoutRun {
  scout_id: string;
  status: 'starting' | 'running' | 'done' | 'error';
  topics_found: number;
  elapsed_seconds: number;
  error?: string | null;
}

export async function startScoutRun(domains?: string[]): Promise<ScoutRun> {
  return apiFetch<ScoutRun>('/api/scout/run', {
    method: 'POST',
    body: JSON.stringify(domains ? { domains } : {}),
  });
}

export async function getScoutStatus(scoutId: string): Promise<ScoutRun> {
  return apiFetch<ScoutRun>(`/api/scout/status/${scoutId}`);
}

export async function getScoutTopics(limit = 50): Promise<ScoutTopic[]> {
  return apiFetch<ScoutTopic[]>(`/api/scout/topics?limit=${limit}`);
}

export async function ingestTopic(topicId: string): Promise<{
  message: string;
  sources_downloaded: number;
  errors: number;
  output_dir: string;
  paths: string[];
}> {
  return apiFetch(`/api/scout/ingest/${topicId}`, { method: 'POST' });
}

export function getWsBase(): string {
  const base = API_BASE.replace(/^http/, 'ws').replace(/^https/, 'wss');
  return base;
}
