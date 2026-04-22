const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';
const ADMIN_API_KEY = process.env.NEXT_PUBLIC_ADMIN_API_KEY ?? '';

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
  id: string;
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
  const headers = new Headers(options?.headers);
  headers.set('Content-Type', 'application/json');
  if (ADMIN_API_KEY) {
    headers.set('X-API-Key', ADMIN_API_KEY);
  }

  const res = await fetch(url, {
    ...options,
    headers,
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

export function getAdminApiKey(): string {
  return ADMIN_API_KEY;
}

export function withApiKeyHeaders(headers?: HeadersInit): Headers {
  const merged = new Headers(headers);
  if (ADMIN_API_KEY) {
    merged.set('X-API-Key', ADMIN_API_KEY);
  }
  return merged;
}

// ─── Gap Scout Types ─────────────────────────────────────────────────────────

export interface ScoutSource {
  url: string;
  title: string;
  published_at: string;
  source_type: 'arxiv' | 'openalex' | 'hackernews' | 'eurlex' | string;
  verified: boolean;
  // new fields (optional for backwards compat)
  source_tier?: 'S' | 'A' | 'B' | 'C';
  language?: string;
  snippet?: string;
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
  // new fields (optional for backwards compat)
  knowledge_gap_score?: number;
  cutoff_model_targets?: string[];
  format_types?: string[];
  languages?: string[];
  citation_velocity?: number;
  source_tier?: 'S' | 'A' | 'B' | 'C';
  estimated_tokens?: number;
  ingest_ready?: boolean;
  dataset_category?: string;
  dataset_purpose?: string;
  demand_score?: number;
  uniqueness_score?: number;
  quality_score?: number;
  quality_gate_passed?: boolean;
  quality_gate_reasons?: string[];
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

// ─── KROK 11/12 — nowe typy Gap Scout ────────────────────────────────────────

export interface CrawlerStatus {
  source_id: string;
  poll_interval: number;
  is_paused: boolean;
  consecutive_errors: number;
  last_seen_id?: string | null;
}

export interface WebSubSubscription {
  topic_url: string;
  source_type: string;
  tier: string;
  verified: boolean;
  expires_in_s: number;
  delivery_count: number;
}

export interface WebSubStats {
  total_subscriptions: number;
  verified: number;
  pending_verification: number;
  feeds: number;
}

export interface ScoutSourcesResponse {
  crawlers: CrawlerStatus[];
  total_crawlers: number;
  active_crawlers: number;
  websub_subscriptions: WebSubSubscription[];
  websub_stats: WebSubStats;
}

export interface ModelGapsResponse {
  models: Record<string, ScoutTopic[]>;
  model_count: number;
  total_topics: number;
}

export interface FeedbackResponse {
  topic_id: string;
  status: string;
  feedback_count: number;
  avg_rating: number;
}

// ─── KROK 11/12 — nowe funkcje API ───────────────────────────────────────────

export async function getScoutSources(): Promise<ScoutSourcesResponse> {
  return apiFetch<ScoutSourcesResponse>('/api/scout/sources');
}

export async function getGapsByModel(
  model?: string,
  limit = 10,
): Promise<ModelGapsResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (model) params.set('model', model);
  return apiFetch<ModelGapsResponse>(`/api/scout/gaps/models?${params}`);
}

export async function getTrending(limit = 20, minVelocity = 0): Promise<ScoutTopic[]> {
  return apiFetch<ScoutTopic[]>(
    `/api/scout/trending?limit=${limit}&min_velocity=${minVelocity}`,
  );
}

export async function startTargetedRun(
  domains: string[],
  maxTopics = 20,
  minGapScore = 0,
): Promise<ScoutRun> {
  return apiFetch<ScoutRun>('/api/scout/run/targeted', {
    method: 'POST',
    body: JSON.stringify({
      domains,
      max_topics: maxTopics,
      min_gap_score: minGapScore,
    }),
  });
}

export async function submitFeedback(
  topicId: string,
  rating: number,
  helpful: boolean,
  comment = '',
): Promise<FeedbackResponse> {
  return apiFetch<FeedbackResponse>('/api/scout/feedback', {
    method: 'POST',
    body: JSON.stringify({ topic_id: topicId, rating, helpful, comment }),
  });
}

export function getSseUrl(): string {
  return `${API_BASE}/api/scout/live`;
}
