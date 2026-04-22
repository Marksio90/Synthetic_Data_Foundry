# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# First-time setup (copies .env, builds Docker images)
make setup

# Start core stack (API + PostgreSQL + Redis + Ollama)
make up

# Full automated pipeline: generate → quality gate → train → export
make full-run

# Individual pipeline steps
make generate          # Run Q&A generation on PDFs in data/
make gate              # Quality gate check on generated dataset
make train             # SFT + DPO training (requires GPU)
make export            # Export model to GGUF + ZIP

# Logs
make logs-api          # FastAPI backend
make logs-trainer      # Training container
docker compose logs -f foundry-api

# Start monitoring stack (Prometheus + Grafana + Loki + Tempo + Neo4j)
docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d

# Run pipeline with limits (env vars passed via shell)
CHUNK_LIMIT=50 BATCH_ID=test-run make generate
SKIP_DPO=1 make train

# Watermark verification
python scripts/verify_watermark.py --batch-id <id>
```

No test suite exists. Linting: `ruff check .` if ruff is installed.

## Architecture overview

The platform has three runtime layers:

**1. Document-to-dataset pipeline** (`pipeline/graph.py`)  
A LangGraph `StateGraph` that processes each directive chunk through a fixed sequence: `simulate_question → retrieve_context → generate_answer → constitutional_revision → judge_answer`, then branches on quality score into retry / multi-turn continuation / write. After a successful write, two new tail nodes run: `knowledge_graph_node` (entity extraction into Neo4j/networkx) and `self_improving_loop_node` (parameter nudging via EMA). State is typed in `pipeline/state.py` (`FoundryState`).

**2. FastAPI backend** (`api/`)  
`api/main.py` → `api/bootstrap.py` (lifespan, middleware, structured logging via `config/logging_config.py`). Routers in `api/routers/` expose five domains: `pipeline`, `documents`, `training`, `scout`, `chatbot`. Run state lives in `api/state.py` (`RunManager`), fire-and-forget persisted to Redis via `api/redis_state.py`, durably stored in PostgreSQL.

**3. LLM routing** (`utils/llm_router.py`)  
All LLM calls go through LiteLLM proxy (`config/litellm_config.yaml`) using four tier aliases: `foundry/local` (Ollama qwen2.5), `foundry/mid` (gpt-4o-mini), `foundry/quality` (gpt-4o), `foundry/judge` (gpt-4o). Direct SDK fallback when proxy is unreachable.

## Key agents

| File | Role |
|---|---|
| `agents/ingestor.py` | PDF/DOCX/HTML/audio → chunks (LlamaParse → OpenAI Vision → PyMuPDF fallback chain, all with tenacity circuit breakers) |
| `agents/expert.py` | RAG retrieval (pgvector HNSW + BM25) + answer generation; `embed_batch()` batches up to 2048 texts |
| `agents/judge.py` | LLM-as-a-Judge with Pydantic structured output; scores grounding, hallucination, completeness |
| `agents/calibrator.py` | Zero-cost parameter tuning from chunk heuristics (no LLM calls) → `CalibrationResult` |
| `agents/critic_agent.py` | 20% selective critique → generates DPO `rejected` pairs via targeted degradation |
| `agents/memory_agent.py` | Dual-tier: Redis 24h TTL (short-term) + Qdrant semantic search (long-term) |
| `agents/cost_optimizer.py` | Background monitor: auto-downgrade to `foundry/local` at 80% budget, hard-stop at `COST_HARD_LIMIT_USD` |
| `agents/self_improving_loop.py` | EMA-smoothed calibration adjustment from quality signals; persists to `calibration_history` |
| `agents/knowledge_graph.py` | spaCy NER + ESG regex → entity graph; Neo4j backend with networkx fallback |
| `agents/batch_analyzer.py` | OpenAI Batch API (50% cost, 24h window) for non-realtime judging |

## Database schema (`init/01_schema.sql`)

Key tables: `source_documents` → `directive_chunks` (HNSW vector index, BM25 trigram index, FTS) → `generated_samples`. Supporting tables: `watermark_registry`, `openai_batch_jobs`, `workflow_cost_ledger` (FinOps), `calibration_history` (self-improvement audit trail), `scout_domain_history/exclusions`. Stored procedures: `claim_chunk_for_processing()` (ACID idempotency), `finalize_chunk()` (status FSM).

PostgreSQL models are in `db/models.py`; all queries go through `db/repository.py` (no raw SQL in business logic).

## Configuration

All settings in `config/settings.py` (Pydantic `BaseSettings`, loaded from `.env`). Key env vars:

```
OPENAI_API_KEY          required
POSTGRES_PASSWORD       required
OLLAMA_URL              default: http://ollama:11434
MLFLOW_TRACKING_URI     default: http://mlflow:5000
NEO4J_URL               e.g. bolt://neo4j:7687
NEO4J_AUTH              default: neo4j:foundry-neo4j
QUALITY_THRESHOLD       default: 0.82
ADVERSARIAL_RATIO       default: 0.10
MAX_TURNS               default: 3
COST_HARD_LIMIT_USD     default: 50.0
```

## Distributed infrastructure (optional, graceful degradation)

All infrastructure components fail silently if unavailable:

- **Temporal.io** (`orchestration/temporal_workflows.py`) — durable workflow DAG; falls back to direct function calls
- **NATS JetStream** (`nats/bus.py`) — event bus; falls back to in-process `asyncio.Queue`
- **Redis** (`api/redis_state.py`) — cross-replica state cache; falls back to in-memory dict
- **OpenTelemetry** (`telemetry/tracing.py`) — OTLP → Grafana Tempo; falls back to noop tracer

## Training pipeline

`training/train_sft.py` runs Unsloth LoRA fine-tuning inside `docker/Dockerfile.trainer` (not available in the main API container). `training/train_dpo.py` applies DPO alignment on top of the SFT adapter. Both wrap their runs with `training/mlflow_tracker.py` (`FoundryMLflowTracker`) that logs hyperparams, per-step metrics, artifacts, and registers the model in MLflow Model Registry. Training quality is gated by `training/quality_gate.py` before export.

## Watermarking

`pipeline/watermark_v2.py` (`WatermarkV2`) uses two simultaneous techniques: HMAC-keyed synonym substitution (linguistic steganography) and zero-width character injection after punctuation. Detection is blind (no key needed, confidence score); verification is keyed. The old `pipeline/watermark.py` remains for backwards compatibility but new code should use `WatermarkV2`.

## Kubernetes deployment

Helm chart in `helm/foundry/`. Key values: `replicaCount.api`, `replicaCount.worker`, `worker.autoscaling.*`. The chart deploys API + Worker deployments with HPA. Secrets are referenced via `values.yaml` → `secrets.existingSecret`.
