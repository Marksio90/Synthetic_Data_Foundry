# Synthetic Data Foundry — Full Audit + 2026 Redesign (Principal AI Systems Architect Review)

**Date:** April 22, 2026 (UTC)  
**Scope:** backend, frontend, pipeline, data layer, runtime, infra, agent system, operating model  
**Method:** static code/architecture review based on repository sources

---

## 1) AUDIT — Current Architecture Assessment

## 1.1 Executive verdict

Current platform is **feature-rich but operationally under-architected for large-scale multi-tenant AI operations**.

What is strong:
- Real modular split (`agents/`, `api/`, `pipeline/`, `training/`, `frontend/`, `db/`) and explicit API composition.  
- Useful enterprise primitives: admin API key guard, middleware stack, health endpoints, scheduler hooks, run logging, pgvector data model.  
- Practical hybrid world: local inference (Ollama), OpenAI fallback, Dockerized deployment, and basic observability surface (`/metrics`).

What blocks scale:
- Core orchestration state is in-memory and process-local.
- No event bus backbone (only subprocess + in-process task maps).
- Tight coupling between orchestration and API node lifecycle.
- Single-region/single-process assumptions in critical control paths.
- Frontend embeds admin key through public env pattern.

---

## 1.2 Architecture audit (backend / frontend / infra / data)

### Backend

**Strengths**
- FastAPI bootstrap is separated into composition helpers (`configure_middlewares`, `register_routers`, `create_lifespan`) which improves maintainability.  
- Typed config through `pydantic-settings` with broad coverage of pipeline/scout/runtime toggles.  
- DB preflight checks and typed API errors on pipeline endpoints improve operator debuggability.

**Critical issues**
1. **Run state is process-local and volatile by design** (`RunManager` in memory + best-effort snapshot to `/tmp`). Horizontal scaling or pod restarts will produce partial or inconsistent control-plane visibility.
2. **Pipeline execution model relies on spawned subprocesses from API workers**, with run/task maps stored in Python dicts. This is not resilient across worker restarts and complicates cancellation/recovery semantics.
3. **Scheduler and WebSub subscriptions are bound to API lifespan**; this pattern is fragile in multi-replica deployments (duplicate schedulers, duplicate subscriptions) unless leadership is added.
4. **Security boundary leakage to frontend** (public admin key variable model) creates an anti-pattern for production multi-user deployments.

### Frontend

**Strengths**
- Next.js + TypeScript app with dashboard-like domain views and a shared API client layer.
- Centralized fetch wrapper with typed interfaces.

**Critical issues**
1. `NEXT_PUBLIC_ADMIN_API_KEY` indicates control-plane credentials can be shipped to browser clients.
2. Frontend talks directly to backend admin-protected routes; no BFF/server action isolation for privileged operations.
3. Streamlit UI remains in repo, creating dual-surface product complexity and governance overhead.

### Infrastructure

**Strengths**
- Docker Compose stack includes Postgres+pgvector, API, frontend, Ollama and profiles for pipeline/training/chatbot.
- Resource limits and health checks are present for core services.

**Critical issues**
1. Infra is **Compose-first** with no production-grade Kubernetes manifests, autoscaling, workload separation or queue-centric execution topology.
2. No explicit GitOps deployment model in repository (ArgoCD not implemented), no progressive delivery patterns.
3. No formal SLO/SLA and no distributed tracing wiring visible in code paths.

### Data layer

**Strengths**
- SQLAlchemy models include source documents, chunks, generated samples, watermark registry, and batch jobs.
- pgvector columns and metadata-rich sample schema support quality governance and retrieval.

**Critical issues**
1. Vector storage is embedded in primary relational DB; no separation for high-scale ANN workloads.
2. No explicit data retention tiers / lifecycle policies / archival strategy in code.
3. Source lineage exists partially (chunk/sample linkage), but strict evidence trace package for each answer is not formalized as a first-class contract.

---

## 1.3 Code quality & system coherence

### Good patterns observed
- Modular routers and typed schemas.
- Security helper using constant-time comparison.
- Request context middleware with request IDs and process timing headers.
- Tests exist for selected security/middleware/pipeline paths.

### Quality debt
- Mixed language/comments and mixed style conventions increase maintenance friction.
- Orchestration, runtime execution, and state persistence boundaries are not separated into clean control-plane/data-plane services.
- Configuration surface is extensive, but policy enforcement (which knobs are allowed per tenant/environment) is missing.

---

## 1.4 Bottlenecks

1. **Control-plane bottleneck:** in-memory run manager, single-process assumptions.  
2. **Execution bottleneck:** subprocess-per-run orchestration from API worker process.  
3. **Data bottleneck:** Postgres handles transactional + vector duties, increasing contention risk at scale.  
4. **Model-cost bottleneck:** limited explicit LLM routing policy engine (cost/latency/quality dynamic arbitration).  
5. **Ops bottleneck:** limited distributed observability for end-to-end workflow debugging.

---

## 1.5 Security audit

**Positive**
- Admin key guards exist for sensitive endpoints.
- Webhook security options and HMAC verification paths exist.

**Risks**
1. Public frontend env pattern can expose privileged API credentials.
2. No explicit RBAC/ABAC and tenant-scoped auth model in current architecture.
3. No visible secret manager integration (Vault/KMS) pattern in repo.
4. Missing evidence of policy-as-code guardrails for agent tools/actions.

---

## 1.6 Scalability audit

Current design supports **single deployment unit scale-up** but not robust **scale-out**:
- state and scheduling not externalized,
- worker orchestration not queue-native,
- limited fault-domain isolation between API and execution runtime.

---

## 1.7 Cost audit (compute / LLM / storage)

Main cost leak vectors:
1. Repeated long-context generation loops without strict token budgeting and adaptive model downgrade.
2. Embedding + retrieval pipeline without explicit hot/cold cache strategy.
3. Combined OLTP + vector operations in same store for scale workloads.
4. Lack of per-tenant/per-run hard budget enforcement.

---

## 1.8 Risk register (prioritized)

### P0
- Control-plane state loss/inconsistency during restart/scale events.
- Privileged key exposure model in browser clients.
- Duplicate schedulers/websub workers in multi-replica deployment.

### P1
- Execution reliability issues due to subprocess orchestration model.
- Missing queue-first async pipeline semantics.
- No centralized policy engine for tool invocation and compliance.

### P2
- Product complexity from dual UI stacks.
- Observability depth insufficient for enterprise RCA.

---

## 2) NEW ARCHITECTURE (2026-Ready)

## 2.1 Target principles

- Agent-first and event-driven by default.
- Stateless API/control gateways.
- Queue-native execution.
- Clear split: **Control Plane / Agent Runtime / Data Plane / Experience Plane**.
- Portable from laptop → single-node → K8s cluster.

## 2.2 Proposed diagram (descriptive)

```text
[Next.js App + BFF]
        |
        v
[API Gateway + AuthN/Z + Policy]
        |
        v
[Orchestrator Agent Service] <----> [Workflow Graph Engine]
        |                                  |
        | events                           | state transitions
        v                                  v
[NATS/Kafka Event Bus]  <---------->  [Task Queue (Redis Streams / NATS JetStream)]
        |
        +--> [Scout Agent Workers]
        +--> [Analyzer Agent Workers]
        +--> [Validator Agent Workers]
        +--> [Synthesizer Agent Workers]
        +--> [Critic Agent Workers]
        +--> [Cost Optimizer Agent Workers]
        +--> [Memory Agent Service]

Data Plane:
  - PostgreSQL (OLTP, metadata, lineage)
  - Vector DB (Qdrant/Milvus) for ANN retrieval
  - Object Storage (S3/MinIO) for raw docs/artifacts
  - Redis for cache/session/short memory

LLM Plane:
  - Router (quality/cost/latency policy)
  - Providers: OpenAI + local vLLM/Ollama + fallback chain

Observability:
  - OpenTelemetry -> Prometheus/Grafana + Tempo/Jaeger + Loki
```

## 2.3 Component model

1. **Experience Plane**: Next.js app + server-side BFF routes for privileged operations.
2. **Control Plane**: API Gateway, Auth, tenant policy, rate limits, audit controls.
3. **Orchestration Plane**: graph-based state machine (durable), event emission, retries, compensation.
4. **Agent Runtime Plane**: specialized async workers, independent scaling by queue depth.
5. **Data Plane**: transactional store + vector retrieval + object artifact store + cache.
6. **LLM Runtime Plane**: provider abstraction, routing policy, failover, budget enforcement.

## 2.4 Data flow (ingestion → processing → reasoning → output)

1. **Ingestion**: Scout Agent pulls sources/webhooks → raw artifacts to S3/MinIO, metadata event emitted.  
2. **Processing**: Analyzer parses/chunks/classifies; embeddings generated and stored in Vector DB + lineage in Postgres.  
3. **Reasoning**: Synthesizer/Validator/Critic loop with RAG and constrained prompts, source-grounded.  
4. **Output**: signed response package (answer + citations + confidence + trace graph + cost report).  
5. **Feedback**: user/human-review outcomes to Memory Agent for policy and retrieval refinement.

---

## 3) MULTI-AGENT SYSTEM DESIGN

## 3.1 Agent catalog

1. **Scout Agent**
   - Role: collect and prioritize external/internal sources.
   - Tools: crawler connectors, WebSub clients, source scoring, dedup.
   - Async scaling: shard by source domain/region.

2. **Analyzer Agent**
   - Role: parse, normalize, chunk, classify, enrich metadata.
   - Tools: parser stack, OCR, language detector, PII scanner, schema validators.
   - Async scaling: batch workers by document type.

3. **Validator Agent**
   - Role: enforce data quality and factual consistency checks.
   - Tools: schema validators, citation verifier, contradiction checker.
   - Async scaling: policy-driven validation queues.

4. **Synthesizer Agent**
   - Role: generate task outputs (Q&A, summaries, reports).
   - Tools: RAG retriever, prompt templates, structured output adapters.
   - Async scaling: model/provider-aware worker pools.

5. **Memory Agent**
   - Role: short-term and long-term memory management.
   - Tools: Redis (short context), Postgres + Vector DB (long memory), memory compaction jobs.
   - Async scaling: background consolidation and eviction jobs.

6. **Orchestrator Agent**
   - Role: plan graph execution, assign tasks, monitor SLAs.
   - Tools: DAG/state engine, queue APIs, policy engine, compensation workflows.
   - Async scaling: leader election + stateless replicas.

7. **Critic Agent**
   - Role: adversarial critique, hallucination detection, revision suggestion.
   - Tools: cross-check prompts, entailment checks, source-grounding score.
   - Async scaling: selective activation by risk profile.

8. **Cost Optimizer Agent**
   - Role: token budget control, model routing optimization, caching policy tuning.
   - Tools: cost telemetry, routing policies, prompt compression, speculative decoding hooks.
   - Async scaling: periodic optimizer + real-time guardrails.

## 3.2 Interactions and autonomy loops

- Agents communicate via typed events (not direct tight RPC where avoidable).
- Orchestrator owns canonical workflow state; agents remain stateless workers.
- Memory Agent writes retrieval-ready summaries and episodic traces.
- Critic triggers self-reflection loop when confidence/citation score drops below threshold.
- Cost Optimizer can down-route model tier for low-risk tasks, up-route for critical tasks.

## 3.3 Self-improving loops

1. **Self-reflection loop**: Synthesizer → Critic → Synthesizer (max N iterations, policy bounded).  
2. **Retry strategy**:
   - retry same model with prompt repair,
   - fallback model/provider,
   - fallback to “insufficient evidence” response.
3. **Hallucination detection**:
   - citation coverage threshold,
   - entailment verification vs retrieved evidence,
   - unsupported-claim detector.
4. **Strict source citation**:
   - every generated claim references source chunk IDs/URLs,
   - response blocked when citation policy fails.

---

## 4) TECHNOLOGY STACK (2026)

## 4.1 Backend & orchestration
- **Python 3.12+**, FastAPI for control APIs.
- **Temporal or Dagster+event bridge** for durable workflow orchestration (stateful retries/timers).
- **NATS JetStream** (or Kafka if org standard) for event backbone.
- **Celery/Arq alternative only for narrow tasks**; primary should be event-driven orchestrator.

## 4.2 AI/LLM runtime
- Multi-provider abstraction:
  - OpenAI (high-quality complex reasoning paths),
  - vLLM for self-hosted throughput tiers,
  - Ollama for local/dev and edge prototypes.
- Router policy dimensions: quality target, latency SLO, max cost/request, data sensitivity.
- Guardrails: JSON schema constrained decoding + policy checks before publish.

## 4.3 Data layer
- **PostgreSQL 16+** for OLTP and lineage.
- **Qdrant or Milvus** for primary ANN/vector retrieval.
- **S3/MinIO** for immutable raw artifacts and run outputs.
- **Redis** for short-term memory, distributed locks, and hot cache.

## 4.4 Frontend
- **Next.js (App Router) + TypeScript + React**.
- BFF pattern for privileged endpoints.
- UX modules:
  - live pipeline DAG view,
  - agent health/status board,
  - reasoning trace + citations panel,
  - cost and latency observability widgets.

## 4.5 Infra & platform
- **Docker Compose** for local MVP parity.
- **Kubernetes** for production scaling.
- **GitHub Actions + ArgoCD** for CI/CD + GitOps deploy.
- **Prometheus + Grafana + OpenTelemetry + Loki/Tempo** for metrics/logs/traces.

---

## 5) IMPLEMENTATION ROADMAP (MVP → Scale)

## Phase 0 (2–3 weeks): Stabilize foundation
1. Externalize run state from in-memory manager to durable store.
2. Replace browser-exposed admin key pattern with BFF + server-side secrets.
3. Introduce queue backbone and move pipeline execution out of API process.
4. Define canonical event schemas and idempotency keys.

## Phase 1 (4–6 weeks): Agent-first MVP
1. Launch Orchestrator + Scout + Analyzer + Synthesizer + Validator.
2. Implement RAG with source citations as hard requirement.
3. Add basic Critic loop for high-risk tasks.
4. Add first version of cost policy routing.

## Phase 2 (6–10 weeks): Production hardening
1. Add Memory Agent with short/long-term memory strategy.
2. Introduce full observability with trace IDs across agents.
3. Add tenant isolation, RBAC, budget caps and policy enforcement.
4. Add disaster recovery, replay, and dead-letter processing.

## Phase 3 (10–16 weeks): Scale & moat
1. Adaptive routing based on historical quality/cost telemetry.
2. Self-optimizing prompt/routing policies (offline evaluation loop).
3. Verticalized domain packs (compliance/legal/finance workflows).
4. Benchmark harness and continuous regression suite for agent quality.

---

## 6) QUICK WINS (Immediate)

1. Move privileged key handling fully server-side (remove `NEXT_PUBLIC_ADMIN_API_KEY` usage).  
2. Replace API-spawned subprocess runs with queue worker jobs.  
3. Persist pipeline/scout run state in Postgres/Redis, not process memory.  
4. Add leader election/lease for scheduler and webhook subscription tasks.  
5. Define mandatory citation contract in API response schema (claim→source mapping).  
6. Split vector workload to dedicated vector DB for high-QPS retrieval paths.

---

## 7) ADVANCED IDEAS (Competitive Advantage)

1. **Evidence Graph Engine**
   - Build a signed claim-evidence graph per output.
   - Enables enterprise-grade auditability and compliance proofs.

2. **Adaptive Multi-Agent Routing**
   - Runtime policy learns optimal agent/model chain by task class and risk profile.
   - Minimizes cost while preserving quality SLAs.

3. **Autonomous Quality Governor**
   - Critic + Validator + Cost Optimizer triad decides when to revise, when to abstain, when to escalate to human.

4. **Synthetic Data Provenance Ledger**
   - Tamper-evident lineage for dataset generation/training artifacts.
   - Strong differentiation for regulated clients.

5. **Model-Agnostic Execution Fabric**
   - Pluggable provider routing with local/private/public model pools.
   - Avoids lock-in and enables geo/regulatory deployment flexibility.

---

## 8) Final conclusion

Current repository already contains many ingredients of a strong AI platform, but the operating model must move from **single-process orchestration** to **durable, event-driven, agent-native architecture**.

If the redesign above is implemented with discipline, the platform will become:
- autonomous,
- horizontally scalable,
- cost-governed,
- and significantly harder to replicate due to traceability + policy intelligence + multi-agent optimization.
