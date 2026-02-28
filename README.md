# ME Engineering Assistant

A multi-source RAG (Retrieval-Augmented Generation) agent for answering technical questions about ME Corporation's ECU product lines. Built with LangChain, LangGraph, FAISS, and MLflow, served via Docker REST API.

## Architecture

```
User Question
     │
     ▼
┌──────────┐
│ Classify  │  Rule-based router: detect models, series, compare triggers
└────┬─────┘
     │
     ├── ECU_700 ──┐
     ├── ECU_800 ──┤
     │             ▼
     │    ┌─────────────────┐
     │    │ Retrieve Single  │  FAISS similarity search + metadata filter
     │    └────────┬────────┘
     │             │
     ├── COMPARE ──┤
     ├── UNKNOWN ──┤
     │             ▼
     │    ┌─────────────────┐
     │    │ Retrieve Compare │  Full-document injection (docs are ~4KB total)
     │    └────────┬────────┘
     │             │
     ▼             ▼
     ┌──────────────┐
     │  Synthesize   │  LLM generates answer with source citations
     └──────┬───────┘
            │
            ▼
      Final Answer
```

The agent uses a **hybrid retrieval** strategy:
- **Single-source queries** (e.g., "RAM of ECU-850?"): FAISS vector search filtered by series/model metadata
- **Comparison queries** (e.g., "Compare all models"): Full-document injection — the 3 ECU docs total ~4KB, small enough to fit entirely in context

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Ollama + Mistral 7B | Local, open-source, fits Docker, no API keys needed |
| Embedding | `all-MiniLM-L6-v2` | Lightweight (80MB), CPU-friendly, good for technical text |
| Vector Store | Single FAISS index | In-memory, fast, metadata filtering for series/model |
| Router | Rule-based (regex) | Deterministic, explainable, 10/10 accuracy on test set |
| Chunking | Markdown header splitting | Preserves section semantics, tables kept intact |
| Comparison strategy | Full-doc injection | Docs are tiny — similarity search would lose context |

## Project Structure

```
├── pyproject.toml              # Dependencies and project config
├── Makefile                    # Build/run targets
├── Dockerfile                  # Container image
├── docker-compose.yml          # Ollama + RAG service
├── data/
│   ├── docs/                   # ECU specification documents (3 markdown files)
│   ├── faiss_index/            # Pre-built FAISS index
│   └── test-questions.csv      # 10 evaluation questions with expected answers
├── src/me_assistant/
│   ├── config.py               # Central configuration
│   ├── ingest/
│   │   ├── loader.py           # Load markdown docs with metadata
│   │   ├── splitter.py         # Chunk by headers, normalize bold titles
│   │   └── indexer.py          # Build/save/load FAISS index
│   ├── retrieval/
│   │   └── retriever.py        # Series/model filtered search + all-docs retrieval
│   ├── agent/
│   │   ├── state.py            # LangGraph state schema (TypedDict)
│   │   ├── router.py           # Rule-based query classifier (4 routes)
│   │   ├── prompts.py          # Single-source and comparison prompt templates
│   │   ├── nodes.py            # LangGraph node functions
│   │   └── graph.py            # StateGraph construction and compilation
│   ├── model/
│   │   ├── pyfunc.py           # MLflow PythonModel wrapper
│   │   └── log.py              # Log model to MLflow with artifacts
│   └── eval/
│       ├── metrics.py          # Keyword-based accuracy, routing, source metrics
│       └── evaluate.py         # Evaluation runner
├── scripts/
│   ├── ingest.py               # CLI: build FAISS index
│   ├── evaluate.py             # CLI: run evaluation
│   └── entrypoint.sh           # Docker entrypoint
└── tests/
    ├── test_router.py          # Router correctness (17 tests)
    ├── test_retrieval.py       # Retrieval filtering (6 tests)
    ├── test_graph.py           # Node-level integration (4 tests)
    └── test_predict.py         # MLflow schema validation (2 tests)
```

## Setup & Run

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- Mistral 7B model pulled: `ollama pull mistral:7b`

### Local Development

```bash
# Install dependencies
make install

# Build FAISS index from documents
make ingest

# Log model to MLflow
make log-model

# Run tests
make test

# Run pylint
make lint

# Run evaluation (requires Ollama running)
make eval
```

### Docker

```bash
# Build and start both Ollama and RAG service
docker compose up --build

# Query the API
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["question"], "data": [["What is the RAM of ECU-850?"]]}}'
```

## Evaluation Results

10 test questions covering single-source lookup, cross-series comparison, feature availability, and configuration queries.

| Q# | Category | Route | Result |
|----|----------|-------|--------|
| 1 | Single Source - ECU-700 | ECU_700 | PASS |
| 2 | Single Source - ECU-800 | ECU_800 | PASS |
| 3 | Single Source - ECU-800 Enhanced | ECU_800 | PASS |
| 4 | Comparative - Same Series | COMPARE | PASS |
| 5 | Comparative - Cross Series | COMPARE | PASS |
| 6 | Technical Specification | ECU_800 | PASS |
| 7 | Feature Availability | COMPARE | PASS |
| 8 | Storage Comparison | COMPARE | PASS |
| 9 | Operating Environment | COMPARE | PASS* |
| 10 | Configuration/Usage | ECU_800 | PASS |

**Score: 9/10** (Q9 occasionally varies depending on LLM interpretation of "harshest")

- **Routing accuracy:** 10/10
- **Pylint score:** 9.95/10
- **Unit tests:** 28/28 passed

## Testing Strategy

- **Unit tests** (`test_router.py`): All 10 test questions route correctly, plus edge cases (case insensitivity, partial matches, superlative + single model)
- **Retrieval tests** (`test_retrieval.py`): Metadata filtering verified — no cross-series contamination
- **Integration tests** (`test_graph.py`): Individual LangGraph nodes tested (classify, retrieve_single, retrieve_compare)
- **Schema tests** (`test_predict.py`): MLflow model interface validation
- **Evaluation framework** (`eval/`): Keyword-based answer accuracy checking against expected values from test CSV

## Limitations

- **Small document set**: Only 3 ECU documents (~4KB total). The hybrid retrieval strategy is optimized for this scale.
- **Rule-based router**: Relies on keyword/regex patterns. May not handle novel query phrasings or new product lines without pattern updates.
- **Single LLM**: Uses Mistral 7B locally. Latency is ~12-18s per query depending on hardware.
- **No conversation memory**: Each query is independent — no multi-turn dialogue support.
- **Keyword-based evaluation**: Answer accuracy is checked via keyword matching, not semantic similarity.

## Scalability Strategy

### Overview

The current system is a single-node PoC optimized for 3 small documents and batch queries. Below is a phased roadmap for scaling to enterprise production, addressing data volume, concurrency, and organizational growth.

### Phase 0: Current State (Single-Node PoC)

| Aspect | Current Implementation |
|--------|----------------------|
| **Documents** | 3 markdown files (~4KB total) |
| **Vector store** | In-memory FAISS, rebuilt on startup |
| **LLM** | Single Ollama instance, synchronous calls |
| **Serving** | MLflow `models serve`, single worker |
| **Latency** | ~12-18s per query (LLM-dominated) |
| **Throughput** | ~3-4 queries/minute (sequential) |

### Phase 1: Data Scale (100s of Documents)

**Goal:** Support 100-500 ECU documents across multiple product lines without redeployment when new docs arrive.

**Changes:**

| Component | Current → New | Files Affected |
|-----------|--------------|----------------|
| Vector store | FAISS (in-memory) → **Qdrant** (persistent, filtered search) | `config.py`, `ingest/indexer.py`, `retrieval/retriever.py` |
| Indexing | Full rebuild on startup → **Incremental indexing pipeline** with document hashing | `ingest/indexer.py`, new `ingest/watcher.py` |
| Chunking | Header-based splitting → **Adaptive chunking** with table-aware parsing, OCR for scanned PDFs | `ingest/splitter.py`, `ingest/loader.py` |
| Metadata | Simple series/model tags → **Hierarchical taxonomy** (product line → series → model → revision) | `config.py`, `ingest/loader.py` |
| Router | Static regex patterns → **Configurable pattern registry** loaded from YAML, auto-discovery of new model names from ingested docs | `agent/router.py`, new `config/routes.yaml` |

**Technology choices:**
- **Qdrant** over Milvus/Pinecone: self-hosted, native metadata filtering, gRPC API, built-in sharding. Runs as a Docker sidecar alongside Ollama.
- **Document hashing** (SHA-256 of content): skip re-indexing unchanged docs, detect modified docs for re-embedding.

**Key metrics:**
- Index build time for N documents (target: <5 min for 500 docs)
- Retrieval latency with filtered search (target: <500ms p95)
- Storage footprint per document (track embedding + metadata overhead)

**Estimated effort:** 2-3 weeks

### Phase 2: Concurrency (10+ Concurrent Users)

**Goal:** Handle 10-50 concurrent users with sub-5s response times for cached/repeated queries.

**Changes:**

| Component | Current → New | Files Affected |
|-----------|--------------|----------------|
| Serving | MLflow single-worker → **FastAPI with async handlers** + Uvicorn workers | New `api/server.py`, `api/routes.py` |
| LLM calls | Synchronous urllib → **Async httpx** with connection pooling, retry logic | `agent/nodes.py` |
| Caching | None → **Redis** for embedding cache + response cache (TTL-based) | New `cache/redis_client.py`, `retrieval/retriever.py` |
| Load balancing | Single Ollama → **Multiple Ollama replicas** behind nginx or Kubernetes service | `docker-compose.yml`, new `nginx.conf` |
| Queue | None → **Celery/RQ task queue** for LLM requests to prevent overload | New `worker/tasks.py` |

**Architecture:**
```
                    ┌─────────────┐
Client ──▶ FastAPI ─┤ Redis Cache  │ (hit → return cached answer)
                    └──────┬──────┘
                           │ miss
                    ┌──────▼──────┐
                    │  Task Queue  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Ollama #1    Ollama #2    Ollama #3
```

**Caching strategy:**
- **Embedding cache**: Cache query → embedding vector in Redis. Same query hits cache instead of re-computing embeddings. TTL: 24 hours.
- **Response cache**: Cache (question_hash, route) → answer. Exact-match queries return instantly. TTL: 1 hour (documents may update).
- **Semantic cache** (future): Use embedding similarity to match "close enough" queries to cached answers.

**Key metrics:**
- Concurrent request throughput (target: 50 req/min sustained)
- p95 latency (target: <5s for cache hits, <15s for cache misses)
- Cache hit rate (target: >30% for production workloads)
- Ollama GPU utilization across replicas

**Estimated effort:** 3-4 weeks

### Phase 3: Enterprise (Multi-Team, Multi-Product)

**Goal:** Production-grade system supporting multiple product lines, teams, and compliance requirements.

**Changes:**

| Component | Current → New | Files Affected |
|-----------|--------------|----------------|
| Vector store | Single collection → **Per-product-line shards** (ECU, sensors, actuators) | `retrieval/retriever.py`, `config.py` |
| Access control | None → **RBAC** via JWT tokens, team-scoped document access | New `auth/` module |
| Model routing | Single LLM → **Complexity-based routing** (simple → small model, complex → large model) | `agent/router.py`, `agent/nodes.py` |
| Observability | Python logging → **OpenTelemetry** traces + **LangSmith** for LLM observability | All agent modules, new `telemetry/` module |
| Evaluation | Manual CLI → **Automated CI/CD evaluation** with regression detection | `.github/workflows/`, `eval/` |
| A/B testing | None → **Experiment framework** for prompt/model/retrieval variants | New `experiments/` module, `eval/evaluate.py` |

**Model complexity routing:**
```python
# Route queries by estimated complexity
if is_simple_lookup(query):       # "RAM of ECU-850?"
    model = "phi3:mini"           # Fast, small model (~2s)
elif is_comparison(query):        # "Compare all models"
    model = "mistral:7b"          # Medium model (~12s)
elif is_reasoning(query):         # "Why would I choose 850b over 850?"
    model = "llama3:70b"          # Large model, high quality (~30s)
```

**Observability stack:**
- **OpenTelemetry**: Distributed traces across API → retrieval → LLM, with span attributes for route, model, latency.
- **LangSmith**: LLM-specific observability — prompt/response pairs, token usage, quality scoring.
- **Grafana dashboards**: Real-time metrics — latency percentiles, error rates, cache hit ratios, accuracy trends.
- **Alerting**: PagerDuty alerts for latency spikes (>30s p95), accuracy drops (below 80% on eval suite), Ollama health failures.

**CI/CD evaluation pipeline:**
```
PR opened → build → run eval suite → compare accuracy to main branch
  ├─ accuracy >= baseline   → ✅ auto-approve
  └─ accuracy < baseline    → ❌ block merge, flag regression
```

**Key metrics:**
- Per-team query volume and latency SLAs
- Model routing distribution (% queries per model tier)
- End-to-end trace duration breakdown (embedding / retrieval / LLM / total)
- Evaluation accuracy trend over time (regression detection)
- Cost per query (compute + LLM inference)

**Estimated effort:** 8-12 weeks

### Summary

| Phase | Scale | Latency Target | Key Technology | Effort |
|-------|-------|---------------|----------------|--------|
| **0 (current)** | 3 docs, 1 user | <20s | FAISS, Ollama, MLflow | Done |
| **1 (data)** | 500 docs | <15s | Qdrant, incremental indexing | 2-3 weeks |
| **2 (concurrency)** | 50 users | <5s (cached) | Redis, FastAPI, Ollama replicas | 3-4 weeks |
| **3 (enterprise)** | Multi-team | SLA-based | OpenTelemetry, RBAC, model routing | 8-12 weeks |
