# ME Engineering Assistant

A multi-source RAG (Retrieval-Augmented Generation) agent for answering technical questions about ME Corporation's ECU product lines. Built with LangChain, LangGraph, FAISS, and MLflow, served via Docker REST API.

## Architecture

```
User Question
     │
     ▼
┌──────────┐
│ Classify │  Rule-based router: detect models, series, compare triggers
└────┬─────┘
     │
     ├── ECU_700 ──┐
     ├── ECU_800 ──┤
     │             ▼
     │    ┌─────────────────┐
     │    │ Retrieve Single │  FAISS similarity search + metadata filter
     │    └────────┬────────┘
     │             │
     ├── COMPARE ──┤
     ├── UNKNOWN ──┤
     │             ▼
     │    ┌─────────────────┐
     │    │ Retrieve Compare│  Full-document injection (docs are ~4KB total)
     │    └────────┬────────┘
     │             │
     ▼             ▼
┌────────────────┐
│ Check Evidence │  Is retrieved context sufficient?
└───────┬────────┘
        │
        ├── sufficient ─────────────────┐
        │                               ▼
        │                  ┌─────────────────────────┐
        │                  │ Validate Confidence     │  Flag low-confidence for
        │                  │ (Human-in-the-Loop)     │  human review via interrupt()
        │                  └────────────┬────────────┘
        │                               │
        └── insufficient                ▼
        │                  ┌──────────────┐
        ▼                  │  Synthesize  │  LLM generates answer with citations
┌───────────────┐          └──────┬───────┘
│ Rewrite Query │                │
└───────┬───────┘                ▼
        │                  Final Answer
        ▼
   (retry retrieval,
    max 2 attempts)
```

The agent uses a **hybrid retrieval** strategy with **multi-step reasoning**:
- **Single-source queries** (e.g., "RAM of ECU-850?"): FAISS vector search filtered by series/model metadata
- **Comparison queries** (e.g., "Compare all models"): Full-document injection — the 3 ECU docs total ~4KB, small enough to fit entirely in context
- **Evidence validation**: After retrieval, the agent checks if the context is sufficient. If not, it rewrites the query and retries (max 2 attempts).
- **Human-in-the-loop**: Low-confidence queries (unknown models, empty context) trigger a LangGraph `interrupt()` for human review before synthesis.

The graph uses **deterministic rule-based routing** rather than LLM-driven tool selection. This achieves 100% routing accuracy on the test set and avoids an extra LLM call per query. LangChain tool wrappers were prototyped but removed in favor of this more reliable approach.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Ollama + Mistral 7B | Local, open-source, fits Docker, no API keys needed |
| Embedding | `all-MiniLM-L6-v2` | Lightweight (80MB), CPU-friendly, good for technical text |
| Vector Store | Single FAISS index | In-memory, fast, metadata filtering for series/model |
| Router | Rule-based (regex) | Benchmarked against LLM: 100% vs 90% accuracy, <1ms vs ~1.2s latency, 100% vs 93% consistency |
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
│       ├── metrics.py          # Keyword accuracy, routing, source metrics + LLM-as-Judge
│       └── evaluate.py         # Evaluation runner (keyword + LLM judge)
├── scripts/
│   ├── ingest.py               # CLI: build FAISS index
│   ├── evaluate.py             # CLI: run evaluation (keyword + LLM judge)
│   ├── benchmark_routing.py    # Routing strategy A/B benchmark (regex vs LLM)
│   ├── benchmark_full.py       # Full pipeline 4-strategy benchmark
│   └── entrypoint.sh           # Docker entrypoint
├── benchmarks/
│   └── full_pipeline_benchmark_report.md  # Detailed experiment report
└── tests/
    ├── test_router.py          # Router correctness (16 tests)
    ├── test_retrieval.py       # Retrieval filtering (6 tests)
    ├── test_graph.py           # Node-level integration (4 tests)
    ├── test_integration.py     # Full pipeline with mocked LLM (3 tests)
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

10 test questions covering single-source lookup, cross-series comparison, feature availability, and configuration queries. Evaluated with **two complementary methods**:

- **Keyword matching** — deterministic, requires exact technical values (e.g., "+85°C", "LPDDR4")
- **LLM-as-Judge** — uses `Evaluation_Criteria` from the test CSV as a grading rubric, scores 1-5

| Q# | Category | Route | Keyword | Judge |
|----|----------|-------|---------|-------|
| 1 | Single Source - ECU-700 | ECU_700 | FAIL | 5/5 |
| 2 | Single Source - ECU-800 | ECU_800 | PASS | 5/5 |
| 3 | Single Source - ECU-800 Enhanced | ECU_800 | PASS | 5/5 |
| 4 | Comparative - Same Series | COMPARE | PASS | 5/5 |
| 5 | Comparative - Cross Series | COMPARE | PASS | 4/5 |
| 6 | Technical Specification | ECU_800 | FAIL | 5/5 |
| 7 | Feature Availability | COMPARE | PASS | 5/5 |
| 8 | Storage Comparison | COMPARE | PASS | 5/5 |
| 9 | Operating Environment | COMPARE | FAIL | 4/5 |
| 10 | Configuration/Usage | ECU_800 | PASS | 5/5 |

| Metric | Score |
|--------|-------|
| Keyword accuracy | 7/10 |
| LLM Judge (avg) | 4.8/5 |
| Routing accuracy | 10/10 |
| Source accuracy | 10/10 |
| Avg latency | ~12s (GPU) |
| Pylint score | 9.95/10 |
| Unit tests | 31/31 passed |

**Why the two metrics diverge:** Q1 and Q6 produce factually correct answers but use slightly different formatting (e.g., "85 degrees Celsius" instead of "+85°C"), causing keyword FAIL but judge 5/5. The dual evaluation avoids both false negatives (keyword-only) and false positives (judge-only).

### Routing Strategy Benchmark

To validate the rule-based routing decision, we benchmarked regex routing against LLM routing (Mistral 7B classification) on 15 questions (10 standard + 5 edge cases), each run 3 times:

| Metric | Regex Router | LLM Router |
|--------|-------------|------------|
| Accuracy | 15/15 (100%) | 14/15 (93%) |
| Avg Latency | <1 ms | 1,200 ms |
| Consistency (3 runs) | 100% | 93% |

The LLM router consistently misroutes Q9 ("Which ECU can operate in the harshest temperature conditions?") — a superlative query requiring comparison across all models. The regex router catches this via explicit superlative keyword patterns (`harshest`, `highest`, `best`, etc.). The LLM also shows inconsistency on Q4 across repeated runs.

**Conclusion:** Regex routing is strictly better for this domain — higher accuracy, deterministic behavior, and 1000x lower latency.

*Script: `scripts/benchmark_routing.py --runs 3`*

### Full Pipeline Benchmark

To quantify the cost of replacing each rule-based component with an LLM alternative, we ran 4 strategy combinations on the 10 standard questions:

| Strategy | Routing | Query Rewrite | Route Acc | Answer Acc | Avg Latency | Overhead |
|----------|---------|---------------|-----------|------------|-------------|----------|
| **A (current)** | Regex | Keyword | **10/10** | 6/10 | **10.3s** | baseline |
| B | LLM | Keyword | 9/10 | 6/10 | 13.3s | +3.1s |
| C | Regex | LLM | **10/10** | 7/10 | 16.2s | +5.9s |
| D (full LLM) | LLM | LLM | 9/10 | 8/10 | 23.4s | +13.1s |

Average per-stage latency breakdown:

| Stage | A: Regex+Keyword | D: LLM+LLM |
|-------|-----------------|-------------|
| Route | <1 ms | 4,190 ms |
| Retrieve | 220 ms | 144 ms |
| Rewrite | <1 ms | 3,263 ms |
| **Synthesize** | **10,042 ms** | **15,756 ms** |

**Key findings:**
1. **Synthesis dominates latency** (78-97% of total time) — routing and rewrite overhead is comparatively small
2. **LLM rewrite improves answer accuracy** (Strategy C: +1 over A) at +6s cost
3. **Strategy A provides the best latency/accuracy trade-off** for the 20s SLA requirement
4. **Answer accuracy is bounded by synthesis quality**, not routing or retrieval

*Script: `scripts/benchmark_full.py` | Full report: `benchmarks/full_pipeline_benchmark_report.md`*

## Testing & Validation Strategy

### Automated Testing

The test suite follows a 4-tier testing pyramid, all runnable offline without an LLM:

| Layer | Test File | Count | What It Validates |
|-------|-----------|-------|-------------------|
| Unit | `test_router.py` | 16 | All 10 test questions route correctly, plus edge cases (case insensitivity, 850b vs 850 partial match, superlative + single model) |
| Retrieval | `test_retrieval.py` | 6 | Metadata filtering by series/model, no cross-series contamination, full-doc injection for COMPARE |
| Integration | `test_graph.py` | 4 | LangGraph node functions (classify, retrieve_single, retrieve_compare) with real FAISS index |
| Integration | `test_integration.py` | 3 | Full graph pipeline end-to-end with mocked LLM, verifying state flow through all nodes |
| Schema | `test_predict.py` | 2 | MLflow PythonModel interface: class instantiation and input schema validation |

Run all tests: `make test` (requires no Ollama or GPU — all LLM calls are mocked where needed).

### Domain Expertise Validation

The evaluation framework (`eval/`) validates the agent against a **golden dataset** of 10 domain-expert-curated questions (`data/test-questions.csv`). Each question has:

- **Expected keywords**: Specific technical values that must appear in the answer (e.g., "+85°C", "LPDDR4", "5 TOPS"). A question passes only if ALL required keywords are found.
- **Expected route**: The correct classification (ECU_700, ECU_800, COMPARE) — validates the agent consults the right product line.
- **Expected sources**: The specific document file(s) the answer should be derived from — validates source tracing.
- **Evaluation criteria**: Natural-language description of what makes a correct answer (e.g., "Accurate comparison; Synthesis of multiple specifications"). Used by the LLM-as-Judge scorer.

The system uses **dual evaluation** to balance strictness and semantic understanding:

| Method | Type | Strengths | Weaknesses |
|--------|------|-----------|------------|
| Keyword matching | Deterministic | Reproducible, no false positives, catches missing facts | False negatives on format differences ("+85°C" vs "85 degrees") |
| LLM-as-Judge | Semantic | Understands equivalent expressions, evaluates explanation quality | Non-deterministic, may be lenient |

Both methods are integrated into `mlflow.evaluate()` as custom metrics (`answer_accuracy` and `llm_judge_score`).

**Validation results across all four dimensions:**

| Dimension | Method | Result | Notes |
|-----------|--------|--------|-------|
| Answer accuracy (keyword) | Exact keyword matching | 7/10 (70%) | Q1, Q6, Q9 fail on format differences, not factual errors |
| Answer accuracy (judge) | LLM-as-Judge (1-5) | 4.8/5 avg | 8 questions score 5/5; Q5 and Q9 score 4/5 (minor omissions) |
| Routing correctness | Route vs expected | 10/10 (100%) | All questions routed to correct product line (ECU_700, ECU_800, COMPARE) |
| Source correctness | Source docs vs expected | 10/10 (100%) | All answers cite the correct source documents |

The gap between keyword (70%) and judge (96%) accuracy confirms the dual evaluation design: keyword matching catches missing facts while LLM judge avoids false negatives from formatting differences.

**Extending the benchmark**: To add new test questions, define the question, required keywords, expected route, expected sources, and evaluation criteria in `test-questions.csv` and `eval/metrics.py`. No model retraining needed.

### Continuous Validation & Production Monitoring

Every evaluation run is tracked in **MLflow** (`me-assistant-evaluation` experiment), logging:

| Metric | Description |
|--------|-------------|
| `overall_accuracy` | Fraction of questions where all required keywords are present |
| `overall_routing_accuracy` | Fraction of questions routed to the correct retrieval path |
| `overall_source_accuracy` | Fraction of questions citing the correct source documents |
| `overall_avg_judge_score` | LLM-as-Judge average score (1-5 scale) |
| `overall_avg_latency_ms` | Mean response time across all questions |
| `overall_p95_latency_ms` | 95th percentile response time |

Each run also logs two **artifacts** for full traceability:

| Artifact | Contents |
|----------|----------|
| `evaluation_results.csv` | Complete per-question results: answer text, route, sources, keyword pass/fail, judge score (1-5), judge reasoning |
| `evaluation_summary.json` | Aggregate scores + per-question summary with `judge_score` and `judge_reason` for every question |

This ensures every evaluation is fully reproducible — not just the aggregate numbers, but the LLM judge's rationale for each individual score.

**Regression detection**: After any change to prompts, retrieval logic, or model configuration, re-run `make eval-mlflow` and compare metrics against the baseline run in MLflow. A drop in accuracy or spike in latency signals a regression before deployment.

**Production monitoring strategy** (see Scalability Roadmap Phase 3 for full plan):
- **Latency alerting**: Flag queries exceeding the 20s SLA threshold
- **Confidence-based routing**: Low-confidence queries (route=UNKNOWN, missing context) are flagged for human review via the LangGraph interrupt mechanism
- **Accuracy drift detection**: Periodic re-evaluation against the golden dataset to detect model degradation over time

## Limitations

- **Small document set**: Only 3 ECU documents (~4KB total). The hybrid retrieval strategy is optimized for this scale.
- **Rule-based router**: Relies on keyword/regex patterns. May not handle novel query phrasings or new product lines without pattern updates.
- **Single LLM**: Uses Mistral 7B locally. Latency is ~10-15s per query (GPU) or ~20-30s (CPU), dominated by LLM synthesis.
- **No conversation memory**: Each query is independent — no multi-turn dialogue support.
- **Evaluation scope**: Keyword matching can produce false negatives on format differences (e.g., "85 degrees C" vs "+85°C"). LLM-as-Judge compensates but is non-deterministic.

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
| **Latency** | ~10-15s per query on GPU (LLM synthesis-dominated) |
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
  ├─ accuracy >= baseline   → auto-approve
  └─ accuracy < baseline    → block merge, flag regression
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
| **0 (current)** | 3 docs, 1 user | ~10-15s (GPU) | FAISS, Ollama, MLflow | Done |
| **1 (data)** | 500 docs | <15s | Qdrant, incremental indexing | 2-3 weeks |
| **2 (concurrency)** | 50 users | <5s (cached) | Redis, FastAPI, Ollama replicas | 3-4 weeks |
| **3 (enterprise)** | Multi-team | SLA-based | OpenTelemetry, RBAC, model routing | 8-12 weeks |
