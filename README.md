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

## Future Work / Scalability

| Phase | Scope | Key Changes |
|-------|-------|-------------|
| Phase 1: Data Scale | 100s of documents | Persistent vector DB (Qdrant/Milvus), incremental indexing, metadata sharding per product line |
| Phase 2: Concurrency | 10+ concurrent users | Embedding cache (Redis), LLM request queue, horizontal scaling, response caching |
| Phase 3: Enterprise | Multi-team, multi-product | Per-product vector store shards, RBAC, model complexity routing, observability (OpenTelemetry), A/B testing |
