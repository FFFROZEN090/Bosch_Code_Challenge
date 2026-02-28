# ME Engineering Assistant - Refined Implementation Plan

## Context

Build a multi-tool AI agent ("ME Engineering Assistant") for a Bosch code challenge. The agent answers questions about ECU-700 and ECU-800 series specifications using RAG + LangGraph + MLflow, served via Docker REST API. Must pass 8/10 test queries, <20s latency, pylint >85%. Optional challenges: MLflow eval, HITL, scalability, advanced agent behaviors.

---

## Critical Insight: Documents Are Tiny (~90 lines total)

The 3 ECU docs total ~4KB. This **fundamentally changes the architecture**:
- For comparison/multi-model queries (5/10 test questions), passing ALL docs as LLM context is **optimal, not a fallback**
- FAISS is still built (spec requires vector storage) but primarily serves single-source queries
- The challenge's own spec acknowledges this: *"Since documents are relatively small, you can implement a fallback strategy that passes document content directly as context"*

**Architecture: Hybrid retrieval** — FAISS for targeted single-model lookups, full-doc injection for comparison/all-model queries.

---

## Critique of Original Plan (Key Issues Fixed)

| # | Original Plan Issue | Fix in This Plan |
|---|---|---|
| 1 | LLM/Embedding choice unspecified | Concrete choice: Ollama + Mistral 7B (or qwen2.5) + `all-MiniLM-L6-v2` embeddings |
| 2 | Router misses implicit comparisons (Q7 "which models", Q8 "all models", Q9 superlatives) | Extended keyword patterns: `all models`, `which.*models`, `across`, superlatives (`harshest`, `best`, `most`), `every` |
| 3 | LangChain barely mentioned (spec requires it) | Explicitly use LangChain: `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `FAISS` |
| 4 | Stage 4 "baseline without LLM" wastes time | Eliminated. Go directly to LLM synthesis |
| 5 | Wrong document filenames | Corrected: `ECU-700_Series_Manual.md`, `ECU-800_Series_Base.md`, `ECU-800_Series_Plus.md` |
| 6 | Two FAISS stores can't handle intra-series comparison (Q4: 850 vs 850b) | Single FAISS store with metadata filtering (series + model fields) |
| 7 | Table handling vague | Specific strategy: parse markdown tables into "fact chunks" (one chunk per spec table, preserve header+all rows) |
| 8 | MLflow artifacts undefined | Explicit: bundle FAISS index, raw docs, config.json; embedding model loaded via pip deps |
| 9 | No prompt engineering | Two specific prompts: SINGLE_SOURCE_PROMPT and COMPARISON_PROMPT |
| 10 | No test question analysis | Each question mapped to route + retrieval strategy below |
| 11 | 11 stages over-engineered for 8-10 hrs | Consolidated to 6 stages (core) + 4 stages (optional) |
| 12 | UNKNOWN route triggers HITL (wrong for core) | UNKNOWN defaults to ALL_DOCS retrieval (search everything) |

---

## Test Question → Architecture Mapping

| Q# | Route | Retrieval Strategy | Key Challenge |
|----|-------|-------------------|---------------|
| 1 | ECU_700 | FAISS filter series=700 | Exact temp value |
| 2 | ECU_800 | FAISS filter series=800, model=ECU-850 | Exact RAM value |
| 3 | ECU_800 | FAISS filter series=800, model=ECU-850b | NPU description |
| 4 | COMPARE | All 800-series docs (both Base + Plus) | Intra-series diff |
| 5 | COMPARE | All docs (700 + 800) | Cross-series CAN comparison |
| 6 | ECU_800 | FAISS filter model=ECU-850b | Power values (load vs idle) |
| 7 | COMPARE | **All docs** — must confirm negative (750 NO OTA) | Negative confirmation critical |
| 8 | COMPARE | **All docs** — need all 3 models | Complete enumeration |
| 9 | COMPARE | **All docs** — superlative requires full comparison | Superlative → compare route |
| 10 | ECU_800 | FAISS filter model=ECU-850b | Exact command syntax |

**Conclusion:** 5 questions need single-source, 5 need all-docs. Router accuracy is make-or-break.

---

## Technology Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| **LLM** | Ollama + `mistral:7b` (or `qwen2.5:7b`) | Local, open-source, fits Docker, <20s on modern hardware. Cloud API (OpenAI) as documented alternative. |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` | Local, small (80MB), good for technical text, runs on CPU |
| **Vector Store** | FAISS (single index, metadata filtering) | Lightweight, in-memory, LangChain integration |
| **LangChain** | Document loaders, text splitters, embeddings, FAISS wrapper | Required by spec, provides clean abstractions |
| **LangGraph** | StateGraph with conditional routing | Required by spec |
| **MLflow** | Custom PythonModel, pyfunc logging | Required by spec |

---

## LLM Provider: Ollama + Local Model (Decided)

- **Primary:** Ollama with `mistral:7b` (or `qwen2.5:7b` if Mistral quality insufficient)
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (local, CPU-friendly, 80MB)
- **Docker:** Ollama runs as sidecar service in docker-compose
- **Latency budget:** ~15s target (embedding ~1s, FAISS ~0.1s, LLM generation ~12-14s)
- **Design:** LLM wrapped behind LangChain `ChatOllama` — swappable to `ChatOpenAI` via config for development speed if needed

---

## Project Structure

```
me-engineering-assistant/
├── pyproject.toml
├── .pylintrc
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── README.md
├── data/
│   ├── docs/
│   │   ├── ECU-700_Series_Manual.md
│   │   ├── ECU-800_Series_Base.md
│   │   └── ECU-800_Series_Plus.md
│   └── test-questions.csv
├── src/me_assistant/
│   ├── __init__.py
│   ├── config.py                  # Central config (paths, model names, thresholds)
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── loader.py              # Load markdown docs, attach metadata
│   │   ├── splitter.py            # Chunking: section + fact chunks
│   │   └── indexer.py             # Build FAISS index, save to disk
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py           # retrieve_by_model(), retrieve_by_series(), retrieve_all()
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py               # TypedDict for LangGraph state
│   │   ├── router.py              # Rule-based query classifier
│   │   ├── prompts.py             # Synthesis + comparison prompt templates
│   │   ├── nodes.py               # LangGraph node functions (route, retrieve, synthesize)
│   │   └── graph.py               # Build and compile the LangGraph StateGraph
│   ├── model/
│   │   ├── __init__.py
│   │   ├── pyfunc.py              # MLflow PythonModel subclass
│   │   └── log.py                 # Script: log model to MLflow
│   └── eval/
│       ├── __init__.py
│       ├── evaluate.py            # MLflow evaluation runner
│       └── metrics.py             # Custom metrics (accuracy, routing, latency)
├── tests/
│   ├── test_router.py
│   ├── test_retrieval.py
│   ├── test_graph.py
│   └── test_predict.py
└── scripts/
    ├── ingest.py                  # CLI: build index
    ├── serve.py                   # CLI: start MLflow serving
    └── evaluate.py                # CLI: run evaluation
```

---

## Implementation Stages

### Stage 1: Project Skeleton + Document Ingestion + Vector Store
**Files:** `pyproject.toml`, `.pylintrc`, `config.py`, `ingest/*`, `retrieval/*`

**1a. Project setup**
- `pyproject.toml` with pinned dependencies: `langchain`, `langchain-community`, `langgraph`, `mlflow`, `faiss-cpu`, `sentence-transformers`, `langchain-ollama` (or `langchain-openai`)
- `.pylintrc` configured for >85% target
- `Makefile` with targets: `install`, `ingest`, `log-model`, `serve`, `eval`, `lint`, `test`

**1b. Document loading** (`ingest/loader.py`)
- Read each `.md` file, attach metadata: `{series: "700"|"800", model: "ECU-750"|"ECU-850"|"ECU-850b", source_file: "..."}`
- Store raw full-text per document (needed for comparison queries)
- **Important:** ECU-700 line 18 has a malformed table row (missing leading `|`). Handle gracefully.

**1c. Chunking strategy** (`ingest/splitter.py`)
- Use LangChain's `MarkdownHeaderTextSplitter` to split by headers first
- For each section, keep specification tables INTACT as single chunks (they're small enough)
- Attach inherited metadata (series, model, section_path) to every chunk
- Also create a **"full document" chunk** per file for the all-docs retrieval path
- Expected output: ~10-15 chunks total (docs are tiny)

**1d. Vector store** (`ingest/indexer.py`, `retrieval/retriever.py`)
- Use LangChain's `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`
- Build single FAISS index from all chunks via `FAISS.from_documents()`
- Save index to `data/faiss_index/`
- Retriever functions:
  - `retrieve_by_series(query, series)` → FAISS search with metadata filter
  - `retrieve_by_model(query, model)` → FAISS search with metadata filter
  - `retrieve_all_docs()` → return raw full-text of all 3 documents (no similarity search needed)

**Verification:** Run `python scripts/ingest.py`, check index loads, test a few queries manually.

---

### Stage 1.5: LLM & Embedding 环境部署（Stage 2 的前置依赖）

Stage 2 的 `synthesize_node` 和最终验证都需要 LLM 运行。本步骤确保所有模型服务就绪。

**1.5a. Ollama 安装与启动（本地开发环境）**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama 服务（后台守护进程）
ollama serve
```

**1.5b. 拉取 LLM 模型**

```bash
# 主选模型（~4.1GB，首次需要几分钟下载）
ollama pull mistral:7b

# 备选：如果 Mistral 回答质量不够或机器内存不足
ollama pull qwen2.5:7b      # 备选 A：中文+英文都强
ollama pull phi3:mini        # 备选 B：轻量（2.3GB），适合低配机器
```

**1.5c. 验证 Ollama 服务可用**

```bash
# 检查服务是否在运行
curl http://localhost:11434

# 检查模型是否已下载
ollama list

# 快速测试 LLM 能否响应
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:7b",
  "prompt": "What is 2+2?",
  "stream": false
}'
```

**1.5d. Embedding 模型（自动处理）**

`sentence-transformers/all-MiniLM-L6-v2` 会在 Stage 1d 首次调用 `HuggingFaceEmbeddings()` 时自动从 HuggingFace 下载到本地缓存（~80MB）。无需手动操作，但**首次运行需要网络连接**。

如需离线使用，可预先下载：
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**1.5e. 代码中的连接配置** (`config.py`)

```python
# LLM 配置 — 后续 Stage 2 synthesize_node 和 Stage 4 Docker 都依赖这里
LLM_PROVIDER = "ollama"                          # "ollama" | "openai"
OLLAMA_BASE_URL = "http://localhost:11434"        # 本地开发
OLLAMA_MODEL = "mistral:7b"                       # Docker 中会切为 http://ollama:11434
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LangChain 初始化示例（供后续 Stage 引用）
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
```

**1.5f. 硬件需求与故障排查**

| 场景 | 最低配置 | 建议配置 |
|------|---------|---------|
| Mistral 7B | 8GB RAM | 16GB RAM |
| Phi3 Mini | 4GB RAM | 8GB RAM |
| Embedding (MiniLM) | 512MB RAM | 同上 |

常见问题：
- **`ollama serve` 报端口占用**：`lsof -i :11434` 查看冲突进程，或 `OLLAMA_HOST=0.0.0.0:11435 ollama serve` 换端口
- **模型推理超慢（>30s）**：换轻量模型 `phi3:mini`，或检查是否在用 swap 内存
- **GPU 加速**（可选）：如有 Apple Silicon / NVIDIA GPU，Ollama 会自动检测并使用，无需额外配置

**验收标准：**
- `ollama list` 显示 `mistral:7b`（或所选模型）
- `curl http://localhost:11434/api/generate` 在 5s 内返回响应
- `python -c "from langchain_ollama import ChatOllama; print('OK')"` 无报错

---

### Stage 2: Router + LangGraph Agent
**Files:** `agent/state.py`, `agent/router.py`, `agent/prompts.py`, `agent/nodes.py`, `agent/graph.py`

**2a. State definition** (`agent/state.py`)
```python
class AgentState(TypedDict):
    question: str
    route: str              # ECU_700 | ECU_800 | COMPARE | UNKNOWN
    matched_models: list[str]
    route_reason: str
    context: str            # retrieved text
    answer: str
    sources: list[dict]
    confidence: float
    latency_ms: float
```

**2b. Router** (`agent/router.py`) — Rule-based, deterministic, explainable

Priority order:
1. **COMPARE triggers** (highest priority):
   - Explicit: `compare`, `vs`, `versus`, `difference`, `differ`, `contrast`
   - Multi-model: `all models`, `all ECU`, `each model`, `every model`, `across.*model`
   - Feature scan: `which.*model.*support`, `which.*ECU.*support`, `which.*have`
   - Superlatives: `harshest`, `highest`, `lowest`, `best`, `most`, `least`, `maximum.*across`, `minimum.*across`
   - Multiple model mentions: if query mentions models from BOTH series
2. **Model-specific** (exact match):
   - `ECU-750`, `750` → ECU_700
   - `ECU-850b`, `850b` → ECU_800 (model=ECU-850b)
   - `ECU-850`, `850` (but NOT `850b`) → ECU_800 (model=ECU-850)
3. **Series-level**:
   - `ECU-700`, `700 series` → ECU_700
   - `ECU-800`, `800 series` → ECU_800
4. **Fallback**: UNKNOWN → treat as COMPARE (search everything)

Return: `RouteResult(route, matched_models, reason)`

**Verification against all 10 test questions:**
- Q1 "temperature for ECU-750" → matches `ECU-750` → ECU_700 ✓
- Q2 "RAM...ECU-850" → matches `ECU-850` → ECU_800 ✓
- Q3 "AI capabilities...ECU-850b" → matches `ECU-850b` → ECU_800 ✓
- Q4 "differences between ECU-850 and ECU-850b" → matches multiple 800 models → COMPARE ✓
- Q5 "Compare CAN bus...ECU-750 and ECU-850" → matches `compare` keyword → COMPARE ✓
- Q6 "power consumption...ECU-850b" → matches `ECU-850b` → ECU_800 ✓
- Q7 "Which ECU models support OTA" → matches `which.*model.*support` → COMPARE ✓
- Q8 "storage capacity...all ECU models" → matches `all.*model` → COMPARE ✓
- Q9 "harshest temperature" → matches superlative `harshest` → COMPARE ✓
- Q10 "enable NPU...ECU-850b" → matches `ECU-850b` → ECU_800 ✓

**All 10 routed correctly.**

**2c. Prompts** (`agent/prompts.py`)

SINGLE_SOURCE_PROMPT:
```
You are an engineering assistant for ME Corporation. Answer the question using ONLY the provided documentation.

Rules:
- Be precise with numerical values, units, and model numbers
- Cite the source document for each fact
- If information is not found, say so explicitly
- Do NOT invent specifications

Documentation:
{context}

Question: {question}

Answer:
```

COMPARISON_PROMPT:
```
You are an engineering assistant for ME Corporation. Answer the comparison question using ONLY the provided documentation.

Rules:
- Compare ALL relevant models systematically
- Present specifications side-by-side where applicable
- Explicitly state when a feature is NOT available for a model (negative confirmation)
- Be precise with numerical values and units
- Cite source documents

Documentation:
{context}

Question: {question}

Answer:
```

**2d. LangGraph nodes** (`agent/nodes.py`)
- `classify_node(state)` → run router, set route + matched_models
- `retrieve_single_node(state)` → FAISS retrieval filtered by series/model
- `retrieve_compare_node(state)` → inject all 3 full documents as context
- `synthesize_node(state)` → call LLM with appropriate prompt template
- Each node logs its action for debugging

**2e. Graph construction** (`agent/graph.py`)
```
START → classify → conditional_edge:
  ├─ ECU_700 → retrieve_single → synthesize → END
  ├─ ECU_800 → retrieve_single → synthesize → END
  ├─ COMPARE → retrieve_compare → synthesize → END
  └─ UNKNOWN → retrieve_compare → synthesize → END
```

Use `StateGraph(AgentState)` with `add_conditional_edges` from `classify` node.

**Verification:** Run the graph manually with all 10 test questions, check routes and answers.

---

### Stage 3: MLflow Model Packaging
**Files:** `model/pyfunc.py`, `model/log.py`

**3a. Custom PythonModel** (`model/pyfunc.py`)
```python
class MEAssistantModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load FAISS index from artifacts
        # Load raw docs from artifacts
        # Initialize embedding model (downloaded via pip dep)
        # Initialize LLM connection (Ollama URL from config)
        # Build the LangGraph

    def predict(self, context, model_input, params=None):
        # model_input: DataFrame with 'question' column
        # For each question: run graph, collect answer
        # Return DataFrame with columns: answer, route, sources, confidence, latency_ms
```

**3b. Log model** (`model/log.py`)
```python
# Artifacts to bundle:
artifacts = {
    "faiss_index": "data/faiss_index/",
    "docs": "data/docs/",
    "config": "config.json"
}

# Signature
signature = ModelSignature(
    inputs=Schema([ColSpec("string", "question")]),
    outputs=Schema([
        ColSpec("string", "answer"),
        ColSpec("string", "route"),
        ColSpec("string", "sources"),
        ColSpec("double", "confidence"),
        ColSpec("double", "latency_ms"),
    ])
)

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=MEAssistantModel(),
    artifacts=artifacts,
    pip_requirements=[...],
    signature=signature,
)
```

**Verification:** `mlflow.pyfunc.load_model(uri).predict(pd.DataFrame({"question": ["RAM in ECU-850?"]}))` returns correct answer.

---

### Stage 4: Docker + REST API
**Files:** `Dockerfile`, `docker-compose.yml`

**4a. docker-compose.yml** (two services)
```yaml
services:
  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    # Pull model on startup or use pre-built image

  rag-service:
    build: .
    ports: ["5001:5001"]
    depends_on: [ollama]
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    command: >
      mlflow models serve
        -m runs:/<run_id>/model
        -p 5001
        --no-conda
```

**4b. Dockerfile**
- Base: `python:3.11-slim`
- Install package via pip
- Run ingest + log-model at build time (or entrypoint)
- Artifacts baked into image

**Verification:** `docker compose up` → `curl -X POST http://localhost:5001/invocations -H 'Content-Type: application/json' -d '{"dataframe_split": {"columns": ["question"], "data": [["RAM in ECU-850?"]]}}'`

---

### Stage 5: Evaluation + Testing
**Files:** `eval/*`, `tests/*`

**5a. MLflow evaluation** (`eval/evaluate.py`)
- Load test-questions.csv
- Run `mlflow.evaluate()` with logged model
- Custom metrics (`eval/metrics.py`):
  - `answer_accuracy`: keyword/value matching against expected answers
  - `routing_correctness`: check route matches expected category
  - `source_correctness`: check sources come from right documents
  - `avg_latency`: mean response time

**5b. Unit tests**
- `test_router.py`: All 10 questions → correct route. Edge cases (typos, mixed case).
- `test_retrieval.py`: Single-source queries return chunks from correct document only.
- `test_graph.py`: End-to-end graph execution, check state transitions.
- `test_predict.py`: MLflow predict() returns correct DataFrame schema.

**Verification:** `make eval` prints score >= 8/10. `make test` passes. `make lint` shows pylint >85%.

---

### Stage 6: README + Documentation
**Content:**
- Architecture diagram (text-based)
- Key decisions table (chunking, routing, embedding, LLM)
- Setup & run instructions (make ingest → make serve → curl example)
- Docker instructions (docker compose up → curl example)
- Evaluation results (8/10 table)
- Testing strategy (unit + integration + eval framework)
- Limitations (small doc set, rule-based router, single LLM)
- Future work (see Scalability below)

---

## Optional Challenges (after core is solid)

### Stage 7: MLflow Evaluation Framework (Bonus 1)
**Files:** `eval/evaluate.py`, `eval/metrics.py`

**7a. Evaluation runner** (`eval/evaluate.py`)
- Load `test-questions.csv` into a pandas DataFrame
- Load the logged MLflow model
- Run `mlflow.evaluate()` with the model and evaluation DataFrame
- Log results as an MLflow run with metrics and artifacts

**7b. Custom metrics** (`eval/metrics.py`)
- `answer_accuracy(predictions, targets)`: Check if key values from expected answer appear in predicted answer. Use keyword extraction — e.g., for Q1 check "+85°C" appears; for Q10 check "me-driver-ctl --enable-npu --mode=performance" appears.
- `routing_correctness(predictions, targets)`: Map each question's category to expected route, compare with actual route in output.
- `source_correctness(predictions, targets)`: Verify `sources` field references the correct document(s).
- `response_latency(predictions)`: Check `latency_ms` column, compute avg/p95/max.
- `overall_score`: Count of questions where `answer_accuracy` passes (target: >=8).

**7c. Evaluation artifacts**
- Save per-question results table (question, expected, predicted, pass/fail) as CSV artifact
- Log summary metrics: `accuracy`, `avg_latency_ms`, `p95_latency_ms`, `routing_accuracy`
- Enable experiment comparison in MLflow UI

**Verification:** `make eval` produces MLflow run with visible metrics. Score >= 8/10.

---

### Stage 8: Human-in-the-Loop (Bonus 2)
**Files:** Modify `agent/nodes.py`, `agent/graph.py`, `agent/state.py`

**8a. Confidence scoring**
Add to `AgentState`:
```python
needs_human_review: bool
review_reason: str
```

Confidence is LOW when:
- `route == UNKNOWN`
- FAISS top-1 similarity score < 0.3 (tunable threshold)
- COMPARE query but retrieved docs only cover one series
- Query mentions a model name not in our database

**8b. LangGraph interrupt mechanism**
- Add `validate_confidence` node between `retrieve` and `synthesize`
- When confidence is low: use LangGraph `interrupt({"reason": ..., "draft_context": ...})`
- Graph pauses, returns intermediate state to caller
- Caller (API/CLI) can:
  - Provide corrected route/model → resume with `Command(resume={"route": "ECU_700"})`
  - Approve draft → resume with `Command(resume={"approve": True})`

**8c. Updated graph**
```
START → classify → conditional_edge → retrieve → validate_confidence
  ├─ confident → synthesize → END
  └─ needs_review → interrupt (pause) → [human input] → synthesize → END
```

**8d. API integration**
- predict() returns `{"status": "needs_review", ...}` for low-confidence queries
- Add `/resume` endpoint (or document CLI usage) for human correction

**Verification:** Craft an ambiguous query (e.g., "What about the ECU-900?") → triggers interrupt. Provide correction → get final answer.

---

### Stage 9: Advanced Agent Behaviors (Bonus 4)
**Files:** `agent/tools.py`, modify `agent/graph.py`, `agent/nodes.py`

**9a. Multi-step retrieval loop**
- After initial retrieval, check if evidence is sufficient:
  - For comparison: do we have data from ALL expected models?
  - For single-source: is the top similarity score > threshold?
- If insufficient: **query rewrite** — extract model name from question, append specific technical terms, retry FAISS with expanded query
- Maximum 2 retrieval attempts (prevent infinite loops)
- Log each retrieval step in state for observability

**9b. Tool-based architecture**
Define LangChain tools:
```python
@tool
def search_ecu700(query: str) -> str:
    """Search ECU-700 series documentation."""

@tool
def search_ecu800(query: str) -> str:
    """Search ECU-800 series documentation."""

@tool
def compare_models(feature: str) -> str:
    """Compare a specific feature across all ECU models."""

@tool
def get_full_specs(model: str) -> str:
    """Get complete specification table for a specific ECU model."""
```

- Wire tools into LangGraph as a ToolNode
- The router/planner node decides which tools to call based on query analysis
- Tool calls and results logged in state

**9c. Updated graph with loop**
```
START → classify → retrieve → check_evidence
  ├─ sufficient → synthesize → END
  └─ insufficient → rewrite_query → retrieve (loop, max 2x) → synthesize → END
```

**Verification:** Demo a question where initial retrieval misses (e.g., vague query) → agent rewrites and retries → finds answer on second attempt. Show tool call logs.

---

### Stage 10: Scalability Strategy (Bonus 3)
**File:** `README.md` (dedicated section)

Write a detailed scalability roadmap:

| Phase | Scope | Changes | Expected Outcome |
|-------|-------|---------|-------------------|
| **Phase 0** (current) | Single-node PoC | In-memory FAISS, single Ollama, batch predict | <20s, 10 docs |
| **Phase 1** (data scale) | 100s of documents | Persistent vector DB (Qdrant), incremental indexing pipeline, metadata-based sharding per product line, async document watcher | Minutes to index new docs, no redeployment |
| **Phase 2** (concurrency) | 10+ concurrent users | Embedding cache (Redis), LLM request queue, horizontal scaling of retrieval service, response caching for repeated queries | <5s p95 for cached queries |
| **Phase 3** (enterprise) | Multi-team, multi-product | Per-product-line vector store shards, role-based access control, model complexity routing (simple→small LLM, complex→large LLM), full observability (OpenTelemetry + LangSmith), A/B testing framework | Production SLA compliance |

For each phase, include:
- Specific modules/files that change
- Technology choices with justification
- Estimated effort
- Key metrics to track

---

## Execution Order (Priority-Driven)

```
Stage 1   (Ingest + VectorStore)    ~1.5 hrs
  ↓
Stage 1.5 (LLM & Embedding 部署)    ~0.5 hrs   ← 安装 Ollama + 拉模型 + 验证
  ↓
Stage 2   (Router + LangGraph)      ~2 hrs      ← CHECKPOINT: run 10 questions, must get 8/10
  ↓
Stage 3   (MLflow packaging)        ~1 hr
  ↓
Stage 5   (Eval + Tests)            ~1 hr       ← formal validation with MLflow
  ↓
Stage 4   (Docker + REST)           ~1.5 hrs
  ↓
Stage 7   (MLflow Eval Framework)   ~1 hr       ← Bonus 1
  ↓
Stage 9   (Advanced Agent)          ~1.5 hrs    ← Bonus 4 (multi-step + tools)
  ↓
Stage 8   (HITL)                    ~1 hr       ← Bonus 2
  ↓
Stage 10  (Scalability writeup)     ~0.5 hr     ← Bonus 3 (README section)
  ↓
Stage 6   (README finalization)     ~1 hr
```

**Total: ~12.5 hours**

**Critical checkpoints:**
1. After Stage 2: Run all 10 test questions through graph. Must get >=8/10. Do NOT proceed until answers are correct. Debug router patterns and prompts here.
2. After Stage 5: Formal eval confirms 8/10. Latency <20s confirmed.
3. After Stage 4: `docker compose up` + curl works end-to-end.

**If stuck on latency (>20s):**
- Reduce FAISS top_k from 5 to 3
- Use smaller Ollama model (`phi3:mini` or `gemma2:2b`)
- For comparison queries, trim full-doc context to just spec tables
- Pre-warm Ollama model on container startup
