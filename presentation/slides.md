# ME Engineering Assistant — Presentation Script

> **Duration**: 15-20 minutes (within a 60-minute code review session)
> **Audience**: Technical interviewers who wrote the challenge — they know the domain
> **Narrative**: How I analyzed, decomposed, and built this step by step

---

## Slide 1: Title

**ME Engineering Assistant**
*How I Approached the Challenge*

Your Name | Date

---

**Speaker Notes:**

Hi, thanks for having me. Today I'll walk through how I approached this challenge — not just the final system, but how I broke down the requirements, what decisions I made along the way, and where I changed course. I'll follow the order I actually built things, ending with a live demo.

---

## Slide 2: Reading the Requirements — What I Identified

### Core Task Decomposition

After reading the challenge, I broke it into **4 sub-problems**:

| # | Sub-Problem | Why It's Non-Trivial |
|---|------------|---------------------|
| 1 | **Multi-source retrieval** | Can't just dump everything into one prompt — need to know *which* doc to consult |
| 2 | **Intelligent routing** | "Compare ECU-750 and ECU-850" needs a different retrieval strategy than "RAM of ECU-850" |
| 3 | **Production packaging** | Not a notebook — installable package, MLflow model, Docker container, REST API |
| 4 | **Validation strategy** | How do you prove a RAG agent gives correct answers? |

### My First Observation

> The documents are **tiny** (~4KB total, 3 files). This changes everything about the architecture — I don't need heavy infrastructure, but I do need **precision**.

---

**Speaker Notes:**

The first thing I did was read the challenge carefully and identify the sub-problems. It's not just "build a chatbot" — there are four distinct engineering challenges here.

The first is multi-source retrieval. There are documents from different product lines, and the system needs to know which ones to consult. The second is intelligent routing — a comparison query and a single-source query need fundamentally different retrieval strategies. The third is production packaging — the challenge explicitly asks for an installable Python package, MLflow model, and containerized REST API. The fourth is validation — how do you actually prove the system works?

My first key observation was about the data. Three Markdown files, about 4 kilobytes total. That's small enough to fit entirely in any LLM's context window. This changed my architectural thinking significantly — I don't need a heavy vector database or complex chunking. But the small size doesn't make the problem easier; it makes precision more important, because every wrong answer is a bigger percentage of the test set.

---

## Slide 3: Architecture Planning — Before Writing Code

### I Started with the Data Flow

```
Question → What product? → Where to search? → Is the evidence good? → Generate answer
```

This naturally maps to a **pipeline with decision points**, not a single LLM call.

### Key Architectural Decisions Made Upfront

1. **LangGraph StateGraph** over AgentExecutor
   - I need explicit branches (single-source vs comparison), not LLM-decided routing
   - I need a retry loop for failed retrievals
   - I need the `interrupt()` mechanism for human-in-the-loop

2. **Metadata-driven retrieval** over naive similarity search
   - Tag each document with its series/model at load time
   - Filter retrieval results by metadata, not just similarity

3. **Dual retrieval paths** based on query type
   - Single-source: vector search with metadata filter
   - Comparison: inject all documents (they're tiny enough)

---

**Speaker Notes:**

Before writing any code, I sketched the data flow. A question comes in, we need to figure out what product it's about, decide where to search, check if we found good evidence, and then generate an answer. Each of these is a distinct step with a decision point.

This immediately told me I need a graph-based orchestration — not a single chain call. LangGraph's StateGraph was the right fit because I needed conditional branches: single-source queries and comparison queries need completely different retrieval strategies. I also anticipated needing a retry loop and the interrupt mechanism for human oversight. LangChain's AgentExecutor lets the LLM decide the next step, which doesn't make sense when I already know the control flow.

The second upfront decision was metadata-driven retrieval. Rather than relying purely on embedding similarity to find the right documents, I'd tag each document with its series and model at load time. This way, when the router says "this is an ECU-800 query," the retriever can filter by metadata instead of hoping similarity search returns the right series.

The third decision was recognizing I need two retrieval paths. For a single-source query like "RAM of ECU-850," vector search with metadata filtering works. But for "Compare all models," similarity search would likely return chunks from only one or two documents. Since the documents are tiny, I can just inject all of them into the context.

---

## Slide 4: Step 1 — The Ingest Pipeline (Problems I Found in the Data)

### What Seemed Simple...

Load 3 Markdown files, split them, build a FAISS index. Should be straightforward.

### Problem 1: ECU-700 Has Malformed Tables

```markdown
# What the file looks like:
**CAN Interface** | Single Channel | 1 Mbps

# What it should be:
| **CAN Interface** | Single Channel | 1 Mbps |
```

Missing leading `|` breaks every Markdown parser.

**Solution**: `_fix_malformed_table()` — state machine that tracks table context, auto-prepends `|`.

### Problem 2: ECU-700 Uses Non-Standard Headers

```markdown
# ECU-700 style (non-standard):
**1. Introduction**

# ECU-800 style (standard):
## Introduction
```

LangChain's `MarkdownHeaderTextSplitter` can't parse bold-numbered headers.

**Solution**: `_normalize_bold_headers()` — regex converts `**N. Title**` → `## Title` before splitting.

---

**Speaker Notes:**

The ingest pipeline seemed like the simplest part — load 3 Markdown files, chunk them, build a FAISS index. But I immediately hit two data quality issues.

The ECU-700 document has malformed tables. Some rows are missing the leading pipe character, so instead of a proper Markdown table row, you get something like `**CAN Interface** | Single Channel | 1 Mbps`. Every Markdown parser I tested — LangChain's splitter, Python-Markdown, even the embedding model — mishandles these lines. I wrote a `_fix_malformed_table` function that tracks whether we're inside a table context and auto-prepends the missing pipe character.

The second issue: ECU-700 uses bold-numbered text as headers — `**1. Introduction**` — instead of standard Markdown `##` headers. LangChain's `MarkdownHeaderTextSplitter` doesn't recognize these, so it treats the entire document as one chunk with no section boundaries. I wrote a preprocessing step that uses regex to convert these bold-numbered titles into proper `##` headers before passing to the splitter.

These are small fixes, but they illustrate an important point: *real-world data is messy, and the ingest pipeline is where you deal with it*. I chose to write a custom loader rather than using LangChain's built-in `DirectoryLoader` specifically because I needed this level of control. For 3 known-format files, a simple custom loader is better than a heavyweight framework that can't handle the edge cases.

---

## Slide 5: Step 2 — Routing (My Biggest Design Pivot)

### What I Tried First: LLM-Based Tool Selection

LangChain's tool-calling pattern — let Mistral choose which retrieval function to call.

**Result**: It worked, but...

| Problem | Impact |
|---------|--------|
| +12-18s latency per query | Doubled total response time |
| Inconsistent routing | Same question sometimes routed differently |
| Hard to debug | Why did it pick tool A over tool B? |

### What I Switched To: Deterministic Regex Router

```
Priority 1: COMPARE triggers (compare, vs, difference, which models...)
Priority 2: Model match (ECU-750, ECU-850, ECU-850b)
Priority 3: Series match (700 series, 800 series)
Priority 4: UNKNOWN fallback → search everything
```

**Result**: 10/10 routing accuracy, 0ms latency, fully debuggable.

### The Subtle Bug: 850 vs 850b

```
Naive regex: \b850\b  matches both "850" and "850b"
Fix: Check 850b FIRST, then 850 with negative lookahead (?!b)
```

---

**Speaker Notes:**

This was my biggest design pivot and probably the most interesting part of my development process.

I started with the "standard" approach — LangChain's tool-calling pattern. You define retrieval functions as tools, and the LLM decides which one to invoke. It seemed like the natural choice for "intelligent query routing."

It worked, but it had three problems. First, it added 12 to 18 seconds per query just for the routing decision — because it requires an extra LLM inference call. On a system where synthesis already takes 12 to 18 seconds on CPU, doubling the latency is not acceptable. Second, the routing was inconsistent — the same question could be routed differently on different runs because LLM output is non-deterministic. Third, when routing went wrong, debugging was difficult. The LLM doesn't explain why it chose tool A over tool B.

So I switched to a deterministic regex router. The routing space is small — 3 models, 2 series, 4 possible routes — and the model names follow a strict naming convention. Regex covers it perfectly. The result: 10 out of 10 routing accuracy on the test set, zero milliseconds of latency, and every routing decision comes with a human-readable `reason` field that explains why.

One subtle bug I caught during testing: the naive pattern `\b850\b` matches both "850" and the "850" inside "850b." The fix is to check "850b" first in the pattern list, and use a negative lookahead on the "850" pattern to ensure it's not followed by "b." This was caught by the parametrized test suite — which is exactly why you write tests before deployment.

The lesson: when your classification space is small and well-defined, rules outperform LLM routing on every axis. Don't use an LLM just because you can.

---

## Slide 6: Step 3 — Retrieval (Why One Strategy Wasn't Enough)

### The Problem I Discovered

Single-source queries and comparison queries need **fundamentally different** retrieval:

```
"RAM of ECU-850?"
  → Only need ECU-800 series docs
  → Vector search + metadata filter works perfectly

"Compare storage across all models"
  → Need ALL docs in context
  → Vector search would miss entire products
  → Full-document injection is better (docs are only 4KB)
```

### The FAISS Workaround

FAISS doesn't support metadata filtering natively. Alternatives:

| Option | Verdict |
|--------|---------|
| Switch to ChromaDB/Qdrant | Over-engineering for 3 docs |
| Use FAISS IDSelector | LangChain wrapper doesn't expose it |
| **3x oversampling + Python filter** | Simple, reliable at this scale |

```python
results = index.similarity_search_with_score(query, k=5 * 3)  # get 15
filtered = [doc for doc, _ in results if doc.metadata["series"] == series]
return filtered[:5]  # take top 5 after filtering
```

---

**Speaker Notes:**

Once the router was working, I moved to retrieval — and quickly realized I needed two fundamentally different strategies.

For single-source queries, vector search with metadata filtering works well. The router tells us it's an ECU-800 question, so we only retrieve chunks tagged with series "800." This prevents cross-contamination — the system won't cite ECU-750's 512 KB when you asked about ECU-850's 2 GB.

But for comparison queries, vector search is the wrong tool entirely. If you search for "storage," FAISS returns the top-K most similar chunks — which might all come from the same document. You'd miss entire products in the comparison. Since the documents are only 4 kilobytes total, I can inject all three documents in full. The LLM sees everything and can do a proper comparison.

The implementation detail worth noting is the metadata filtering workaround. FAISS doesn't support native metadata filtering — that's a feature of ChromaDB or Qdrant. But adding a database server for 3 documents is over-engineering. Instead, I use a 3x oversampling strategy: retrieve 15 results, filter by metadata in Python, and take the top 5. At our scale of roughly 10 chunks total, this guarantees sufficient coverage. In the scalability roadmap, migrating to Qdrant is the Phase 1 priority specifically because this workaround won't scale.

---

## Slide 7: Step 4 — Making It Robust (Evidence Check + Retry)

### What Went Wrong Without It

Early testing showed queries like *"What processor does it use?"* returning **empty context** — the query lacked specific terms that match the document embeddings.

### The Multi-Step Solution

```
retrieve → check evidence
                |
          sufficient? ──yes──→ continue
                |
               no
                |
          rewrite query (append keywords, NO LLM call)
                |
          retry retrieve (max 2 attempts)
                |
          continue with best available context
```

### Query Rewrite — Without an LLM

```
Original:  "What processor does it use?"
Rewritten: "What processor does it use? ECU-850 specifications
            technical data ECU-800 series automotive controller"
```

Why not use the LLM to rewrite?
- Adds 12-18s latency
- The problem is vocabulary mismatch, not semantic understanding
- Simple keyword expansion solves it

---

**Speaker Notes:**

During early testing, I found a failure mode: some queries would return empty or near-empty context from the vector search. The query "What processor does it use?" doesn't contain the words "CPU," "Cortex," or "ARM" that appear in the spec documents. The embedding similarity just wasn't high enough to retrieve the relevant chunks.

Rather than accepting empty context and letting the LLM hallucinate, I added an evidence check node. After every retrieval, it checks whether the context is sufficient — specifically, whether it's empty or under 50 characters. That threshold comes from observation: a valid spec table row like "Processor: ARM Cortex-M4 at 120 MHz" always exceeds 50 characters.

When evidence is insufficient, the system rewrites the query. It appends the matched model name, the series, and generic technical terms. This is a pure string operation — no LLM call. The problem is vocabulary mismatch between the query and the document embeddings, not a semantic understanding problem. Adding "ECU-850 specifications technical data" to the query shifts the embedding closer to the document space.

The retry loop runs at most twice — enough to handle vocabulary mismatch without creating infinite loops. If after two attempts the context is still poor, we proceed anyway and let the confidence scoring handle the uncertainty downstream.

---

## Slide 8: Step 5 — Knowing What You Don't Know (HITL)

### The Question That Inspired This

> "What are the specs of the ECU-900?"

There is no ECU-900. What should the system do?

### Option A: Generate an answer anyway (most RAG systems)
### Option B: Flag it and ask for human help (my approach)

### Confidence Scoring

```
Base score by route:
  COMPARE  → 0.9   (all docs injected — most reliable)
  ECU_700  → 0.85
  ECU_800  → 0.85
  UNKNOWN  → 0.5   (can't identify product — risky)

Penalties:
  Failed evidence check  → -0.15
  Empty context          → -0.30
  Short context (<100c)  → -0.10
```

### Three Interrupt Triggers

1. Route = UNKNOWN (unrecognized product)
2. Unknown model mentioned (e.g., "ECU-900" not in our registry)
3. Empty retrieval context

### Implementation: LangGraph `interrupt()`

- Graph **pauses** at checkpoint, sends payload to UI
- Human reviews: approve OR provide corrected route
- Graph **resumes** with `Command(resume=...)` — no restart

---

**Speaker Notes:**

This feature was motivated by a specific question: what happens when someone asks about the ECU-900, which doesn't exist in our documentation? Most RAG systems would generate a plausible-sounding but completely fabricated answer. I wanted the system to recognize this as an uncertain situation and involve a human.

The confidence scoring is heuristic — not LLM-based. I considered having Mistral evaluate its own confidence, but that adds another 15-second LLM call and the self-calibration of a 7B model is unreliable. Instead, I use factors the system already knows: the route type tells us whether we identified the product, and the context quality tells us whether retrieval succeeded.

The COMPARE route gets the highest base score because we inject all documents in full — the LLM always has complete information. UNKNOWN gets the lowest because it means we couldn't even identify what product the user is asking about.

The implementation uses LangGraph's `interrupt()` mechanism. When triggered, the graph state is checkpointed, execution pauses, and a payload is sent to the UI showing the confidence score, detected route, and context preview. The human can approve — let it proceed anyway — or provide a corrected route. Then the graph resumes from the checkpoint, not from scratch.

This is the feature I'm most proud of, because it demonstrates a critical principle: *a production AI system should know what it doesn't know*.

---

## Slide 9: Step 6 — MLflow & Docker (Making It Deployable)

### The Packaging Challenge

This isn't a single model — it's a composite pipeline:

```
FAISS index + embeddings + documents + LangGraph + LLM HTTP calls
```

### MLflow PythonModel

```python
class MEAssistantModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Rebuild pipeline from artifacts
    def predict(self, context, model_input):
        # Input:  DataFrame["question"]
        # Output: DataFrame[answer, route, sources, confidence, latency_ms]
```

Artifacts bundled: `faiss_index/`, `docs/`

### Docker Architecture

```
  Port 80 (Nginx) → /     Streamlit UI (:8501)
                  → /api/  MLflow REST API (:5001)
                            ↓
                     Ollama on host (:11434)
```

- Same Dockerfile, `SERVICE_MODE=api|ui` switches behavior
- Health checks + service dependencies
- One command: `docker compose up --build`

### Problem Encountered: CPU Inference Too Slow

Ollama in Docker = pure CPU = timeouts.
**Fix**: Run Ollama on host machine, Docker connects via `host.docker.internal`.

---

**Speaker Notes:**

The challenge requires an installable Python package, an MLflow model, and a Docker container. The packaging challenge is that this isn't a single model — it's a FAISS index, document files, embedding computations, a LangGraph state machine, and LLM HTTP calls. MLflow's PythonModel interface lets me wrap all of this into one deployable artifact.

The Docker architecture uses Nginx as a reverse proxy — single port 80 exposed — routing to either the Streamlit UI or the MLflow REST API. An interesting design choice: I use the same Docker image for both services, with a `SERVICE_MODE` environment variable controlling whether the container starts Streamlit or MLflow serving. This avoids maintaining two Dockerfiles.

One real problem I encountered: Ollama running inside Docker on macOS has no GPU access. Mistral 7B inference on pure CPU was exceeding timeout limits. The fix was to run Ollama natively on the host machine, where it can use GPU acceleration, and have the Docker services connect to it via `host.docker.internal`. This is a practical architectural lesson — sometimes the right answer is to not containerize everything.

---

## Slide 10: Step 7 — Testing & Evaluation (Proving It Works)

### Test Pyramid: 31 Tests, All Offline

| Layer | Count | What It Validates | Dependencies |
|-------|-------|-------------------|--------------|
| Unit | 16 | Router correctness | None |
| Retrieval | 6 | Metadata filtering | Real FAISS |
| Integration | 7 | Graph node + pipeline | FAISS + mocked LLM |
| Schema | 2 | MLflow interface | None |

**All offline, < 30s, no Ollama/GPU needed.**

### Evaluation: Why Keyword Matching, Not Cosine Similarity

```
"The ECU-850 has 2 GB LPDDR4 RAM."  ← correct
"The ECU-850 has 4 GB LPDDR4 RAM."  ← wrong

Cosine similarity: ~0.95  (can't tell them apart!)
Keyword check: "2 GB" present? → clearly distinguishes
```

### Results

| Metric | Score | Target |
|--------|-------|--------|
| Answer accuracy | **9/10** | 8/10 |
| Routing accuracy | **10/10** | — |
| Response time | **~15s** | <20s |
| Pylint | **9.95/10** | >8.5/10 |

---

**Speaker Notes:**

For testing, I built a 4-tier pyramid. The key constraint: every test must run offline, without Ollama or a GPU. This means all LLM calls are mocked. The point of the test suite isn't to validate Mistral's answer quality — that's what the evaluation framework does. The tests validate the system's logic: does the router classify correctly? Does metadata filtering prevent cross-series contamination? Does state flow correctly through all 15 graph fields?

For evaluation, I chose keyword matching over cosine similarity, and this is an important decision. A wrong answer — "4 GB" instead of "2 GB" — would get a cosine similarity of about 0.95 against the correct answer because the sentences are structurally identical. Keyword matching catches this immediately: is "2 GB" present in the answer? Yes or no.

The evaluation dataset is 10 questions, each with expected keywords, expected route, and expected sources. A question passes only if ALL keywords are found. This is strict, but for technical specification QA, partial correctness is the same as incorrectness.

The results exceed all the success criteria in the challenge: 9 out of 10 accuracy versus the 8/10 target, response time around 15 seconds versus the 20-second limit, and Pylint at 9.95 versus the 85% target.

---

## Slide 11: The Full Picture

### Architecture — Built Incrementally

```
START → classify → [route]
  |
  ├── ECU_700/ECU_800 → retrieve_single → check_evidence
  |                                           |
  |                              sufficient ──┤── insufficient
  |                                  |        |
  |                                  |   rewrite_query → retry
  |                                  |        |
  └── COMPARE/UNKNOWN → retrieve_compare ─────┘
                                  |
                          validate_confidence
                            (HITL interrupt)
                                  |
                             synthesize
                            (Mistral 7B)
                                  |
                               Answer
                    (+ route, sources, confidence)
```

### 15 State Fields — Full Observability

`question`, `answer`, `route`, `matched_models`, `route_reason`,
`context`, `search_query`, `retrieval_attempts`, `evidence_sufficient`,
`evidence_gap`, `sources`, `confidence`, `latency_ms`,
`needs_human_review`, `review_reason`

---

**Speaker Notes:**

This is the complete system as it stands. Every component I've discussed — the routing, the dual retrieval paths, the evidence check with retry, the confidence scoring with human-in-the-loop — they're all nodes in this LangGraph StateGraph.

The graph carries 15 typed state fields through every node. This means at any point, you can inspect exactly what the system decided and why — what route was chosen and for what reason, what context was retrieved, how many retrieval attempts were needed, what the confidence score is and whether human review was triggered.

Each piece was built incrementally. I didn't design this complete graph on day one. I started with classify and retrieve, then added the evidence check when I found that retrieval sometimes fails, then added confidence scoring when I realized the system needs to know what it doesn't know. The architecture emerged from solving real problems, not from theoretical design.

---

## Slide 12: Scalability — What I'd Do Next

| Phase | Scale | Key Change |
|-------|-------|-----------|
| **0 (Current)** | 3 docs, 1 user | FAISS, regex router, Ollama |
| **1** | 500 docs | Qdrant (native metadata filter), configurable route patterns |
| **2** | 50 users | FastAPI + async, Redis cache, Ollama replicas |
| **3** | Enterprise | OpenTelemetry, RBAC, complexity-based model routing |

### What Would Break First at Scale

1. **FAISS oversampling** — 3x works for 10 chunks, not for 10,000
2. **Hardcoded regex patterns** — can't manually add patterns for 500 products
3. **Synchronous LLM calls** — blocks everything during 15s inference
4. **No caching** — same question = same 15s wait every time

### Each Phase Keeps What Works, Replaces What Doesn't

- LangGraph graph structure → stays (just swap retriever implementations)
- Metadata tagging at ingest → stays (just richer taxonomy)
- Keyword evaluation → stays (just more test questions)
- Rule-based routing → evolves (YAML config → auto-discovery)

---

**Speaker Notes:**

I want to be honest about what doesn't scale and show that I've thought about the path forward.

The first thing that breaks is the FAISS oversampling workaround. With 10 chunks, retrieving 3x and filtering works. With 10,000 chunks, you'd need to retrieve 15,000 results just to filter to 5 — that's wasteful. Qdrant with native metadata filtering is the Phase 1 solution.

The second is the hardcoded regex patterns. Adding regex for every new product model is unsustainable at 500 products. The fix is a configurable pattern registry — ideally auto-discovered from ingested documents.

The third is synchronous LLM calls. Right now, each query blocks for 15 seconds. With 50 concurrent users, you need async processing, a task queue, and multiple Ollama replicas.

What's important is that the core architecture — the LangGraph graph structure, the metadata tagging approach, the separation of routing from retrieval — these survive every phase. I'd swap out the retriever implementation and the router configuration, but the pipeline topology stays the same. That's the sign of a well-decomposed system.

---

## Slide 13: Live Demo

### I'll Walk Through Three Scenarios

**1. Single-Source** → "What is the RAM of ECU-850?"
- Watch: router → ECU_800, FAISS retrieval, confidence 0.85

**2. Comparison** → "Compare the storage capacity across all ECU models"
- Watch: router → COMPARE, full-doc injection, all 3 models in answer

**3. Human-in-the-Loop** → "What are the specs of the ECU-900?"
- Watch: router → UNKNOWN, confidence drops, interrupt triggers, review panel appears

### What to Notice

- Pipeline step visualization in the sidebar
- Route badge + confidence score in metadata
- Source document citations
- HITL review panel: approve vs correct route

---

**Speaker Notes:**

Let me show you the system in action. I'll run through three queries that exercise different paths through the pipeline.

First, a straightforward single-source query. "What is the RAM of ECU-850?" — watch the sidebar for the pipeline visualization. The router identifies ECU-850, routes to ECU_800, retrieves from the correct document, and the answer should say "2 GB LPDDR4."

Second, a comparison. "Compare the storage capacity across all ECU models." This triggers the COMPARE route — notice how the retrieval strategy is different. Instead of vector search, it injects all three documents. The answer should cover the ECU-750's 2 MB flash, the ECU-850's 16 GB eMMC, and the ECU-850b's 32 GB.

Third, the human-in-the-loop scenario. "What are the specs of the ECU-900?" — there's no ECU-900. Watch what happens: the router can't match a model, confidence drops to around 0.35, and the system pauses. The review panel shows why, and I can either approve or provide a corrected route.

[If demo fails, use prepared screenshots]

---

## Slide 14: Summary — My Approach

### How I Worked

1. **Analyzed requirements** before writing code — identified 4 sub-problems
2. **Built incrementally** — data pipeline → routing → retrieval → robustness → HITL → packaging
3. **Pivoted when evidence said to** — abandoned LLM routing after measuring it
4. **Tested at every layer** — 31 offline tests, keyword-based evaluation
5. **Documented decisions** — not just what I built, but *why* and *what I considered*

### Key Lessons

| What I Tried | What I Learned |
|-------------|---------------|
| LLM-based routing | Rules > LLM when classification space is small |
| LangChain `ChatOllama` | httpx had Docker compatibility issues → used `urllib` |
| Ollama in Docker | No GPU on macOS → moved to host-native |
| Semantic similarity eval | Can't distinguish "2 GB" from "4 GB" → keyword matching |

### Results vs Requirements

| Requirement | Target | Achieved |
|-------------|--------|----------|
| Answer accuracy | 8/10 | **9/10** |
| Response time | <20s | **~15s** |
| Pylint score | >85% | **99.5%** |
| Tests | Documented strategy | **31 tests, 4-tier pyramid** |

Thank you — happy to dive into any part of the code.

---

**Speaker Notes:**

To wrap up, I want to highlight how I approached this challenge rather than just what I built.

I started by analyzing the requirements and breaking them into sub-problems before writing any code. I built incrementally — each component motivated by a real problem I encountered. When LLM routing didn't meet my latency and accuracy standards, I pivoted to regex. When I discovered that retrieval sometimes fails, I added the evidence check and retry loop. When I realized the system has no way to flag uncertain answers, I added confidence scoring and human-in-the-loop.

The lessons I learned are practical: rules beat LLMs for small classification spaces, LangChain wrappers can hide compatibility issues, Docker doesn't always mean better, and semantic similarity can't validate technical accuracy.

The final system exceeds all the success criteria — 9/10 accuracy against the 8/10 target, 15 seconds against the 20-second limit, and 99.5% Pylint against the 85% threshold.

I'm happy to dive into any part of the codebase — the router patterns, the graph topology, the test fixtures, the Docker configuration — whatever you'd like to explore.

---

## Appendix: Anticipated Q&A

### "Why not GPT-4 or Claude instead of Mistral 7B?"

The challenge specifies open-source/local models. Mistral 7B achieves 9/10 on our test set — sufficient for the PoC. The architecture is LLM-agnostic: swapping models requires changing one environment variable (`OLLAMA_MODEL`).

### "What would you change if you started over?"

I'd use Qdrant from the start instead of FAISS — the oversampling workaround works but feels fragile. I'd also consider a lightweight classification model for routing instead of regex, to handle more flexible query phrasings without manual pattern maintenance.

### "How would you handle PDF or Word documents?"

Add format-specific loaders (`PyMuPDF`, `python-docx`) that output the same `LoadedDocument(NamedTuple)` interface. The rest of the pipeline — chunking, indexing, retrieval — stays unchanged.

### "What about multi-turn conversations?"

Currently single-turn only. Multi-turn requires coreference resolution ("What about its storage?" → resolve "its" to the previous model) and conversation state in LangGraph. This is a Phase 2 enhancement.

### "How do you handle conflicting specifications across documents?"

The metadata filtering ensures we retrieve from the correct document per query. If two documents had conflicting specs for the same model, the system would surface both in context. A proper solution would add document versioning with a "latest" flag.

### "Why is Q9 scored as PASS* instead of PASS?"

Q9 asks "Which ECU operates in the harshest temperature conditions?" The expected keywords are "+105C" and "+85C." Mistral sometimes phrases the answer differently — describing the temperature range rather than citing exact values. This is a known limitation of keyword-based evaluation; the answer is factually correct but doesn't match the expected tokens.
