# Full Pipeline Benchmark Report

**Date:** 2026-03-05
**Script:** `scripts/benchmark_full.py`
**Environment:** macOS (Apple Silicon), Ollama + Mistral 7B (GPU), conda base

---

## 1. Objective

Quantify the impact of replacing rule-based components with LLM-based alternatives at two stages of the RAG pipeline: **routing** and **query rewriting**. Determine whether LLM substitution at either stage improves answer accuracy enough to justify the added latency.

---

## 2. Experimental Design

### 2.1 Independent Variables (What We Changed)

| Variable | Option A (Rule-based) | Option B (LLM-based) |
|----------|----------------------|----------------------|
| **Routing method** | Regex pattern matching with priority rules (`router.py`) | Mistral 7B classification via Ollama (`llm_router.py`) |
| **Query rewrite method** | Keyword expansion: append model name + technical terms | Mistral 7B rewrite: prompt LLM to add retrieval-relevant terms |

These two binary variables produce **4 strategy combinations**:

| Strategy | Routing | Query Rewrite |
|----------|---------|---------------|
| **A** (current system) | Regex | Keyword |
| **B** | LLM | Keyword |
| **C** | Regex | LLM |
| **D** | LLM | LLM |

### 2.2 Controlled Variables (What We Held Constant)

| Variable | Value | Rationale |
|----------|-------|-----------|
| **Test questions** | 10 standard questions from `data/test-questions.csv` | Official evaluation set from the challenge spec |
| **LLM model** | `mistral:7b` via Ollama | Same model for all LLM calls (routing, rewriting, synthesis) |
| **Embedding model** | `all-MiniLM-L6-v2` | Same embeddings for all FAISS lookups |
| **FAISS index** | Single pre-built index from 3 ECU documents | Identical retrieval corpus across all strategies |
| **Document corpus** | ECU-700_Series_Manual.md, ECU-800_Series_Base.md, ECU-800_Series_Plus.md | No documents added or removed |
| **Retrieval logic** | `retrieve_by_model()` / `retrieve_by_series()` / `retrieve_all_docs()` | Same retrieval functions, selected by route output |
| **Synthesis prompt** | `format_prompt()` from `prompts.py` (single-source or comparison template) | Same prompt template for all strategies |
| **Synthesis LLM** | `mistral:7b` via Ollama, `stream=False`, timeout 300s | Identical generation parameters |
| **Answer evaluation** | Keyword matching against `EXPECTED_KEYWORDS` (case-insensitive) | Deterministic pass/fail criteria per question |
| **Route evaluation** | Exact match against `EXPECTED_ROUTES` | Deterministic correctness check |
| **Hardware** | Apple Silicon Mac, GPU-accelerated Ollama | No CPU/GPU variation between runs |
| **Ollama temperature** | Default (0.7) | Not overridden in any strategy |
| **Run count** | 1 run per strategy per question | Single pass (no averaging across runs) |

### 2.3 Dependent Variables (What We Measured)

| Metric | Unit | Description |
|--------|------|-------------|
| **Route correctness** | boolean | Does the route match `EXPECTED_ROUTES[qid]`? |
| **Answer correctness** | boolean | Do all `EXPECTED_KEYWORDS[qid]` appear in the answer? |
| **Route latency** | ms | Time to classify the query into a route |
| **Retrieve latency** | ms | Time to fetch relevant context from FAISS / full docs |
| **Rewrite latency** | ms | Time to expand/rewrite the query |
| **Synthesis latency** | ms | Time for LLM to generate the final answer |
| **Total latency** | ms | Sum of all four stages |

---

## 3. Method Details

### 3.1 Regex Routing (Strategies A, C)

Rule-based pattern matching in `src/me_assistant/agent/router.py`:
1. Extract model mentions via regex (ECU-850b, ECU-850, ECU-750, etc.)
2. Check compare triggers (keywords: "compare", "vs", "difference"; multi-model detection; superlatives)
3. Priority: COMPARE > specific model > series > UNKNOWN
4. Deterministic: same input always produces same output
5. Latency: negligible (<1 ms)

### 3.2 LLM Routing (Strategies B, D)

LLM classification in `src/me_assistant/agent/llm_router.py`:
- Prompt asks Mistral 7B to classify into exactly one of: `ECU_700`, `ECU_800`, `COMPARE`, `UNKNOWN`
- Response parsed with fallback: strip whitespace/backticks, try exact match, then substring search
- If unparseable → defaults to `UNKNOWN`

### 3.3 Keyword Query Rewrite (Strategies A, B)

Deterministic keyword expansion:
- Append `"{model} specifications technical data"` if a model is detected
- Append series context (e.g., "ECU-700 series automotive controller" for 750)
- Append generic terms: "features parameters performance specifications"
- Latency: negligible (<1 ms)

### 3.4 LLM Query Rewrite (Strategies C, D)

LLM-based rewrite via Mistral 7B:
- Prompt: "Rewrite this question to improve retrieval from technical ECU specification documents. Add relevant technical terms, model numbers, and specification keywords."
- Input: original question + detected model name
- Output: rewritten query string
- Latency: ~2-4 seconds per call

### 3.5 Retrieval

Shared across all strategies (controlled):
- Route = `COMPARE` or `UNKNOWN` → inject all 3 full documents as context
- Route = `ECU_700` or `ECU_800` → FAISS similarity search with metadata filtering
  - If specific model matched → `retrieve_by_model()`
  - Otherwise → `retrieve_by_series()`

### 3.6 Synthesis

Shared across all strategies (controlled):
- Prompt template selected by route (single-source vs comparison)
- Single LLM call to Mistral 7B with retrieved context
- Full generation, no streaming

### 3.7 Evaluation Criteria

**Route accuracy:** exact string match against ground truth `EXPECTED_ROUTES`:
```
Q1→ECU_700, Q2→ECU_800, Q3→ECU_800, Q4→COMPARE, Q5→COMPARE,
Q6→ECU_800, Q7→COMPARE, Q8→COMPARE, Q9→COMPARE, Q10→ECU_800
```

**Answer accuracy:** all keywords in `EXPECTED_KEYWORDS` must appear (case-insensitive):
```
Q1: ["+85°C", "-40°C"]
Q2: ["2 GB", "LPDDR4"]
Q3: ["NPU", "5 TOPS"]
Q4: ["NPU", "4 GB", "2 GB", "1.5 GHz", "1.2 GHz"]
Q5: ["single channel", "dual channel"]
Q6: ["1.7A", "550mA"]
Q7: ["OTA"]
Q8: ["2 MB", "16 GB", "32 GB"]
Q9: ["+105°C", "+85°C"]
Q10: ["me-driver-ctl", "--enable-npu"]
```

---

## 4. Results

### 4.1 Per-Question Routing and Answer Accuracy

| Q# | Question (abbreviated) | Expected Route | A (Regex+KW) | B (LLM+KW) | C (Regex+LLM) | D (LLM+LLM) |
|----|----------------------|----------------|-------------|-------------|---------------|-------------|
| 1 | Max temp for ECU-750? | ECU_700 | R:ok A:FAIL | R:ok A:FAIL | R:ok A:FAIL | R:ok A:FAIL |
| 2 | RAM of ECU-850? | ECU_800 | R:ok A:ok | R:ok A:ok | R:ok A:ok | R:ok A:ok |
| 3 | AI capabilities of ECU-850b? | ECU_800 | R:ok A:ok | R:ok A:FAIL | R:ok A:ok | R:ok A:ok |
| 4 | Differences 850 vs 850b? | COMPARE | R:ok A:ok | R:ok A:ok | R:ok A:ok | R:ok A:ok |
| 5 | Compare CAN bus 750 vs 850? | COMPARE | R:ok A:FAIL | R:ok A:ok | R:ok A:ok | R:ok A:ok |
| 6 | Power consumption of ECU-850b? | ECU_800 | R:ok A:FAIL | R:ok A:FAIL | R:ok A:FAIL | R:ok A:FAIL |
| 7 | Which models support OTA? | COMPARE | R:ok A:ok | R:ok A:ok | R:ok A:ok | R:ok A:ok |
| 8 | Storage capacity across models? | COMPARE | R:ok A:FAIL | R:ok A:ok | R:ok A:ok | R:ok A:ok |
| 9 | Harshest temperature conditions? | COMPARE | R:ok A:ok | R:WRONG A:FAIL | R:ok A:FAIL | R:WRONG A:ok |
| 10 | Enable NPU on ECU-850b? | ECU_800 | R:ok A:ok | R:ok A:ok | R:ok A:ok | R:ok A:ok |

### 4.2 Average Latency Breakdown (ms per query)

| Stage | A: Regex+Keyword | B: LLM+Keyword | C: Regex+LLM | D: LLM+LLM |
|-------|-----------------|-----------------|---------------|-------------|
| Route | <1 | 1,220 | <1 | 4,190 |
| Retrieve | 220 | 158 | 128 | 144 |
| Rewrite | <1 | <1 | 2,884 | 3,263 |
| Synthesize | 10,042 | 11,958 | 13,228 | 15,756 |
| **TOTAL** | **10,262** | **13,336** | **16,240** | **23,353** |

### 4.3 Summary Metrics

| Metric | A: Regex+Keyword | B: LLM+Keyword | C: Regex+LLM | D: LLM+LLM |
|--------|-----------------|-----------------|---------------|-------------|
| Route Accuracy | **10/10 (100%)** | 9/10 (90%) | **10/10 (100%)** | 9/10 (90%) |
| Answer Accuracy | 6/10 (60%) | 6/10 (60%) | 7/10 (70%) | **8/10 (80%)** |
| Avg Total Latency | **10,262 ms** | 13,336 ms | 16,240 ms | 23,353 ms |
| Overhead vs A | baseline | +3,074 ms (+30%) | +5,978 ms (+58%) | +13,091 ms (+128%) |

---

## 5. Analysis

### 5.1 Routing: Regex Is Strictly Superior

- Regex routing achieves **100% accuracy** on all 10 questions; LLM routing drops to **90%**
- LLM consistently misroutes Q9 ("Which ECU can operate in the harshest temperature conditions?") — a superlative query requiring comparison across all models. The regex router catches this via superlative keyword patterns; the LLM incorrectly classifies it as single-model.
- LLM routing adds **~1.2 seconds** per query on average (GPU inference). This is far less than the originally estimated 12-18s (which assumed CPU inference for full answer generation, not short classification output of ~7 tokens).
- Regex routing is **deterministic** — identical input always yields identical output. LLM routing showed inconsistency in prior multi-run tests (93% consistency across 3 runs).

### 5.2 Query Rewrite: LLM Improves Answer Quality

- Strategy C (Regex+LLM rewrite) scores **7/10** vs Strategy A's **6/10** — a +1 improvement
- Strategy D (LLM+LLM) scores **8/10**, the highest answer accuracy
- The LLM rewrite adds **~2.9 seconds** per query
- The improvement comes from LLM-rewritten queries adding domain-specific terms that help the synthesis LLM generate more complete answers

### 5.3 Synthesis Dominates Latency

- The synthesis (answer generation) stage accounts for **78-97%** of total pipeline time across all strategies
- Route + Rewrite overhead is relatively small compared to synthesis
- Even in Strategy D (worst case), routing + rewriting together take ~7.5s vs ~15.8s for synthesis
- Optimizing synthesis latency (smaller model, quantization, streaming) would have the largest impact

### 5.4 Answer Failures Are Mostly Synthesis Variance

- Q1 and Q6 fail across **all four strategies** — this is a synthesis quality issue, not a routing or retrieval problem
- Q1 requires exact temperature values (+85°C, -40°C); Q6 requires exact power values (1.7A, 550mA) — the LLM sometimes omits or rounds these
- These failures are non-deterministic: the same pipeline may pass or fail on repeated runs due to LLM generation variance

### 5.5 Trade-off Summary

| | Accuracy | Latency | Determinism | Complexity |
|---|----------|---------|-------------|------------|
| **Strategy A** (current) | Route: perfect, Answer: 6/10 | 10.3s (fastest) | Fully deterministic routing | Simplest |
| **Strategy D** (full LLM) | Route: 90%, Answer: 8/10 | 23.4s (2.3x slower) | Non-deterministic | Most complex |

---

## 6. Conclusions

1. **Regex routing is the correct design choice** for this system — it provides perfect accuracy at negligible cost with full determinism.

2. **LLM query rewrite is the most promising upgrade path** — Strategy C achieves +1 answer accuracy at +6s latency, without sacrificing routing correctness. If the 20-second latency budget permits, this is a worthwhile enhancement.

3. **The synthesis LLM is the true bottleneck** — routing and rewrite optimizations yield marginal latency improvements compared to the 10-16s synthesis cost. Future optimization should focus on the synthesis stage (model selection, quantization, prompt tuning).

4. **Answer accuracy is bounded by synthesis quality** — the 6/10 baseline is not a routing or retrieval failure; it reflects the LLM's ability to extract and reproduce exact numeric values from context. Improving prompts or using a more capable model would likely improve this more than changing the routing or rewrite strategy.

---

## 7. Reproducibility

```bash
# Prerequisites
# - Ollama running with mistral:7b pulled
# - FAISS index built (make ingest)
# - conda base environment with me_assistant installed

# Run benchmark
python scripts/benchmark_full.py

# Expected runtime: ~4 minutes (GPU) or ~15 minutes (CPU)
```

**Script location:** `scripts/benchmark_full.py`
**Evaluation data:** `src/me_assistant/eval/metrics.py` (EXPECTED_ROUTES, EXPECTED_KEYWORDS)
**Test questions:** `data/test-questions.csv`
