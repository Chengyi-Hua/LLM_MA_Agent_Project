# LLM Multi-Agent Wikipedia Generation

## Overview

This project builds a multi-agent RAG pipeline to automatically generate Wikipedia-style articles for geographic islands. Four methods are implemented and compared:

| Method | Name | Description |
|--------|------|-------------|
| method0 | Pure Generation | No RAG — LLM generates from island name only |
| method1 | Naive RAG | All chunks retrieved globally, single prompt |
| method2 | Hierarchical RAG | Chunks retrieved per section, sections generated independently |
| method3 | Inter-Section RAG | Multi-agent: NLI dependency graph + context-aware generation |


---

## Repository Structure

```
LLM_MA_Agent_Project/
├── 00_evaluation/
│   ├── __pycache__/
│   ├── evaluations/                 # Evaluation CSV outputs are saved here
│   ├── experiments/                 # Optional experiment files or intermediate runs
│   ├── references/                  # Human Wikipedia reference articles
│   ├── batch_experiments.py         # Runs batch generation/evaluation experiments
│   ├── eval_utils.py                # Shared evaluation helpers
│   ├── evaluate_basic.py            # Basic writing + informativeness evaluation
│   ├── evaluation.py                # Older/general evaluation entry point
│   ├── fetch_wikipedia_references.py # Fetches original Wikipedia articles for references/
│   ├── full_evaluation.py           # Main full evaluation pipeline
│   ├── metrics_artifact_diagnostics.py # Artifact and pipeline diagnostic metrics
│   ├── metrics_cscs.py              # Cross-Sectional Consistency Score
│   ├── metrics_informativeness.py   # ROUGE-L and METEOR
│   └── metrics_verifiability.py     # Citation recall, precision, rate, and NLI support checking
│
├── methods/
│   ├── __init__.py
│   ├── base_rag.py              # Shared logic: LLM init, reranking, output format
│   ├── pure_generation.py       # Method 0
│   ├── naive_rag.py             # Method 1
│   ├── hierarchical_rag.py      # Method 2
│   └── inter_section_rag.py     # Method 3 — coordinates Agent 2 + Agent 3
├── agents/
│   ├── __init__.py
│   ├── agent2_orchestrator.py   # NLI + DAG + topological sort
│   ├── agent3_generator.py      # Context-aware section generation
│   └── agent4_evaluator.py      # LLM-as-judge evaluation
├── retrieval/
│   └── rag_data_pipeline.py     # Fetches data via Tavily, saves to data/
├── evaluation/
│   └── ...
├── data/
│   ├── __init__.py
│   ├── Surtsey_rag_context.json
│   ├── Nishinoshima_(Ogasawara)_rag_context.json
│   └── outputs/                 # Generated articles saved here
├── logs/
│   └── agent2_plans/            # Agent 2 DAG plans saved here (auto-created)
├── config/
│   └── settings.yaml            # Model provider, reranker, and NLI settings
├── pipeline.py                  # Generation only (requires existing data file)
├── full_pipeline.py             # End-to-end: island name → article
├── requirements.txt
├── .env.example                 # API key template
└── .gitignore
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create your `.env` file**
```bash
cp .env.example .env
```
Fill in your API keys:
```
OPENROUTER_API_KEY=sk-or-...
TAVILY_API_KEY=...          # only needed for full_pipeline.py (data collection)
```
> ⚠️ Never commit `.env` to git. It is already listed in `.gitignore`.

**3. Check your model in `config/settings.yaml`**

All methods use OpenRouter by default. To switch models, edit the `model:` field:
```yaml
method1:
  provider: "openrouter"
  model: "openai/gpt-4.1-mini"   # any model from openrouter.ai/models
```
Supported providers: `openrouter`, `openai`, `groq`.

---

## Running

> ⚠️ Always run from the **project root**. Running individual files directly causes `ModuleNotFoundError`.

### Option A — Full pipeline (recommended)
Fetches data from Tavily if not already cached, then generates the article.
```bash
python full_pipeline.py --island Surtsey --method method3
python full_pipeline.py --island Surtsey --method all

# Re-fetch data even if cache exists
python full_pipeline.py --island Surtsey --method all --force-refresh
```

### Option B — Generation only (data file already exists)
```bash
python pipeline.py --input data/Surtsey_rag_context.json --method method3
python pipeline.py --input data/Surtsey_rag_context.json --method all

# Filenames with parentheses must be quoted in zsh
python pipeline.py --input "data/Nishinoshima_(Ogasawara)_rag_context.json" --method all
```

Output files are saved to `data/outputs/` with a timestamp.

---

## Method 3 Architecture

Methods 0–2 are single-class RAG pipelines with no agents.  
Method 3 is multi-agent — `InterSectionRAG` coordinates two specialized agents:

```
full_pipeline.py / pipeline.py
  └── InterSectionRAG.generate()
        │
        ├── Agent 2 (GraphAwareRAG)            agents/agent2_orchestrator.py
        │     ├── Rerank chunks per section (BM25 or cross-encoder)
        │     ├── Summarize each section via LLM
        │     ├── Compute NLI entailment matrix across all section pairs
        │     ├── Build directed dependency graph (DAG)
        │     ├── Remove cycles (weakest edge removal)
        │     └── Topological sort → execution order + dependency map
        │           saved to logs/agent2_plans/<island>_plan.json
        │
        └── Agent 3 (Agent3Generator)           agents/agent3_generator.py
              ├── Follow DAG execution order from Agent 2
              ├── For each section: inject dependency summaries as hard context
              └── Generate section → standard output dict for Agent 4
```

### How the NLI dependency detection works

Agent 2 uses a cross-encoder NLI model to ask: *"Does section A logically entail section B?"* for every pair of sections. For example:

- **Geology → Biology**: high entailment probability (geological substrate determines what can grow)
- **Biology → Geology**: low entailment probability (biology doesn't determine geology)

This asymmetry is what builds the directed graph. Sections with no incoming edges (like Geology) are generated first. Sections that depend on others (like Biology) receive summaries of their dependencies injected as hard context into their generation prompt, ensuring cross-section coherence.

---

## Input Format

Produced by `retrieval/rag_data_pipeline.py` and stored in `data/`:

```json
{
  "metadata": {
    "original_input": "Surtsey",
    "resolved_entity_name": "Surtsey",
    "total_sections": 4
  },
  "blueprint_data": {
    "island_name": "Surtsey",
    "sections_data": {
      "Geology": {
        "chunks": [
          {
            "chunk_id": "chunk_0001",
            "text": "...",
            "source_url": "https://...",
            "retrieval_score": 0.92
          }
        ]
      }
    }
  }
}
```

---

## Output Format

Produced by all methods, consumed by Agent 4 for evaluation:

```json
{
  "method": "method3",
  "island_name": "Surtsey",
  "metadata": {
    "reranker": "bm25",
    "rerank_strategy": "per-section",
    "use_top_l": true,
    "top_l": 5,
    "top_l_applied_at": "per-section"
  },
  "generated_article": "==Geology==\n...\n\n==Biology==\n...",
  "sections": [
    {
      "section_name": "Geology",
      "content": "...",
      "citations": ["https://..."]
    }
  ]
}
```

---

## Configuration

All settings are in `config/settings.yaml`.

| Setting | Location | Description |
|---------|----------|-------------|
| `provider` / `model` | `llm.method0–3` | LLM for generation |
| `provider` / `model` | `llm.agent2.llm` | LLM for Agent 2 summaries |
| `model_name` | `llm.agent2.graph_logic.nli_model` | NLI cross-encoder model |
| `threshold` | `llm.agent2.graph_logic.algorithm` | Entailment threshold for DAG edges (default: 0.3) |
| `reranker_type` | `methods` | `bm25`, `cross-encoder`, or `none` |
| `top_l` | `methods` | Number of chunks kept after reranking |
| `target_entity` | `pipeline_config` | Default island for `rag_data_pipeline.py` |



---

## Evaluation

## Evaluation Framework

This project evaluates generated Wikipedia-style island articles using a combination of WikiGenBench-style metrics and project-specific diagnostics. The evaluation is designed as a paired comparison: every selected island is generated by all four methods, and the resulting articles are evaluated with the same metrics.

### Methods Compared

The project compares four generation methods:

| Method | Description | Uses RAG Context | Uses Agent 2 Plan |
|---|---|---:|---:|
| `method0` | Pure generation baseline | No | No |
| `method1` | Naive RAG | Yes | No |
| `method2` | Hierarchical RAG | Yes | No |
| `method3` | Agent 2 + Agent 3 context-aware generation | Yes | Yes |

The Agent 2 plan is an island-level artifact. It is available during evaluation for all methods, but it is only used during generation by `method3`. For the other methods, it is used only as an external comparison reference for alignment and cross-section diagnostics.

---

## Evaluation Files

The evaluation code is organized under `00_evaluation/` and `agents/`.

### Main evaluation driver




## Evaluation Categories and Metrics

The evaluation is divided into **core quality metrics** and **diagnostic metrics**.

The core quality metrics are used to compare the generated articles across methods.  
The diagnostic metrics are used to explain why a method performed well or poorly, but they are not treated as main quality scores.

---

# 1. Core Evaluation Metrics

## 1.1 Writing

Writing is evaluated by **Agent 4**, an LLM-as-a-judge evaluator.

This category evaluates the generated article as a Wikipedia-style article. It does **not** compare the article to the reference Wikipedia article, and it does **not** judge factual correctness or citation correctness.

| Metric | Scale | Description |
|---|---:|---|
| `fluency_score` | 0–5 | Measures grammar, readability, sentence-level clarity, and whether the prose sounds natural. |
| `structure_score` | 0–5 | Measures section quality, heading quality, and whether the article has an appropriate Wikipedia-style structure. |
| `organization_score` | 0–5 | Measures logical flow within and across sections, coherence, and repetition control. |
| `writing_score` | 0–5 | Average of `fluency_score`, `structure_score`, and `organization_score`. |

### Interpretation

| Score Range | Meaning |
|---:|---|
| 0–1 | Very poor or unusable writing |
| 2 | Weak writing with major issues |
| 3 | Acceptable writing |
| 4 | Good writing |
| 5 | Excellent Wikipedia-style writing |

---

## 1.2 Informativeness

Informativeness measures how much useful article content is present compared with the human-written Wikipedia reference article.

This project uses two types of informativeness metrics:

1. lexical overlap metrics
2. concept-based LLM evaluation

---

### 1.2.1 Lexical Informativeness

These metrics compare the generated article against the original Wikipedia article.

| Metric | Scale | Description |
|---|---:|---|
| `rouge_l` | 0–100 | Measures longest-common-subsequence overlap between the generated article and the reference article. Higher means more overlap with the reference. |
| `meteor` | 0–100 | Measures token-level similarity between the generated article and the reference article, with more emphasis on recall than precision. |

### Interpretation

| Metric | Higher Means |
|---|---|
| `rouge_l` | The generated article shares more wording and sequence structure with the reference. |
| `meteor` | The generated article covers more reference-like content at the token level. |

These metrics are useful but imperfect because a good article may use different wording than the reference.

---

### 1.2.2 Concept-Based Informativeness

Concept-based informativeness is evaluated by **Agent 4**.  
The judge compares the generated article with the human Wikipedia reference article and scores whether the important concepts from the reference are covered accurately.

| Metric | Scale | Description |
|---|---:|---|
| `concept_coverage_score` | 0–5 | Measures how many important concepts from the reference article are included in the generated article. |
| `concept_accuracy_score` | 0–5 | Measures whether the concepts that are included are factually accurate relative to the reference article. |
| `concept_relevance_score` | 0–5 | Measures whether the generated content stays relevant to the target island and avoids unrelated material. |
| `concept_organization_score` | 0–5 | Measures whether the covered concepts are arranged in a coherent and useful article structure. |
| `concept_score` | 0–5 | Average of the four concept sub-scores. |

Optional explanatory fields:

| Field | Description |
|---|---|
| `missing_key_concepts` | Important concepts from the reference article that are missing in the generated article. |
| `inaccurate_or_unsupported_concepts` | Concepts that appear inaccurate, unsupported, or inconsistent with the reference. |
| `concept_rationale` | Short explanation of the concept scores. |

### Interpretation

| Score Range | Meaning |
|---:|---|
| 0–1 | Very poor concept coverage or severe inaccuracies |
| 2 | Weak coverage; many important concepts missing |
| 3 | Moderate coverage with some missing or weak concepts |
| 4 | Good coverage and mostly accurate concepts |
| 5 | Excellent coverage of the key reference concepts |

---

## 1.3 Verifiability

Verifiability evaluates whether the generated article’s cited claims are supported by the cited retrieved documents.

This evaluation uses the generated citation markers, the section-level citation lists, the RAG context file, and an NLI model.

| Metric | Scale | Description |
|---|---:|---|
| `citation_recall` | 0–1 | Fraction of all generated sentences that are supported by at least one citation. |
| `citation_precision` | 0–1 | Fraction of citation links that actually support the sentence they cite. |
| `citation_rate` | 0–1 | Fraction of generated sentences that contain citation markers. |

Additional count fields:

| Field | Description |
|---|---|
| `num_sentences` | Number of generated sentences evaluated. |
| `num_cited_sentences` | Number of generated sentences that contain at least one citation marker. |
| `num_citation_links` | Total number of citation links, including multiple citations in one sentence. |
| `num_supported_citation_links` | Number of citation links judged as supporting the cited sentence. |

### Interpretation

| Metric | Higher Means |
|---|---|
| `citation_recall` | More generated claims are supported by citations. |
| `citation_precision` | Citations are more likely to actually support the claims they cite. |
| `citation_rate` | More sentences contain citations. |

Notes:

- `method0` usually receives zero citation scores because it does not generate citations.
- Low `citation_precision` may mean that the model inserted citations, but the cited source does not entail the sentence.
- The NLI checker is strict, so scores may be conservative.

---

## 1.4 Cross-Sectional Consistency

Cross-sectional consistency is measured using the project-specific **CSCS** metric.

```text
CSCS = Cross-Sectional Consistency Score

```text
00_evaluation/full_evaluation.py