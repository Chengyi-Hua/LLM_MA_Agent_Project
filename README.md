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
│   ├── evaluations/                          # Evaluation CSV outputs and aggregated summaries
│   ├── references/                           # Human Wikipedia reference articles
│   ├── eval_utils.py                         # Shared helpers for loading files, references, citations, and text normalization
│   ├── evaluate_basic.py                     # Basic evaluation: Agent 4 writing + ROUGE-L + METEOR
│   ├── fetch_wikipedia_references.py         # Fetches original Wikipedia articles for references/
│   ├── full_evaluation.py                    # Main full evaluation pipeline
│   ├── compute_non_agent4_metrics_if_needed  # Recomputes non-Agent-4 metrics while preserving writing/concept scores
│   ├── aggregate_evaluations.py              # Aggregates per-island evaluation CSVs into method-level summaries
│   ├── metrics_artifact_diagnostics.py       # Artifact, output, citation-index, context, and plan diagnostics
│   ├── metrics_cscs.py                       # Cross-Sectional Consistency Score for method3
│   ├── metrics_informativeness.py            # ROUGE-L and METEOR with citation markers removed
│   └── metrics_verifiability.py              # Citation rate, sentence support, link precision, and NLI checking
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
# Evaluation Summary

This project evaluates generated Wikipedia-style island articles across four generation methods. Each selected island is generated by all methods, and all outputs are evaluated with the same metric pipeline.

The evaluation focuses on five article-level quality dimensions:

1. Writing quality
2. Informativeness
3. Concept-based evaluation
4. Citation and verifiability quality
5. CSCS cross-sectional consistency for Method 3

Each row in the evaluation CSV corresponds to one generated article for one island and one method.

---

## Methods Compared

| Method | Description | Uses RAG Context | Uses Agent 2 Plan |
|---|---|---:|---:|
| `method0` | Pure generation baseline | No | No |
| `method1` | Naive RAG | Yes | No |
| `method2` | Hierarchical RAG | Yes | No |
| `method3` | Agent 2 dependency planning + Agent 3 context-aware generation | Yes | Yes |

`method3` is the only method that uses the Agent 2 dependency plan during generation. Therefore, CSCS is computed only for `method3`.

---

## Evaluation Files

| File | Purpose |
|---|---|
| `00_evaluation/full_evaluation.py` | Runs the full evaluation pipeline. |
| `00_evaluation/evaluate_basic.py` | Runs writing, ROUGE-L, and METEOR only. |
| `00_evaluation/recompute_non_agent4_metrics.py` | Recomputes non-Agent-4 metrics while keeping existing writing and concept scores. |
| `00_evaluation/aggregate_evaluations.py` | Aggregates per-island results into method-level summaries. |
| `00_evaluation/error_analysis.py` | Converts metric results into error classes and frequency tables. |

Metric implementations:

| File | Metrics |
|---|---|
| `00_evaluation/metrics_informativeness.py` | ROUGE-L and METEOR |
| `00_evaluation/metrics_verifiability.py` | Citation and verifiability metrics |
| `00_evaluation/metrics_cscs.py` | CSCS for Method 3 |
| `agents/agent4_evaluator.py` | Writing and concept-based LLM evaluation |

---

## Metric Groups

| Metric Group | What It Measures | Main Columns |
|---|---|---|
| Writing quality | Fluency, structure, and organization | `fluency_score`, `structure_score`, `organization_score`, `writing_score` |
| ROUGE-L / METEOR | Lexical overlap with the Wikipedia reference article | `rouge_l`, `meteor` |
| Concept-based evaluation | Semantic coverage, accuracy, relevance, and organization | `concept_coverage_score`, `concept_accuracy_score`, `concept_relevance_score`, `concept_organization_score`, `concept_score` |
| Citation / verifiability | Whether cited claims are supported by cited evidence | `citation_rate`, `citation_precision`, `citation_recall`, `citation_link_precision` |
| CSCS | Whether Method 3 preserves dependency information across sections | `cscs` |

---

## Interpretation Principles

The evaluation should be interpreted as a multi-dimensional comparison.

Key points:

- High writing quality does not imply factual grounding.
- High citation rate does not imply citation correctness.
- ROUGE-L and METEOR measure lexical overlap, not full semantic correctness.
- Concept scores measure reference-based content quality.
- Citation precision measures whether cited claims are supported.
- CSCS applies only to `method3`.

No single metric determines the best method. Different methods may perform better on different quality dimensions.

## Error Analysis

Error analysis converts the raw evaluation metrics into interpretable failure categories. More information see the dedicated readme

The script:

```bash
00_evaluation/error_analysis.py