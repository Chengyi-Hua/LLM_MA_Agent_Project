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

## Agent 4 — Evaluation

Agent 4 receives the output JSON from `data/outputs/` and evaluates each generated article using LLM-as-judge on three dimensions:
