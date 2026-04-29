# LLM Multi-Agent Wikipedia Generation Project

## Project Overview
This project builds a multi-agent RAG pipeline to automatically generate full-length Wikipedia-style articles for geographic islands. We compare three baseline generation methods (Methods 0–2) and propose an inter-section aware multi-agent method (Method 3) to improve cross-section coherence.

## Repository Structure
```
LLM_MA_Agent_Project/
├── methods/                         # Yen-An: Baseline RAG methods (Method 0–3)
│   ├── __init__.py                  # Package init — required for imports
│   ├── base_rag.py                  # Shared logic: LLM switching, reranking, output format
│   ├── pure_generation.py           # Method 0: No RAG
│   ├── naive_rag.py                 # Method 1: Retrieve-then-Read
│   ├── hierarchical_rag.py          # Method 2: Plan-Retrieve-Read
│   └── inter_section_rag.py         # Method 3: thin coordinator → Agent 2 + Agent 3
├── retrieval/                       # Eden: Data acquisition and preprocessing
│   └── ...
├── agents/
│   ├── __init__.py                  # Package init — required for imports
│   ├── agent2_orchestrator.py       # fuzheng: NLI, DAG, topological sort (GraphAwareRAG)
│   ├── agent3_generator.py          # Thomas: context-aware inter-section generation
│   └── agent4_evaluator.py          # Chengyi: LLM-as-judge evaluation
├── evaluation/                      # Chengyi: Evaluation metrics
│   └── ...
├── data/
│   ├── __init__.py                  # Package init — required for imports
│   ├── mock_data.py                 # Mock input for testing
│   └── outputs/                     # Generated articles saved here
├── config/
│   └── settings.yaml                # Model provider, retrieval, and NLI settings
├── pipeline.py                      # Main entry point
├── requirements.txt
└── .env.example                     # API key template
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys in .env
```

Your `.env` file should look like:
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

---

## ⚠️ Important: How to Run (VSCode / terminal)

**Always run from the project root directory.** Running files directly (e.g. `python methods/pure_generation.py`) will cause `ModuleNotFoundError: No module named 'methods'`.

```bash
# ✅ Correct — run from project root
cd /path/to/LLM_MA_Agent_Project
python pipeline.py --method method0

# ❌ Wrong — never run individual method files directly
python methods/pure_generation.py
```

For **VSCode debugging**, create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run pipeline",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/pipeline.py",
            "args": ["--method", "method3"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

---

## Run

```bash
python pipeline.py --method method0   # Method 0: No RAG (baseline)
python pipeline.py --method method1   # Method 1: Naive RAG
python pipeline.py --method method2   # Method 2: Hierarchical RAG
python pipeline.py --method method3   # Method 3: Inter-Section Aware (multi-agent)
python pipeline.py --method all       # Run all methods and compare
```

---

## Switch Model

Edit `config/settings.yaml`. Each method and agent has its own entry:

```yaml
llm:
  method3:
    provider: "openai"   # or "groq"
    model: "gpt-5.4-mini"
  agent2:
    llm:
      provider: "openai"
      model: "gpt-5.4-mini"
```

---

## Architecture: Method 3 (Multi-Agent)

Methods 0–2 are single-class RAG pipelines with no agents.
Method 3 is the multi-agent method — `InterSectionRAG` coordinates two agents:

```
pipeline.py
  └── InterSectionRAG.generate()
        │
        ├── Agent 2 (GraphAwareRAG)       agents/agent2_orchestrator.py
        │     ├── Rerank chunks per section
        │     ├── Generate NLI summaries
        │     ├── Build entailment DAG
        │     └── Topological sort → execution order + dependency map
        │
        └── Agent 3 (Agent3Generator)     agents/agent3_generator.py
              ├── Follow DAG execution order
              ├── Inject dependency summaries as hard context
              └── Generate each section → standard output for Agent 4
```

---

## Input Format (Eden → all methods)

```json
{
  "metadata": {
    "original_input": "Nishinoshima",
    "resolved_entity_name": "Nishinoshima (Ogasawara)",
    "total_sections": 3
  },
  "blueprint_data": {
    "island_name": "Nishinoshima (Ogasawara)",
    "sections_data": {
      "Geology": {
        "chunks": [
          {
            "chunk_id": "chunk_0001",
            "text": "...",
            "source_url": "https://...",
            "retrieval_score": 0.999
          }
        ]
      },
      "Ecology": {
        "chunks": [{"chunk_id": "...", "text": "...", "source_url": "...", "retrieval_score": 0.0}]
      }
    }
  }
}
```

> Chunks are pre-assigned to sections by Eden.

---

## Agent 2 Output Format (Agent 2 → Agent 3)

```json
{
  "status": "success",
  "order": ["Geology", "Climate", "Ecology", "Human Impact"],
  "dependency": {
    "Geology": [],
    "Climate": ["Geology"],
    "Ecology": ["Geology", "Climate"],
    "Human Impact": ["Ecology"]
  },
  "summaries": {
    "Geology": "Nishinoshima is an active basaltic volcano...",
    "Climate": "The island has a subtropical climate..."
  }
}
```

---

## Output Format (all methods → Agent 4 / Chengyi)

```json
{
  "method": "method3",
  "island_name": "Nishinoshima (Ogasawara)",
  "metadata": {
    "reranker": "cross-encoder",
    "rerank_strategy": "per-section",
    "use_top_l": true,
    "top_l": 5,
    "top_l_applied_at": "per-section"
  },
  "generated_article": "==Geology==\n...\n\n==Ecology==\n...",
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

## Notes for Contributors

- **fuzheng (Agent 2):** `agent2_orchestrator.py` no longer uses `google.colab`. API keys are loaded from `.env` via `load_dotenv()`. Make sure your `.env` has `OPENAI_API_KEY` and `GROQ_API_KEY` set before running locally.

- **Thomas (Method 3 / Agent 3):** `inter_section_rag.py` and `agent3_generator.py` are complete. `Agent3Generator` inherits from `HierarchicalRAG` and reuses `_call_llm()`, `_rerank_chunks()`, `_format_chunks_for_prompt()`, and `_build_output()` from `BaseRAG`. Do not add raw API calls to Agent 3.

- **Everyone:** The `__init__.py` files in `methods/`, `agents/`, and `data/` are required — do not delete them. Always run via `python pipeline.py` from the project root.