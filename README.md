# LLM Multi-Agent Wikipedia Generation Project

## Project Overview
This project builds a multi-agent RAG pipeline to automatically generate full-length Wikipedia-style articles for geographic islands. We compare three generation methods (Method 0-2) and propose an inter-section aware method (Method 3) to improve cross-section coherence.

## Repository Structure
```
LLM_MA_Agent_Project/
├── methods/                         # Yen-An: Baseline RAG methods (Method 0-2)
│   ├── base_rag.py                  # Shared logic: LLM switching, reranking, output format
│   ├── pure_generation.py           # Method 0: No RAG
│   ├── naive_rag.py                 # Method 1: Retrieve-then-Read
│   ├── hierarchical_rag.py          # Method 2: Plan-Retrieve-Read
│   └── inter_section_rag.py         # Method 3: Thomas (inherit from HierarchicalRAG)
├── retrieval/                       # Eden: Data acquisition and preprocessing
│   └── ...
├── agents/
│   ├── agent2_orchestrator.py       # fuzheng: NLI, DAG, topological sort
│   ├── agent3_generator.py          # Thomas: context-aware inter-section generation
│   └── agent4_evaluator.py          # Chengyi: LLM-as-judge evaluation
├── evaluation/                      # Chengyi: Evaluation metrics
│   └── ...
├── data/
│   ├── mock_data.py                 # Mock input for testing
│   └── outputs/                     # Generated articles saved here
├── config/
│   └── settings.yaml                # Model provider and retrieval settings
├── pipeline.py                      # Main entry point
├── requirements.txt
└── .env.example                     # API key template
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys in .env
```

## Switch Model
Edit `config/settings.yaml`:
```yaml
llm:
  provider: "groq"   # or "openai"
```

## Run
```bash
python pipeline.py --method method0
python pipeline.py --method method1
python pipeline.py --method method2
python pipeline.py --method method3  # Thomas: implement inter_section_rag.py first
python pipeline.py --method all
```

## Input Format (Eden → everyone)
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
        "chunks": [...]
      }
    }
  }
}
```
Note: chunks are pre-assigned to sections by Eden.

## Output Format (Yen-An → Chengyi)
```json
{
  "method": "method1",
  "island_name": "Nishinoshima",
  "generated_article": "==Geology==\n...",
  "sections": [
    {
      "section_name": "Geology",
      "content": "...",
      "citations": ["https://..."]
    }
  ]
}
```

## For Thomas (Method 3)
Create `methods/inter_section_rag.py`. Inherit `HierarchicalRAG` and override `_generate_section()`.
Use the `context` parameter to inject summaries from previously generated sections:
```python
from methods.hierarchical_rag import HierarchicalRAG

class InterSectionRAG(HierarchicalRAG):
    def _generate_section(self, island_name, section, all_chunks, context=""):
        # context = summaries from previously generated sections
        ...
```
Then add Method 3 to `pipeline.py` under `METHOD_MAP`.