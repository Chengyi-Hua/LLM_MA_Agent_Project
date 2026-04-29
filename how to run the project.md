# How to Run the Project

## ❌ The wrong way (causes ModuleNotFoundError)
```bash
python methods/pure_generation.py
python methods/hierarchical_rag.py
```
This fails because Python doesn't know where `methods/` is as a package.

## ✅ The correct way — always run from the project root

### Run the full pipeline
```bash
cd /path/to/LLM_MA_Agent_Project
python pipeline.py --method method0
python pipeline.py --method method1
python pipeline.py --method method2
python pipeline.py --method method3
python pipeline.py --method all
```

### Run a specific method file directly (using -m flag)
```bash
cd /path/to/LLM_MA_Agent_Project
python -m methods.pure_generation
python -m methods.hierarchical_rag
python -m agents.agent3_generator
```

### In VSCode — set the correct working directory
Open `.vscode/launch.json` and make sure `cwd` points to the project root:
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
            "cwd": "${workspaceFolder}",
            "env": {}
        }
    ]
}
```

## Required folder structure (all __init__.py must exist)
```
LLM_MA_Agent_Project/
├── methods/
│   ├── __init__.py          ← required
│   ├── base_rag.py
│   ├── pure_generation.py
│   ├── naive_rag.py
│   ├── hierarchical_rag.py
│   └── inter_section_rag.py
├── agents/
│   ├── __init__.py          ← required
│   ├── agent2_orchestrator.py
│   ├── agent3_generator.py
│   └── agent4_evaluator.py
├── data/
│   ├── __init__.py          ← required
│   └── mock_data.py
├── config/
│   └── settings.yaml
├── .env                     ← your API keys go here
└── pipeline.py
```

## .env file (in project root)
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```