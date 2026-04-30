"""
pipeline.py — main entry point for the LLM Multi-Agent RAG pipeline.

Usage (always run from project root):
    python pipeline.py --input data/Surtsey_rag_context.json
    python pipeline.py --input "data/Nishinoshima_(Ogasawara)_rag_context.json"  
    python pipeline.py --input data/Surtsey_rag_context.json --method method3
    python pipeline.py --input data/Surtsey_rag_context.json --method all
"""

import argparse
import json
import os
from datetime import datetime

from methods import PureGeneration, NaiveRAG, HierarchicalRAG, InterSectionRAG
from methods.base_rag import load_config


METHOD_MAP = {
    "method0": PureGeneration,
    "method1": NaiveRAG,
    "method2": HierarchicalRAG,
    "method3": InterSectionRAG,
}


def load_input(json_path: str) -> dict:
    """Load a RAG context JSON file produced by rag_data_pipeline.py."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Input file not found: {json_path}\n"
            f"Run retrieval/rag_data_pipeline.py first to generate the data file."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(method_name: str, input_data: dict, config: dict) -> dict:
    cls         = METHOD_MAP[method_name]
    runner      = cls(config=config)
    island_name = input_data["blueprint_data"]["island_name"]

    print(f"\n[{method_name}] Generating article for: {island_name}")
    result = runner.generate(input_data)
    print(f"[{method_name}] Done. Sections generated: {len(result['sections'])}")
    return result


def save_output(results: dict, method_name: str, input_path: str):
    """Save results to data/outputs/ with timestamp."""
    os.makedirs("data/outputs", exist_ok=True)
    island_tag  = os.path.basename(input_path).replace("_rag_context.json", "")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/outputs/result_{method_name}_{island_tag}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM Multi-Agent RAG pipeline.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the RAG context JSON file (produced by rag_data_pipeline.py). "
             "Example: data/Surtsey_rag_context.json"
    )
    parser.add_argument(
        "--method",
        default="method1",
        choices=list(METHOD_MAP.keys()) + ["all"],
        help="Which method to run. Default: method1. Use 'all' to run all methods."
    )
    args = parser.parse_args()

    config     = load_config()
    input_data = load_input(args.input)

    if args.method == "all":
        results = {}
        for name in METHOD_MAP:
            try:
                results[name] = run(name, input_data, config)
            except Exception as e:
                print(f"\n[{name}] ❌ Failed: {e}")
                results[name] = {"error": str(e)}
        save_output(results, "all", args.input)

    else:
        result = run(args.method, input_data, config)
        save_output({args.method: result}, args.method, args.input)


if __name__ == "__main__":
    main()