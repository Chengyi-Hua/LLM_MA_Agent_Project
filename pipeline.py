"""
pipeline.py — main entry point for running all methods.

Usage (always run from project root):
    python pipeline.py --method method0
    python pipeline.py --method method1
    python pipeline.py --method method2
    python pipeline.py --method method3
    python pipeline.py --method all
"""

import argparse
import json
import os
from datetime import datetime

from methods import PureGeneration, NaiveRAG, HierarchicalRAG, InterSectionRAG
from methods.base_rag import load_config
from data.mock_data import get_mock_input


METHOD_MAP = {
    "method0": PureGeneration,
    "method1": NaiveRAG,
    "method2": HierarchicalRAG,
    "method3": InterSectionRAG,
}


def run(method_name: str, input_data: dict, config: dict) -> dict:
    cls    = METHOD_MAP[method_name]
    runner = cls(config=config)

    island_name = input_data["blueprint_data"]["island_name"]
    print(f"\n[{method_name}] Generating article for: {island_name}")

    result = runner.generate(input_data)

    print(f"[{method_name}] Done. Sections generated: {len(result['sections'])}")
    return result


def save_output(results: dict, method_name: str):
    """Save results to data/outputs/ with timestamp."""
    os.makedirs("data/outputs", exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/outputs/result_{method_name}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM Multi-Agent RAG pipeline.")
    parser.add_argument(
        "--method",
        default="method1",
        choices=list(METHOD_MAP.keys()) + ["all"],
        help="Which method to run. Use 'all' to run all methods."
    )
    args = parser.parse_args()

    config     = load_config()
    input_data = get_mock_input()   # swap with real Tavily data from Eden later

    if args.method == "all":
        results = {}
        for name in METHOD_MAP:
            try:
                results[name] = run(name, input_data, config)
            except Exception as e:
                print(f"\n[{name}] ❌ Failed: {e}")
                results[name] = {"error": str(e)}
        save_output(results, "all")

    else:
        result = run(args.method, input_data, config)
        save_output({args.method: result}, args.method)


if __name__ == "__main__":
    main()