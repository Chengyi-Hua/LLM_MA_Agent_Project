"""
pipeline.py — main entry point for running all methods.
Usage:
    python pipeline.py --method method0
    python pipeline.py --method method1
    python pipeline.py --method method2
    python pipeline.py --method method3  # Thomas: implement inter_section_rag.py first
    python pipeline.py --method all
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from methods import PureGeneration, NaiveRAG, HierarchicalRAG
from methods.base_rag import load_config


METHOD_MAP = {
    "method0": PureGeneration,
    "method1": NaiveRAG,
    "method2": HierarchicalRAG,
    # method3: Thomas adds InterSectionRAG here
    # "method3": InterSectionRAG,
}


def load_input(path: str = "data/mock_data.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def run(method_name: str, input_data: dict, config: dict) -> dict:
    cls = METHOD_MAP[method_name]
    runner = cls(config=config)
    print(f"\n[{method_name}] Generating article for: {input_data['blueprint_data']['island_name']}")
    result = runner.generate(input_data)
    print(f"[{method_name}] Done. Sections generated: {len(result['sections'])}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="method1",
        choices=list(METHOD_MAP.keys()) + ["all"]
    )
    parser.add_argument(
        "--input",
        default="data/mock_data.json",
        help="Path to input JSON file"
    )
    args = parser.parse_args()

    config = load_config()
    input_data = load_input(args.input)

    if args.method == "all":
        results = {}
        for name in METHOD_MAP:
            results[name] = run(name, input_data, config)
    else:
        results = {args.method: run(args.method, input_data, config)}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/outputs/result_{args.method}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()