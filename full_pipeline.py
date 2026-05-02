"""
full_pipeline.py — end-to-end pipeline: island name → generated article.

Combines rag_data_pipeline.py (data collection) and pipeline.py (generation).
Data is cached in data/ so repeated runs skip the retrieval step.

Usage (always run from project root):
    python full_pipeline.py --island Surtsey
    python full_pipeline.py --island Surtsey --method method3
    python full_pipeline.py --island Hawaii --method all
    python full_pipeline.py --island Surtsey --method all --force-refresh
"""

import argparse
import json
import os
import sys
from datetime import datetime

from methods import PureGeneration, NaiveRAG, HierarchicalRAG, InterSectionRAG
from methods.base_rag import load_config

# Import data collection from retrieval/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "retrieval"))
from retrieval.rag_data_pipeline import run_rag_pipeline


METHOD_MAP = {
    "method0": PureGeneration,
    "method1": NaiveRAG,
    "method2": HierarchicalRAG,
    "method3": InterSectionRAG,
}

DATA_DIR = "data"


# ── Step 1: Data collection ────────────────────────────────────────────────────

def get_or_fetch_data(island: str, force_refresh: bool = False) -> tuple:
    """
    Returns (input_data, json_path).
    If a cached JSON already exists in data/ and force_refresh is False, loads it.
    Otherwise runs the full Tavily retrieval pipeline.
    """
    # Search for existing file matching this island name
    safe_name = island.replace(" ", "_").replace("/", "_")
    cached_path = None

    for fname in os.listdir(DATA_DIR):
        if fname.endswith("_rag_context.json") and safe_name.lower() in fname.lower():
            cached_path = os.path.join(DATA_DIR, fname)
            break

    if cached_path and not force_refresh:
        print(f"✅ Found cached data: {cached_path}")
        print("   (Use --force-refresh to re-fetch from Tavily)\n")
        with open(cached_path, "r", encoding="utf-8") as f:
            return json.load(f), cached_path

    # No cache — run retrieval
    print(f"🌐 No cached data found for '{island}'. Running Tavily retrieval...\n")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise EnvironmentError(
            "TAVILY_API_KEY not found in .env\n"
            "Add it to your .env file to enable data collection."
        )

    result = run_rag_pipeline(island, tavily_key)
    if result is None:
        raise ValueError(
            f"Data collection failed for '{island}'.\n"
            f"Check that it is a valid Wikipedia entity."
        )

    # Find the file that was just saved
    exact_name = result["metadata"]["resolved_entity_name"]
    safe_exact = exact_name.replace(" ", "_").replace("/", "_")
    json_path  = os.path.join(DATA_DIR, f"{safe_exact}_rag_context.json")
    return result, json_path


# ── Step 2: Generation ─────────────────────────────────────────────────────────

def run_method(method_name: str, input_data: dict, config: dict) -> dict:
    cls         = METHOD_MAP[method_name]
    runner      = cls(config=config)
    island_name = input_data["blueprint_data"]["island_name"]

    print(f"\n[{method_name}] Generating article for: {island_name}")
    result = runner.generate(input_data)
    print(f"[{method_name}] Done. Sections: {len(result['sections'])}")
    return result


def save_output(results: dict, method_name: str, island: str):
    os.makedirs("data/outputs", exist_ok=True)
    safe_island = island.replace(" ", "_").replace("/", "_")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/outputs/result_{method_name}_{safe_island}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Output saved to {output_path}")
    return output_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: island name → generated Wikipedia article."
    )
    parser.add_argument(
        "--island",
        required=True,
        help="Island name to generate an article for. Example: Surtsey"
    )
    parser.add_argument(
        "--method",
        default="method1",
        choices=list(METHOD_MAP.keys()) + ["all"],
        help="Which generation method to use. Default: method1."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch data from Tavily even if a cached file already exists."
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    config = load_config()

    # Step 1: get data (cached or fresh)
    print(f"\n{'='*60}")
    print(f"Full Pipeline | island: {args.island} | method: {args.method}")
    print(f"{'='*60}\n")

    input_data, json_path = get_or_fetch_data(args.island, args.force_refresh)

    # Step 2: generate
    if args.method == "all":
        results = {}
        for name in METHOD_MAP:
            try:
                results[name] = run_method(name, input_data, config)
            except Exception as e:
                print(f"\n[{name}] ❌ Failed: {e}")
                results[name] = {"error": str(e)}
        save_output(results, "all", args.island)

    else:
        result = run_method(args.method, input_data, config)
        save_output({args.method: result}, args.method, args.island)


if __name__ == "__main__":
    main()