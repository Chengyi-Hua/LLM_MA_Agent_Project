"""
00_evaluation/batch_experiments.py

Batch runner for cached RAG context files.

Run from project root:

    python 00_evaluation/batch_experiments.py

Examples:

    # Run all cached islands, all methods, default config
    python 00_evaluation/batch_experiments.py

    # Run all cached islands with all reranker/scope combinations
    python 00_evaluation/batch_experiments.py --all-rerank-combos

    # Run only specific methods
    python 00_evaluation/batch_experiments.py --methods method1,method2,method3

    # Run only first 3 islands for testing
    python 00_evaluation/batch_experiments.py --limit 3

    # Re-run existing outputs instead of skipping
    python 00_evaluation/batch_experiments.py --no-resume
"""

import argparse
import copy
import csv
import itertools
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Load API keys from project-root .env
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_PATH)

if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
from methods import PureGeneration, NaiveRAG, HierarchicalRAG, InterSectionRAG
from methods.base_rag import load_config


METHOD_MAP = {
    "method0": PureGeneration,
    "method1": NaiveRAG,
    "method2": HierarchicalRAG,
    "method3": InterSectionRAG,
}


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "experiments")


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def parse_csv_list(value: str):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def discover_context_files(limit=None):
    """
    Finds all cached island RAG files in data/.
    Ignores mock_data.json and anything not ending in _rag_context.json.
    """
    files = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith("_rag_context.json"):
            continue

        path = os.path.join(DATA_DIR, fname)
        island_name = fname.replace("_rag_context.json", "").replace("_", " ")

        files.append(
            {
                "island": island_name,
                "path": path,
                "filename": fname,
            }
        )

    if limit:
        files = files[:limit]

    return files


def load_input(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_methods(args):
    methods = parse_csv_list(args.methods)

    if not methods or methods == ["all"]:
        return list(METHOD_MAP.keys())

    invalid = [m for m in methods if m not in METHOD_MAP]
    if invalid:
        raise ValueError(f"Invalid methods: {invalid}. Valid methods: {list(METHOD_MAP.keys())}")

    return methods


def build_config_combinations(args):
    if args.all_rerank_combos:
        reranker_types = ["bm25", "cross-encoder", "none"]
        rerank_scopes = ["per-section", "global"]
    else:
        reranker_types = parse_csv_list(args.reranker_types)
        rerank_scopes = parse_csv_list(args.rerank_scopes)

    top_l_values = parse_int_csv(args.top_l_values)

    combos = []

    for reranker_type, rerank_scope, top_l in itertools.product(
        reranker_types,
        rerank_scopes,
        top_l_values,
    ):
        tag = f"rerank-{reranker_type}_scope-{rerank_scope}_top{top_l}"

        combos.append(
            {
                "reranker_type": reranker_type,
                "rerank_scope": rerank_scope,
                "top_l": top_l,
                "tag": tag,
            }
        )

    return combos


def apply_combo_to_config(base_config: dict, combo: dict) -> dict:
    config = copy.deepcopy(base_config)

    config.setdefault("methods", {})
    config["methods"]["use_top_l"] = True
    config["methods"]["top_l"] = combo["top_l"]
    config["methods"]["reranker_type"] = combo["reranker_type"]
    config["methods"]["rerank_scope"] = combo["rerank_scope"]

    return config


def run_method(method_name: str, input_data: dict, config: dict) -> dict:
    cls = METHOD_MAP[method_name]
    runner = cls(config=config)

    island_name = input_data.get("blueprint_data", {}).get("island_name", "Unknown island")

    print(f"[{method_name}] Generating article for: {island_name}")
    result = runner.generate(input_data)
    print(f"[{method_name}] Done. Sections generated: {len(result.get('sections', []))}")

    return result


def write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_summary_csv(path: str, rows: list):
    if not rows:
        return

    fieldnames = [
        "timestamp",
        "experiment_name",
        "island",
        "input_json",
        "method",
        "combo_tag",
        "reranker_type",
        "rerank_scope",
        "top_l",
        "status",
        "seconds",
        "result_path",
        "error",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(args):
    context_files = discover_context_files(limit=args.limit)

    if not context_files:
        raise FileNotFoundError(
            "No *_rag_context.json files found in data/. "
            "Run retrieval first or add cached RAG context files."
        )

    methods = resolve_methods(args)
    combos = build_config_combinations(args)
    base_config = load_config()

    experiment_name = args.experiment_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    experiment_path = os.path.join(EXPERIMENT_DIR, experiment_name)
    results_dir = os.path.join(experiment_path, "results")
    manifest_path = os.path.join(experiment_path, "manifest.jsonl")
    summary_csv_path = os.path.join(experiment_path, "summary.csv")

    os.makedirs(results_dir, exist_ok=True)

    total_runs = len(context_files) * len(methods) * len(combos)

    print("\n" + "=" * 80)
    print("Batch Experiment")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Context files: {len(context_files)}")
    print(f"Methods: {methods}")
    print(f"Config combinations: {len(combos)}")
    print(f"Total runs: {total_runs}")
    print(f"Output folder: {experiment_path}")
    print("=" * 80 + "\n")

    summary_rows = []
    run_index = 0

    for context in context_files:
        island = context["island"]
        input_path = context["path"]

        print("\n" + "-" * 80)
        print(f"Loading island context: {context['filename']}")
        print("-" * 80)

        try:
            input_data = load_input(input_path)
            actual_island_name = input_data.get("blueprint_data", {}).get("island_name", island)
        except Exception as e:
            print(f"❌ Failed to load {input_path}: {e}")
            continue

        for method_name in methods:
            for combo in combos:
                run_index += 1

                island_tag = safe_name(actual_island_name)
                combo_tag = safe_name(combo["tag"])

                result_filename = f"result_{method_name}_{island_tag}_{combo_tag}.json"
                result_path = os.path.join(results_dir, result_filename)

                print(
                    f"\n[{run_index}/{total_runs}] "
                    f"{actual_island_name} | {method_name} | {combo['tag']}"
                )

                if args.resume and os.path.exists(result_path):
                    print(f"⏭️  Skipping existing result: {result_path}")

                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_name": experiment_name,
                        "island": actual_island_name,
                        "input_json": input_path,
                        "method": method_name,
                        "combo_tag": combo["tag"],
                        "reranker_type": combo["reranker_type"],
                        "rerank_scope": combo["rerank_scope"],
                        "top_l": combo["top_l"],
                        "status": "skipped_existing",
                        "seconds": 0,
                        "result_path": result_path,
                        "error": "",
                    }

                    summary_rows.append(row)
                    append_jsonl(manifest_path, row)
                    write_summary_csv(summary_csv_path, summary_rows)
                    continue

                start = time.time()

                try:
                    run_config = apply_combo_to_config(base_config, combo)

                    result = run_method(
                        method_name=method_name,
                        input_data=input_data,
                        config=run_config,
                    )

                    payload = {
                        "metadata": {
                            "experiment_name": experiment_name,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "island": actual_island_name,
                            "input_json": input_path,
                            "method": method_name,
                            "combo": combo,
                        },
                        "result": result,
                    }

                    write_json(result_path, payload)

                    seconds = round(time.time() - start, 2)

                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_name": experiment_name,
                        "island": actual_island_name,
                        "input_json": input_path,
                        "method": method_name,
                        "combo_tag": combo["tag"],
                        "reranker_type": combo["reranker_type"],
                        "rerank_scope": combo["rerank_scope"],
                        "top_l": combo["top_l"],
                        "status": "success",
                        "seconds": seconds,
                        "result_path": result_path,
                        "error": "",
                    }

                    print(f" Saved: {result_path}")

                except Exception as e:
                    seconds = round(time.time() - start, 2)

                    error_trace = traceback.format_exc()

                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_name": experiment_name,
                        "island": actual_island_name,
                        "input_json": input_path,
                        "method": method_name,
                        "combo_tag": combo["tag"],
                        "reranker_type": combo["reranker_type"],
                        "rerank_scope": combo["rerank_scope"],
                        "top_l": combo["top_l"],
                        "status": "failed",
                        "seconds": seconds,
                        "result_path": result_path,
                        "error": error_trace,
                    }

                    print(f"❌ Failed: {e}")
                    traceback.print_exc()

                summary_rows.append(row)
                append_jsonl(manifest_path, row)
                write_summary_csv(summary_csv_path, summary_rows)

    print("\n" + "=" * 80)
    print("Batch complete")
    print("=" * 80)
    print(f"Results:  {results_dir}")
    print(f"Summary:  {summary_csv_path}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run cached island RAG context files across methods/config combinations."
    )

    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods or 'all'. Example: method1,method2. Default: all.",
    )

    parser.add_argument(
        "--reranker-types",
        default="bm25",
        help="Comma-separated reranker types. Options: bm25,cross-encoder,none. Default: bm25.",
    )

    parser.add_argument(
        "--rerank-scopes",
        default="per-section",
        help="Comma-separated rerank scopes. Options: per-section,global. Default: per-section.",
    )

    parser.add_argument(
        "--top-l-values",
        default="5",
        help="Comma-separated top_l values. Example: 3,5,8. Default: 5.",
    )

    parser.add_argument(
        "--all-rerank-combos",
        action="store_true",
        help="Run bm25/cross-encoder/none × per-section/global.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N cached island files. Useful for testing.",
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional output folder name under 00_evaluation/experiments/.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip existing result files. Default: true.",
    )

    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-run even if result files already exist.",
    )

    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()