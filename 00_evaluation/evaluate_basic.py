"""
00_evaluation/evaluate_basic.py

Basic evaluation:
  1. Writing via Agent 4
  2. ROUGE-L
  3. METEOR

Run from project root:

    python 00_evaluation/evaluate_basic.py ^
      --input data/outputs/result_all_Surtsey_20260503_210618.json
"""

import argparse
import csv
import os
import sys
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.dirname(__file__)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EVAL_DIR)

from agents.agent4_evaluator import Agent4Evaluator
from methods.base_rag import load_config
from eval_utils import (
    discover_input_files,
    load_reference_article,
    normalize_result_file,
)
from metrics_informativeness import compute_informativeness


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "evaluations")
DEFAULT_REFERENCES_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "references")


def write_csv(output_path: str, rows: list):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp",
        "source_file",
        "reference_file",
        "island_name",
        "method",

        "writing_status",
        "fluency_score",
        "structure_score",
        "organization_score",
        "writing_score",
        "writing_rationale",
        "writing_error",

        "informativeness_status",
        "rouge_l",
        "meteor",
        "informativeness_error",
    ]

    exists = os.path.exists(output_path)

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


def evaluate(args):
    input_files = discover_input_files(args.input, args.input_dir)

    if not input_files:
        raise FileNotFoundError("No input files found. Use --input or --input-dir.")

    output_path = args.output or os.path.join(
        DEFAULT_OUTPUT_DIR,
        f"basic_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    config = load_config()
    writing_evaluator = Agent4Evaluator(config=config)

    print("\n" + "=" * 80)
    print("Basic Evaluation")
    print("=" * 80)
    print(f"Input files: {len(input_files)}")
    print(f"Output CSV:  {output_path}")
    print("=" * 80 + "\n")

    for input_file in input_files:
        print(f"\nReading: {input_file}")
        items = normalize_result_file(input_file)

        for item in items:
            island = item["island_name"]
            method = item["method"]
            article = item["generated_article"]

            print(f"Evaluating: {island} | {method}")

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source_file": item["source_file"],
                "reference_file": "",
                "island_name": island,
                "method": method,

                "writing_status": "",
                "fluency_score": "",
                "structure_score": "",
                "organization_score": "",
                "writing_score": "",
                "writing_rationale": "",
                "writing_error": "",

                "informativeness_status": "",
                "rouge_l": "",
                "meteor": "",
                "informativeness_error": "",
            }

            try:
                writing = writing_evaluator.evaluate_article(
                    article=article,
                    island_name=island,
                    method_name=method,
                )

                scores = writing.get("scores", {})

                row["writing_status"] = writing.get("status", "")
                row["fluency_score"] = scores.get("fluency_score", "")
                row["structure_score"] = scores.get("structure_score", "")
                row["organization_score"] = scores.get("organization_score", "")
                row["writing_score"] = scores.get("writing_score", "")
                row["writing_rationale"] = scores.get("brief_rationale", "")

            except Exception as e:
                row["writing_status"] = "failed"
                row["writing_error"] = str(e)

            reference, reference_file = load_reference_article(
                island,
                args.references_dir,
            )

            row["reference_file"] = reference_file

            info = compute_informativeness(article, reference)
            row.update(info)

            write_csv(output_path, [row])

            print(
                f"{island} | {method} | "
                f"fluency={row['fluency_score']} | "
                f"structure={row['structure_score']} | "
                f"organization={row['organization_score']} | "
                f"writing_avg={row['writing_score']} | "
                f"ROUGE-L={row['rouge_l']} | "
                f"METEOR={row['meteor']}"
            )

    print("\nDone.")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Basic evaluator: Agent 4 writing + ROUGE-L + METEOR."
    )

    parser.add_argument("--input", default=None, help="Path to one result JSON.")
    parser.add_argument("--input-dir", default=None, help="Directory of result JSONs.")

    parser.add_argument(
        "--references-dir",
        default=DEFAULT_REFERENCES_DIR,
        help="Directory containing human Wikipedia references.",
    )

    parser.add_argument("--output", default=None, help="Output CSV path.")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()