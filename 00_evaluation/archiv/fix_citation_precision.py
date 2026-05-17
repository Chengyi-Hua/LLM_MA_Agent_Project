"""
00_evaluation/fix_citation_precision.py

Recompute citation_precision in existing per-island CSV files without rerunning
the full evaluation.

Old citation_precision:
    supported citation links / total citation links

New citation_precision:
    supported cited sentences / cited sentences

This uses existing columns:
    citation_recall
    num_sentences
    num_cited_sentences

Run from project root:

    python 00_evaluation/fix_citation_precision.py

Output:
    00_evaluation/evaluations/per_island_corrected/*_eval.csv
"""

import argparse
import os
import glob
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "00_evaluation",
    "evaluations",
    "per_island",
)

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "00_evaluation",
    "evaluations",
    "per_island_corrected",
)


def to_number(series):
    return pd.to_numeric(series, errors="coerce")


def fix_file(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    required = [
        "citation_recall",
        "citation_precision",
        "num_sentences",
        "num_cited_sentences",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {missing}"
        )

    # Keep old value as diagnostic.
    df["citation_precision_old_link_level"] = df["citation_precision"]

    citation_recall = to_number(df["citation_recall"])
    num_sentences = to_number(df["num_sentences"])
    num_cited_sentences = to_number(df["num_cited_sentences"])

    # Recover supported cited sentence count from old recall:
    # citation_recall = supported_cited_sentences / total_sentences
    supported_cited_sentences = (citation_recall * num_sentences).round()

    new_precision = supported_cited_sentences / num_cited_sentences

    # If there are no cited sentences, precision is 0.
    new_precision = new_precision.where(num_cited_sentences > 0, 0.0)

    # Keep blanks if the original recall/sentence counts were missing.
    new_precision = new_precision.where(
        citation_recall.notna() & num_sentences.notna() & num_cited_sentences.notna(),
        "",
    )

    df["citation_precision"] = pd.to_numeric(new_precision, errors="coerce").round(4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Fix citation_precision in existing evaluation CSVs."
    )

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing original per-island *_eval.csv files.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where corrected CSVs are written.",
    )

    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*_eval.csv")))

    if not files:
        raise FileNotFoundError(f"No *_eval.csv files found in {args.input_dir}")

    print("\nFixing citation_precision")
    print("=" * 80)
    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Files:      {len(files)}")
    print("=" * 80)

    for input_path in files:
        output_path = os.path.join(args.output_dir, os.path.basename(input_path))
        fix_file(input_path, output_path)
        print(f"Saved corrected: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()