"""
00_evaluation/aggregate_evaluations.py

Aggregate per-island evaluation CSVs into method-level summary tables.

Inputs:
    00_evaluation/evaluations/per_island/*_eval.csv

Outputs:
    00_evaluation/evaluations/aggregated/all_evaluations_combined.csv
    00_evaluation/evaluations/aggregated/aggregate_by_method.csv
    00_evaluation/evaluations/aggregated/aggregate_by_method_pretty.csv
    00_evaluation/evaluations/aggregated/status_counts_by_method.csv

Run from project root:

    python 00_evaluation/aggregate_evaluations.py
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
    "aggregated",
)


# ---------------------------------------------------------------------
# Metrics to aggregate
# ---------------------------------------------------------------------

CORE_METRICS = [
    # Writing
    "fluency_score",
    "structure_score",
    "organization_score",
    "writing_score",

    # Informativeness: lexical
    "rouge_l",
    "meteor",

    # Informativeness: concept-based
    "concept_coverage_score",
    "concept_accuracy_score",
    "concept_relevance_score",
    "concept_organization_score",
    "concept_score",

    # Verifiability
    "citation_recall",
    "citation_precision",
    "citation_rate",

    # CSCS
    # This should only have values for method3.
    "cscs",
]


DIAGNOSTIC_METRICS = [
    # Generated output diagnostics
    "generated_section_count",
    "generated_empty_section_count",
    "generated_error_section_count",
    "generated_sentence_count",

    # Citation formatting diagnostics
    "citation_marker_count",
    "valid_citation_marker_count",
    "invalid_citation_marker_count",
    "sections_with_invalid_citations",
    "citation_index_validity",

    # Context artifact diagnostics
    "context_section_count",
    "context_total_chunks",
    "context_empty_chunk_section_count",
    "context_chunk_coverage",
    "context_avg_chunks_per_section",

    # Plan artifact diagnostics
    "plan_node_count",
    "plan_edge_count",
    "plan_graph_density",
    "plan_avg_dependencies",
    "plan_max_dependencies",
    "plan_source_node_count",
    "plan_missing_summary_count",
    "plan_order_violation_count",

    # Plan-output alignment diagnostics
    "planned_section_coverage",
]


STATUS_COLUMNS = [
    "writing_status",
    "informativeness_status",
    "concept_status",
    "verifiability_status",
    "cscs_status",
    "context_status",
    "plan_status",
    "plan_output_alignment_status",
]


METHOD_ORDER = ["method0", "method1", "method2", "method3"]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_per_island_csvs(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "*_eval.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No *_eval.csv files found in: {input_dir}")

    frames = []

    for path in files:
        df = pd.read_csv(path)
        df["eval_file"] = os.path.basename(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if "method" not in combined.columns:
        raise ValueError("Input CSVs must contain a 'method' column.")

    if "island_name" not in combined.columns:
        raise ValueError("Input CSVs must contain an 'island_name' column.")

    combined["method"] = combined["method"].astype(str).str.strip()

    return combined


def coerce_numeric(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in metric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_numeric_by_method(
    df: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    available_metrics = [m for m in metric_columns if m in df.columns]

    if not available_metrics:
        raise ValueError("None of the requested metric columns exist in the data.")

    grouped = df.groupby("method", dropna=False)

    rows = []

    for method, group in grouped:
        row = {
            "method": method,
            "num_rows": len(group),
            "num_islands": group["island_name"].nunique(),
        }

        for metric in available_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()

            row[f"{metric}_n"] = int(values.shape[0])

            if values.shape[0] == 0:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
                row[f"{metric}_min"] = ""
                row[f"{metric}_max"] = ""
            else:
                row[f"{metric}_mean"] = round(values.mean(), 4)
                row[f"{metric}_std"] = round(values.std(ddof=1), 4) if values.shape[0] > 1 else 0.0
                row[f"{metric}_min"] = round(values.min(), 4)
                row[f"{metric}_max"] = round(values.max(), 4)

        rows.append(row)

    out = pd.DataFrame(rows)

    out["method"] = pd.Categorical(
        out["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    out = out.sort_values("method").reset_index(drop=True)

    return out


def make_pretty_mean_std_table(
    aggregate_df: pd.DataFrame,
    metric_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for _, src in aggregate_df.iterrows():
        row = {
            "method": src["method"],
            "num_rows": src["num_rows"],
            "num_islands": src["num_islands"],
        }

        for metric in metric_columns:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            n_col = f"{metric}_n"

            if mean_col not in aggregate_df.columns:
                continue

            mean = src.get(mean_col, "")
            std = src.get(std_col, "")
            n = src.get(n_col, "")

            if pd.isna(mean) or mean == "":
                row[metric] = ""
            else:
                row[metric] = f"{float(mean):.4f} ± {float(std):.4f} (n={int(n)})"

        rows.append(row)

    return pd.DataFrame(rows)


def make_status_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    available_status_cols = [c for c in STATUS_COLUMNS if c in df.columns]

    for method, group in df.groupby("method", dropna=False):
        row = {
            "method": method,
            "num_rows": len(group),
            "num_islands": group["island_name"].nunique(),
        }

        for col in available_status_cols:
            counts = group[col].fillna("").astype(str).str.strip().value_counts()

            for status, count in counts.items():
                if status == "":
                    status = "blank"

                row[f"{col}__{status}"] = int(count)

        rows.append(row)

    out = pd.DataFrame(rows)

    out["method"] = pd.Categorical(
        out["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    out = out.sort_values("method").reset_index(drop=True)

    return out


def make_micro_verifiability_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pooled sentence-level citation metrics by method.

    Macro aggregation:
        mean citation_precision across islands

    Micro aggregation:
        total supported cited sentences / total cited sentences

    This is useful for verifiability because methods may generate different
    numbers of sentences per island.
    """
    required = [
        "method",
        "island_name",
        "citation_recall",
        "num_sentences",
        "num_cited_sentences",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Cannot compute micro verifiability. Missing columns: {missing}"
        )

    work = df.copy()

    work["citation_recall"] = pd.to_numeric(
        work["citation_recall"],
        errors="coerce",
    )
    work["num_sentences"] = pd.to_numeric(
        work["num_sentences"],
        errors="coerce",
    )
    work["num_cited_sentences"] = pd.to_numeric(
        work["num_cited_sentences"],
        errors="coerce",
    )

    # Recover sentence-level supported count:
    # citation_recall = supported_cited_sentences / num_sentences
    work["supported_cited_sentences"] = (
        work["citation_recall"] * work["num_sentences"]
    ).round()

    rows = []

    for method, group in work.groupby("method", dropna=False):
        total_sentences = group["num_sentences"].sum()
        total_cited_sentences = group["num_cited_sentences"].sum()
        total_supported_cited_sentences = group["supported_cited_sentences"].sum()

        if total_sentences > 0:
            micro_citation_rate = total_cited_sentences / total_sentences
            micro_citation_recall = total_supported_cited_sentences / total_sentences
        else:
            micro_citation_rate = ""
            micro_citation_recall = ""

        if total_cited_sentences > 0:
            micro_citation_precision = (
                total_supported_cited_sentences / total_cited_sentences
            )
        else:
            micro_citation_precision = ""

        rows.append(
            {
                "method": method,
                "num_islands": group["island_name"].nunique(),
                "total_sentences": int(total_sentences),
                "total_cited_sentences": int(total_cited_sentences),
                "total_supported_cited_sentences": int(
                    total_supported_cited_sentences
                ),
                "micro_citation_recall": (
                    round(micro_citation_recall, 4)
                    if micro_citation_recall != ""
                    else ""
                ),
                "micro_citation_precision": (
                    round(micro_citation_precision, 4)
                    if micro_citation_precision != ""
                    else ""
                ),
                "micro_citation_rate": (
                    round(micro_citation_rate, 4)
                    if micro_citation_rate != ""
                    else ""
                ),
            }
        )

    out = pd.DataFrame(rows)

    out["method"] = pd.Categorical(
        out["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return out.sort_values("method").reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-island evaluation CSVs by method."
    )

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-island *_eval.csv files.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where aggregated CSVs are written.",
    )

    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Include diagnostic metrics in the method aggregation.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metric_columns = CORE_METRICS.copy()

    if args.include_diagnostics:
        metric_columns += DIAGNOSTIC_METRICS

    print("\n" + "=" * 80)
    print("Aggregating evaluation CSVs")
    print("=" * 80)
    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Diagnostics included: {args.include_diagnostics}")
    print("=" * 80 + "\n")

    combined = load_per_island_csvs(args.input_dir)
    combined = coerce_numeric(combined, metric_columns)

    combined_path = os.path.join(args.output_dir, "all_evaluations_combined.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    aggregate = aggregate_numeric_by_method(combined, metric_columns)
    aggregate_path = os.path.join(args.output_dir, "aggregate_by_method.csv")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8")

    pretty = make_pretty_mean_std_table(aggregate, metric_columns)
    pretty_path = os.path.join(args.output_dir, "aggregate_by_method_pretty.csv")
    pretty.to_csv(pretty_path, index=False, encoding="utf-8")

    status_counts = make_status_counts(combined)
    status_path = os.path.join(args.output_dir, "status_counts_by_method.csv")
    status_counts.to_csv(status_path, index=False, encoding="utf-8")

    micro_verifiability = make_micro_verifiability_table(combined)
    micro_verifiability_path = os.path.join(
        args.output_dir,
        "micro_verifiability_by_method.csv",
    )
    micro_verifiability.to_csv(
        micro_verifiability_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved combined rows:       {combined_path}")
    print(f"Saved method aggregation:  {aggregate_path}")
    print(f"Saved pretty aggregation:  {pretty_path}")
    print(f"Saved status counts:       {status_path}")
    print(f"Saved status counts:       {status_path}")
    print(f"Saved micro verifiability: {micro_verifiability_path}")

    print("\nRows loaded:")
    print(f"  total rows:   {len(combined)}")
    print(f"  islands:      {combined['island_name'].nunique()}")
    print(f"  methods:      {', '.join(sorted(combined['method'].dropna().unique()))}")

    print("\nDone.")


if __name__ == "__main__":
    main()