"""
00_evaluation/aggregate_evaluations.py

Aggregate full ablation evaluation CSVs into simple summary tables.

Expected input:
    By default, the newest per-variant CSV inside each evaluation variant folder

Outputs:
    00_evaluation/evaluations/aggregated/all_evaluations_combined.csv

    00_evaluation/evaluations/aggregated/default_by_method.csv
    00_evaluation/evaluations/aggregated/default_by_method_pretty.csv

    00_evaluation/evaluations/aggregated/method3_by_variant.csv
    00_evaluation/evaluations/aggregated/method3_by_variant_pretty.csv

    00_evaluation/evaluations/aggregated/method3_with_default_by_variant.csv
    00_evaluation/evaluations/aggregated/method3_with_default_by_variant_pretty.csv

    00_evaluation/evaluations/aggregated/status_counts_default_by_method.csv
    00_evaluation/evaluations/aggregated/status_counts_method3_by_variant.csv

    00_evaluation/evaluations/aggregated/micro_verifiability_default_by_method.csv
    00_evaluation/evaluations/aggregated/micro_verifiability_method3_by_variant.csv

Run from project root:

    python 00_evaluation/aggregate_evaluations.py

Optionally specify one full evaluation CSV:

    python 00_evaluation/aggregate_evaluations.py ^
      --input-csv 00_evaluation/evaluations/full_evaluation_all_variants_YYYYMMDD_HHMMSS.csv
"""

import argparse
import glob
import os
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_EVALUATIONS_DIR = os.path.join(
    PROJECT_ROOT,
    "00_evaluation",
    "evaluations",
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
    "citation_link_precision",
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

def find_latest_csv(pattern: str):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load_variant_folder_csvs(evaluations_dir: str) -> pd.DataFrame:
    """
    Load the newest full_evaluation_*.csv from each variant folder.

    Expected structure:

        evaluations/
          default/
            full_evaluation_default_*.csv
          agent2_model_deepseek_v3.2/
            full_evaluation_agent2_model_deepseek_v3.2_*.csv
          threshold_0.3/
            full_evaluation_threshold_0.3_*.csv

    This intentionally ignores:
        evaluations/full_evaluation_all_variants_*.csv

    so we do not duplicate rows.
    """

    variant_dirs = sorted(
        path
        for path in glob.glob(os.path.join(evaluations_dir, "*"))
        if os.path.isdir(path)
        and os.path.basename(path).lower() != "aggregated"
    )

    files = []

    for variant_dir in variant_dirs:
        latest = find_latest_csv(
            os.path.join(variant_dir, "full_evaluation_*.csv")
        )

        if latest:
            files.append(latest)

    if not files:
        raise FileNotFoundError(
            f"No per-variant full_evaluation_*.csv files found under: {evaluations_dir}"
        )

    frames = []

    for path in files:
        df = pd.read_csv(path)
        df["eval_file"] = os.path.basename(path)
        df["eval_folder"] = os.path.basename(os.path.dirname(path))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    required = ["variant_name", "island_name", "method"]
    missing = [col for col in required if col not in combined.columns]

    if missing:
        raise ValueError(f"Input CSVs missing required columns: {missing}")

    combined["variant_name"] = combined["variant_name"].fillna("").astype(str).str.strip()
    combined["method"] = combined["method"].fillna("").astype(str).str.strip()
    combined["island_name"] = combined["island_name"].fillna("").astype(str).str.strip()

    return combined


def find_latest_full_evaluation_csv(evaluations_dir: str) -> str:
    pattern = os.path.join(evaluations_dir, "full_evaluation_all_variants_*.csv")
    latest = find_latest_csv(pattern)

    if not latest:
        raise FileNotFoundError(
            f"No full_evaluation_all_variants_*.csv found in: {evaluations_dir}"
        )

    return latest


def load_master_evaluation_csv(input_csv, evaluations_dir: str) -> pd.DataFrame:
    if input_csv:
        path = os.path.abspath(input_csv)
    else:
        path = find_latest_full_evaluation_csv(evaluations_dir)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")

    df = pd.read_csv(path)
    df["eval_file"] = os.path.basename(path)
    df["eval_folder"] = "master"

    required = ["variant_name", "island_name", "method"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df["variant_name"] = df["variant_name"].fillna("").astype(str).str.strip()
    df["method"] = df["method"].fillna("").astype(str).str.strip()
    df["island_name"] = df["island_name"].fillna("").astype(str).str.strip()

    return df


def load_evaluation_data(input_csv, evaluations_dir: str, input_mode: str) -> pd.DataFrame:
    """
    input_mode:
      - variant-folders: read newest CSV from each variant folder
      - master: read full_evaluation_all_variants_*.csv
      - auto: try variant folders first, then master
    """

    if input_csv:
        return load_master_evaluation_csv(input_csv, evaluations_dir)

    if input_mode == "master":
        return load_master_evaluation_csv(None, evaluations_dir)

    if input_mode == "variant-folders":
        return load_variant_folder_csvs(evaluations_dir)

    try:
        return load_variant_folder_csvs(evaluations_dir)
    except FileNotFoundError:
        return load_master_evaluation_csv(None, evaluations_dir)


def coerce_numeric(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in metric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_comparison_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a readable label for method3 comparisons.

    default + method3 becomes:
        default_method3

    all other method3 ablations become:
        variant_name
    """

    df = df.copy()

    def label(row):
        variant = str(row.get("variant_name", "")).strip()
        method = str(row.get("method", "")).strip()

        if variant.lower() == "default":
            return f"default_{method}"

        return variant

    df["comparison_label"] = df.apply(label, axis=1)

    return df


def sort_default_methods(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        return df

    out = df.copy()

    out["method"] = pd.Categorical(
        out["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return out.sort_values("method").reset_index(drop=True)


def sort_method3_variants(df: pd.DataFrame) -> pd.DataFrame:
    if "comparison_label" not in df.columns:
        return df

    out = df.copy()

    def sort_key(value):
        value = str(value)
        if value == "default_method3":
            return "000_default_method3"
        return f"100_{value.lower()}"

    out["_sort_key"] = out["comparison_label"].map(sort_key)

    out = out.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_numeric(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    available_metrics = [m for m in metric_columns if m in df.columns]

    if not available_metrics:
        raise ValueError("None of the requested metric columns exist in the data.")

    grouped = df.groupby(group_cols, dropna=False)

    rows = []

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            col: value for col, value in zip(group_cols, group_key)
        }

        row["num_rows"] = len(group)
        row["num_islands"] = group["island_name"].nunique()

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
                row[f"{metric}_std"] = (
                    round(values.std(ddof=1), 4)
                    if values.shape[0] > 1
                    else 0.0
                )
                row[f"{metric}_min"] = round(values.min(), 4)
                row[f"{metric}_max"] = round(values.max(), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def make_pretty_mean_std_table(
    aggregate_df: pd.DataFrame,
    group_cols: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for _, src in aggregate_df.iterrows():
        row = {col: src[col] for col in group_cols if col in src}

        row["num_rows"] = src["num_rows"]
        row["num_islands"] = src["num_islands"]

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


def make_status_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    available_status_cols = [c for c in STATUS_COLUMNS if c in df.columns]

    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            col: value for col, value in zip(group_cols, group_key)
        }

        row["num_rows"] = len(group)
        row["num_islands"] = group["island_name"].nunique()

        for col in available_status_cols:
            counts = group[col].fillna("").astype(str).str.strip().value_counts()

            for status, count in counts.items():
                if status == "":
                    status = "blank"

                row[f"{col}__{status}"] = int(count)

        rows.append(row)

    return pd.DataFrame(rows)


def make_micro_verifiability_table(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Compute pooled sentence-level citation metrics.

    Macro aggregation:
        mean citation_precision across islands

    Micro aggregation:
        total supported cited sentences / total cited sentences
    """

    required = [
        "island_name",
        "citation_recall",
        "num_sentences",
        "num_cited_sentences",
    ]

    missing = [col for col in required + group_cols if col not in df.columns]

    if missing:
        print(f"Skipping micro verifiability. Missing columns: {missing}")
        return pd.DataFrame()

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

    for group_key, group in work.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            col: value for col, value in zip(group_cols, group_key)
        }

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

        row.update(
            {
                "num_islands": group["island_name"].nunique(),
                "total_sentences": int(total_sentences),
                "total_cited_sentences": int(total_cited_sentences),
                "total_supported_cited_sentences": int(total_supported_cited_sentences),
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

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate full ablation evaluation CSVs."
    )

    parser.add_argument(
        "--input-csv",
        default=None,
        help=(
            "Path to full_evaluation_all_variants_*.csv. "
            "If omitted, the newest file in evaluations/ is used."
        ),
    )


    parser.add_argument(
        "--input-mode",
        choices=["auto", "master", "variant-folders"],
        default="variant-folders",
        help=(
            "How to load evaluation rows. "
            "'variant-folders' reads the newest CSV from each variant folder. "
            "'master' reads full_evaluation_all_variants_*.csv. "
            "'auto' tries variant folders first, then master."
        ),
    )
    parser.add_argument(
        "--evaluations-dir",
        default=DEFAULT_EVALUATIONS_DIR,
        help="Directory containing full_evaluation_all_variants_*.csv files.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where aggregated CSVs are written.",
    )

    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Include diagnostic metrics in the aggregation.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metric_columns = CORE_METRICS.copy()

    if args.include_diagnostics:
        metric_columns += DIAGNOSTIC_METRICS

    print("\n" + "=" * 80)
    print("Aggregating full evaluation CSV")
    print("=" * 80)
    print(f"Input CSV:    {args.input_csv or 'not provided'}")
    print(f"Input mode:   {args.input_mode}")
    print(f"Input dir:    {args.evaluations_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Diagnostics:  {args.include_diagnostics}")
    print("=" * 80 + "\n")

    combined = load_evaluation_data(
        input_csv=args.input_csv,
        evaluations_dir=args.evaluations_dir,
        input_mode=args.input_mode,
    )
    combined = make_comparison_label(combined)
    combined = coerce_numeric(combined, metric_columns)

    combined_path = os.path.join(args.output_dir, "all_evaluations_combined.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    # -----------------------------------------------------------------
    # 1. Default only: compare method0, method1, method2, method3
    # -----------------------------------------------------------------
    default_df = combined[
        combined["variant_name"].str.lower() == "default"
    ].copy()

    default_group_cols = ["method"]

    default_agg = aggregate_numeric(
        default_df,
        default_group_cols,
        metric_columns,
    )
    default_agg = sort_default_methods(default_agg)

    default_agg_path = os.path.join(args.output_dir, "default_by_method.csv")
    default_agg.to_csv(default_agg_path, index=False, encoding="utf-8")

    default_pretty = make_pretty_mean_std_table(
        default_agg,
        default_group_cols,
        metric_columns,
    )
    default_pretty = sort_default_methods(default_pretty)

    default_pretty_path = os.path.join(
        args.output_dir,
        "default_by_method_pretty.csv",
    )
    default_pretty.to_csv(default_pretty_path, index=False, encoding="utf-8")

    default_status = make_status_counts(default_df, default_group_cols)
    default_status = sort_default_methods(default_status)

    default_status_path = os.path.join(
        args.output_dir,
        "status_counts_default_by_method.csv",
    )
    default_status.to_csv(default_status_path, index=False, encoding="utf-8")

    default_micro = make_micro_verifiability_table(default_df, default_group_cols)
    default_micro = sort_default_methods(default_micro)

    default_micro_path = os.path.join(
        args.output_dir,
        "micro_verifiability_default_by_method.csv",
    )
    default_micro.to_csv(default_micro_path, index=False, encoding="utf-8")

    # -----------------------------------------------------------------
    # 2. Method3 ablations only: exclude default, compare variant folders
    # -----------------------------------------------------------------
    method3_ablation_df = combined[
        (combined["method"].str.lower() == "method3")
        & (combined["variant_name"].str.lower() != "default")
    ].copy()

    variant_group_cols = ["comparison_label", "variant_name", "variant_type"]

    method3_agg = aggregate_numeric(
        method3_ablation_df,
        variant_group_cols,
        metric_columns,
    )
    method3_agg = sort_method3_variants(method3_agg)

    method3_agg_path = os.path.join(args.output_dir, "method3_by_variant.csv")
    method3_agg.to_csv(method3_agg_path, index=False, encoding="utf-8")

    method3_pretty = make_pretty_mean_std_table(
        method3_agg,
        variant_group_cols,
        metric_columns,
    )
    method3_pretty = sort_method3_variants(method3_pretty)

    method3_pretty_path = os.path.join(
        args.output_dir,
        "method3_by_variant_pretty.csv",
    )
    method3_pretty.to_csv(method3_pretty_path, index=False, encoding="utf-8")

    method3_status = make_status_counts(method3_ablation_df, variant_group_cols)
    method3_status = sort_method3_variants(method3_status)

    method3_status_path = os.path.join(
        args.output_dir,
        "status_counts_method3_by_variant.csv",
    )
    method3_status.to_csv(method3_status_path, index=False, encoding="utf-8")

    method3_micro = make_micro_verifiability_table(
        method3_ablation_df,
        variant_group_cols,
    )
    method3_micro = sort_method3_variants(method3_micro)

    method3_micro_path = os.path.join(
        args.output_dir,
        "micro_verifiability_method3_by_variant.csv",
    )
    method3_micro.to_csv(method3_micro_path, index=False, encoding="utf-8")

    # -----------------------------------------------------------------
    # 3. Method3 comparison with default method3 included first
    # -----------------------------------------------------------------
    method3_with_default_df = combined[
        combined["method"].str.lower() == "method3"
    ].copy()

    method3_with_default_agg = aggregate_numeric(
        method3_with_default_df,
        variant_group_cols,
        metric_columns,
    )
    method3_with_default_agg = sort_method3_variants(method3_with_default_agg)

    method3_with_default_agg_path = os.path.join(
        args.output_dir,
        "method3_with_default_by_variant.csv",
    )
    method3_with_default_agg.to_csv(
        method3_with_default_agg_path,
        index=False,
        encoding="utf-8",
    )

    method3_with_default_pretty = make_pretty_mean_std_table(
        method3_with_default_agg,
        variant_group_cols,
        metric_columns,
    )
    method3_with_default_pretty = sort_method3_variants(
        method3_with_default_pretty
    )

    method3_with_default_pretty_path = os.path.join(
        args.output_dir,
        "method3_with_default_by_variant_pretty.csv",
    )
    method3_with_default_pretty.to_csv(
        method3_with_default_pretty_path,
        index=False,
        encoding="utf-8",
    )

    print("Saved:")
    print(f"  combined rows:                    {combined_path}")
    print(f"  default by method:                {default_agg_path}")
    print(f"  default by method pretty:         {default_pretty_path}")
    print(f"  method3 by variant:               {method3_agg_path}")
    print(f"  method3 by variant pretty:        {method3_pretty_path}")
    print(f"  method3 + default by variant:     {method3_with_default_agg_path}")
    print(f"  method3 + default pretty:         {method3_with_default_pretty_path}")
    print(f"  default status counts:            {default_status_path}")
    print(f"  method3 status counts:            {method3_status_path}")
    print(f"  default micro verifiability:      {default_micro_path}")
    print(f"  method3 micro verifiability:      {method3_micro_path}")

    print("\nRows loaded:")
    print(f"  total rows:        {len(combined)}")
    print(f"  islands:           {combined['island_name'].nunique()}")
    print(f"  variants:          {combined['variant_name'].nunique()}")
    print(f"  methods:           {', '.join(sorted(combined['method'].dropna().unique()))}")
    print(f"  default rows:      {len(default_df)}")
    print(f"  method3 ablations: {len(method3_ablation_df)}")
    print(f"  method3 total:     {len(method3_with_default_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()