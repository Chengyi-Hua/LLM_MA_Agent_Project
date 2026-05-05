"""
00_evaluation/recompute_non_agent4_metrics.py

Recompute all non-Agent-4 evaluation metrics in existing per-island CSVs.

This preserves only the costly Agent 4 evaluation columns:
  - writing scores
  - writing rationales
  - concept scores
  - concept rationales

It recomputes:
  - reference file
  - ROUGE-L
  - METEOR
  - context file
  - plan file
  - artifact / pipeline diagnostics
  - citation diagnostics
  - verifiability metrics
  - citation_link_precision
  - CSCS
  - cscs_checked_edges

Run from project root:

    python 00_evaluation/recompute_non_agent4_metrics.py ^
      --input-dir "00_evaluation/evaluations/per_island_verifiability_claimsplit" ^
      --output-dir "00_evaluation/evaluations/per_island_non_agent4_corrected"

Then aggregate:

    python 00_evaluation/aggregate_evaluations.py ^
      --input-dir "00_evaluation/evaluations/per_island_non_agent4_corrected" ^
      --output-dir "00_evaluation/evaluations/aggregated_non_agent4_corrected"
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.dirname(__file__)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EVAL_DIR)

from methods.base_rag import load_config

from eval_utils import (
    load_reference_article,
    normalize_result_file,
)

from metrics_informativeness import compute_informativeness

from metrics_verifiability import (
    NLIEvaluator,
    compute_verifiability,
    infer_context_path,
    load_url_text_map,
)

from metrics_cscs import compute_cscs, load_agent2_plan

from metrics_artifact_diagnostics import compute_artifact_diagnostics


DEFAULT_PRIMARY_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "00_evaluation",
    "evaluations",
    "per_island_verifiability_claimsplit",
)

DEFAULT_FALLBACK_INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "00_evaluation",
    "evaluations",
    "per_island_corrected",
)

DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "00_evaluation",
    "evaluations",
    "per_island_non_agent4_corrected",
)

DEFAULT_REFERENCES_DIR = os.path.join(
    PROJECT_ROOT,
    "00_evaluation",
    "references",
)

DEFAULT_CONTEXT_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_PLANS_DIR = os.path.join(
    PROJECT_ROOT,
    "logs",
    "agent2_plans",
)


AGENT4_COLUMNS = [
    # Writing
    "writing_status",
    "fluency_score",
    "structure_score",
    "organization_score",
    "writing_score",
    "writing_rationale",
    "writing_error",

    # Concept evaluation
    "concept_status",
    "concept_coverage_score",
    "concept_accuracy_score",
    "concept_relevance_score",
    "concept_organization_score",
    "concept_score",
    "missing_key_concepts",
    "inaccurate_or_unsupported_concepts",
    "concept_rationale",
    "concept_error",
]


RECOMPUTED_COLUMNS = [
    # Reference / lexical informativeness
    "reference_file",
    "informativeness_status",
    "rouge_l",
    "meteor",
    "informativeness_error",

    # Context and plan
    "context_file",
    "plan_file",

    # Artifact usage
    "uses_retrieval_context",
    "uses_agent2_plan",
    "context_artifact_role",
    "agent2_plan_artifact_role",

    # Verifiability
    "verifiability_status",
    "citation_recall",
    "citation_precision",
    "citation_link_precision",
    "citation_rate",
    "num_sentences",
    "num_cited_sentences",
    "num_citation_links",
    "num_supported_citation_links",
    "verifiability_error",

    # CSCS
    "cscs_status",
    "cscs",
    "cscs_edges",
    "cscs_checked_edges",
    "cscs_checked_facts",
    "cscs_error",

    # Diagnostics
    "diagnostics_error",

    "generated_section_count",
    "generated_empty_section_count",
    "generated_error_section_count",
    "generated_sentence_count",

    "citation_marker_count",
    "valid_citation_marker_count",
    "invalid_citation_marker_count",
    "sections_with_invalid_citations",
    "citation_index_validity",

    "context_status",
    "context_section_count",
    "context_total_chunks",
    "context_empty_chunk_section_count",
    "context_empty_chunk_sections",
    "context_chunk_coverage",
    "context_avg_chunks_per_section",

    "plan_status",
    "plan_node_count",
    "plan_edge_count",
    "plan_graph_density",
    "plan_avg_dependencies",
    "plan_max_dependencies",
    "plan_source_node_count",
    "plan_source_nodes",
    "plan_missing_summary_count",
    "plan_missing_summaries",
    "plan_order_violation_count",

    "plan_output_alignment_status",
    "planned_section_coverage",
    "matched_planned_sections",
    "missing_planned_sections",
    "extra_generated_sections",
]


def build_nli_from_config(config: dict, nli_model, nli_threshold: float):
    model_name = nli_model

    if not model_name:
        model_name = (
            config.get("llm", {})
            .get("agent2", {})
            .get("graph_logic", {})
            .get("nli_model", {})
            .get("model_name", "cross-encoder/nli-deberta-v3-base")
        )

    entailment_index = (
        config.get("llm", {})
        .get("agent2", {})
        .get("graph_logic", {})
        .get("nli_model", {})
        .get("entailment_index", 1)
    )

    return NLIEvaluator(
        model_name=model_name,
        entailment_index=entailment_index,
        threshold=nli_threshold,
    )


def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    return df


def set_row_values(df: pd.DataFrame, idx, values: dict):
    for key, value in values.items():
        if key not in df.columns:
            df[key] = ""

        df.at[idx, key] = value


def load_result_items_by_method(source_file: str) -> dict:
    """
    Load result_all_*.json and return:
      lowercased method name -> normalized item
    """
    items = normalize_result_file(source_file)

    by_method = {}

    for item in items:
        method = str(item.get("method", "")).strip().lower()

        if method:
            by_method[method] = item

    return by_method


def recompute_csv(
    input_csv: str,
    output_csv: str,
    references_dir: str,
    context_dir: str,
    plans_dir: str,
    nli: NLIEvaluator,
    cscs_max_facts: int,
):
    # dtype=str and keep_default_na=False preserve blanks instead of turning them into NaN.
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    df = ensure_columns(df, AGENT4_COLUMNS)
    df = ensure_columns(df, RECOMPUTED_COLUMNS)

    source_cache = {}
    url_text_cache = {}
    plan_cache = {}

    for idx, row in df.iterrows():
        method = str(row.get("method", "")).strip()
        method_key = method.lower()

        island = str(row.get("island_name", "")).strip()
        source_file = str(row.get("source_file", "")).strip()
        input_json = str(row.get("input_json", "")).strip()

        # This timestamp marks when the non-Agent-4 metrics were refreshed.
        if "timestamp" in df.columns:
            df.at[idx, "timestamp"] = datetime.now().isoformat(timespec="seconds")

        if not source_file or not os.path.exists(source_file):
            set_row_values(
                df,
                idx,
                {
                    "diagnostics_error": f"Missing source_file: {source_file}",
                    "informativeness_status": "failed",
                    "informativeness_error": f"Missing source_file: {source_file}",
                    "verifiability_status": "failed",
                    "verifiability_error": f"Missing source_file: {source_file}",
                    "cscs_status": "failed",
                    "cscs_error": f"Missing source_file: {source_file}",
                },
            )
            continue

        try:
            if source_file not in source_cache:
                source_cache[source_file] = load_result_items_by_method(source_file)

            item = source_cache[source_file].get(method_key)

            if not item:
                error = f"Could not find method {method} in {source_file}"
                set_row_values(
                    df,
                    idx,
                    {
                        "diagnostics_error": error,
                        "informativeness_status": "failed",
                        "informativeness_error": error,
                        "verifiability_status": "failed",
                        "verifiability_error": error,
                        "cscs_status": "failed",
                        "cscs_error": error,
                    },
                )
                continue

            article = item.get("generated_article", "")
            sections = item.get("sections", [])

            # ---------------------------------------------------------
            # 1. Reference loading + lexical informativeness
            # ---------------------------------------------------------
            try:
                reference, reference_file = load_reference_article(
                    island_name=island,
                    references_dir=references_dir,
                )

                set_row_values(df, idx, {"reference_file": reference_file})

                info = compute_informativeness(article, reference)
                set_row_values(df, idx, info)

            except Exception as e:
                set_row_values(
                    df,
                    idx,
                    {
                        "informativeness_status": "failed",
                        "informativeness_error": str(e),
                    },
                )

            # ---------------------------------------------------------
            # 2. Context path
            # ---------------------------------------------------------
            context_path = str(row.get("context_file", "")).strip()

            if not context_path or not os.path.exists(context_path):
                context_path = infer_context_path(
                    island_name=island,
                    input_json=input_json,
                    context_dir=context_dir,
                )

            set_row_values(df, idx, {"context_file": context_path})

            # ---------------------------------------------------------
            # 3. Agent 2 plan
            # ---------------------------------------------------------
            if island not in plan_cache:
                plan_cache[island] = load_agent2_plan(island, plans_dir)

            plan, plan_file = plan_cache[island]
            set_row_values(df, idx, {"plan_file": plan_file})

            # ---------------------------------------------------------
            # 4. Artifact / pipeline diagnostics
            # ---------------------------------------------------------
            try:
                diagnostics = compute_artifact_diagnostics(
                    sections=sections,
                    context_path=context_path,
                    plan=plan,
                    method_name=method,
                )

                set_row_values(df, idx, diagnostics)
                set_row_values(df, idx, {"diagnostics_error": ""})

            except Exception as e:
                set_row_values(df, idx, {"diagnostics_error": str(e)})

            # ---------------------------------------------------------
            # 5. Verifiability
            # ---------------------------------------------------------
            try:
                if context_path not in url_text_cache:
                    url_text_cache[context_path] = load_url_text_map(context_path)

                url_to_texts = url_text_cache[context_path]

                verif = compute_verifiability(
                    sections=sections,
                    article=article,
                    url_to_texts=url_to_texts,
                    nli=nli,
                )

                set_row_values(df, idx, verif)

            except Exception as e:
                set_row_values(
                    df,
                    idx,
                    {
                        "verifiability_status": "failed",
                        "verifiability_error": str(e),
                    },
                )

            # ---------------------------------------------------------
            # 6. CSCS
            # ---------------------------------------------------------
            if method_key != "method3":
                set_row_values(
                    df,
                    idx,
                    {
                        "cscs_status": "not_applicable",
                        "cscs": "",
                        "cscs_edges": "",
                        "cscs_checked_edges": "",
                        "cscs_checked_facts": "",
                        "cscs_error": (
                            "CSCS is only applicable to method3 because only method3 "
                            "uses the Agent 2 dependency graph during generation."
                        ),
                    },
                )

            else:
                try:
                    cscs = compute_cscs(
                        sections=sections,
                        article=article,
                        plan=plan,
                        nli=nli,
                        max_facts_per_edge=cscs_max_facts,
                    )

                    set_row_values(df, idx, cscs)

                except Exception as e:
                    set_row_values(
                        df,
                        idx,
                        {
                            "cscs_status": "failed",
                            "cscs": "",
                            "cscs_edges": "",
                            "cscs_checked_edges": "",
                            "cscs_checked_facts": "",
                            "cscs_error": str(e),
                        },
                    )

        except Exception as e:
            set_row_values(
                df,
                idx,
                {
                    "diagnostics_error": str(e),
                    "informativeness_status": "failed",
                    "informativeness_error": str(e),
                    "verifiability_status": "failed",
                    "verifiability_error": str(e),
                    "cscs_status": "failed",
                    "cscs_error": str(e),
                },
            )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute all non-Agent-4 metrics while preserving writing and concept scores."
        )
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory containing existing per-island *_eval.csv files with "
            "Agent 4 writing/concept scores. Defaults to "
            "per_island_verifiability_claimsplit if it exists, else per_island_corrected."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where corrected CSVs are written.",
    )

    parser.add_argument(
        "--references-dir",
        default=DEFAULT_REFERENCES_DIR,
        help="Directory containing human reference articles.",
    )

    parser.add_argument(
        "--context-dir",
        default=DEFAULT_CONTEXT_DIR,
        help="Directory containing *_rag_context.json files.",
    )

    parser.add_argument(
        "--plans-dir",
        default=DEFAULT_PLANS_DIR,
        help="Directory containing Agent 2 plan JSON files.",
    )

    parser.add_argument(
        "--nli-model",
        default=None,
        help="Override NLI model.",
    )

    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="NLI entailment threshold. Default: 0.5.",
    )

    parser.add_argument(
        "--cscs-max-facts",
        type=int,
        default=5,
        help="Max upstream facts checked per dependency edge.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir

    if input_dir is None:
        if os.path.isdir(DEFAULT_PRIMARY_INPUT_DIR):
            input_dir = DEFAULT_PRIMARY_INPUT_DIR
        else:
            input_dir = DEFAULT_FALLBACK_INPUT_DIR

    files = sorted(glob.glob(os.path.join(input_dir, "*_eval.csv")))

    if not files:
        raise FileNotFoundError(f"No *_eval.csv files found in: {input_dir}")

    print("\n" + "=" * 80)
    print("Recomputing non-Agent-4 metrics")
    print("=" * 80)
    print(f"Input dir:       {input_dir}")
    print(f"Output dir:      {args.output_dir}")
    print(f"CSV files:       {len(files)}")
    print("Agent 4 writing: preserved")
    print("Agent 4 concept: preserved")
    print("ROUGE/METEOR:    recomputed")
    print("Verifiability:   recomputed")
    print("CSCS:            recomputed for method3 only")
    print("Diagnostics:     recomputed")
    print("=" * 80 + "\n")

    config = load_config()
    nli = build_nli_from_config(
        config=config,
        nli_model=args.nli_model,
        nli_threshold=args.nli_threshold,
    )

    for input_csv in files:
        output_csv = os.path.join(args.output_dir, os.path.basename(input_csv))

        print(f"Updating: {os.path.basename(input_csv)}")

        recompute_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            references_dir=args.references_dir,
            context_dir=args.context_dir,
            plans_dir=args.plans_dir,
            nli=nli,
            cscs_max_facts=args.cscs_max_facts,
        )

        print(f"Saved:    {output_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()



# python 00_evaluation/recompute_non_agent4_metrics.py --input-dir "00_evaluation/evaluations/per_island_verifiability_claimsplit" --output-dir "00_evaluation/evaluations/per_island_non_agent4_corrected"



# python 00_evaluation/aggregate_evaluations.py --input-dir "00_evaluation/evaluations/per_island_non_agent4_corrected" --output-dir "00_evaluation/evaluations/aggregated_non_agent4_corrected"
