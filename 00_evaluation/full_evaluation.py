"""
00_evaluation/full_evaluation.py

Full evaluation:
  1. Writing via Agent 4
  2. Concept-based LLM evaluation via Agent 4
  3. ROUGE-L
  4. METEOR
  5. Citation recall / precision / rate
  6. CSCS
  7. Artifact / pipeline diagnostics

Run from project root:

    python 00_evaluation/full_evaluation.py ^
      --input data/outputs/result_all_Surtsey_20260503_210618.json

Quick test without expensive LLM/NLI parts:

    python 00_evaluation/full_evaluation.py ^
      --input data/outputs/result_all_Surtsey_20260503_210618.json ^
      --skip-writing ^
      --skip-concept ^
      --skip-nli ^
      --skip-cscs
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

from metrics_verifiability import (
    NLIEvaluator,
    compute_verifiability,
    infer_context_path,
    load_url_text_map,
)

from metrics_cscs import compute_cscs, load_agent2_plan

from metrics_artifact_diagnostics import compute_artifact_diagnostics


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "evaluations")
DEFAULT_REFERENCES_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "references")
DEFAULT_CONTEXT_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_PLANS_DIR = os.path.join(PROJECT_ROOT, "logs", "agent2_plans")


def write_csv(output_path: str, rows: list):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp",
        "source_file",
        "input_json",
        "reference_file",
        "context_file",
        "plan_file",
        "island_name",
        "method",

        # Artifact usage flags
        "uses_retrieval_context",
        "uses_agent2_plan",
        "context_artifact_role",
        "agent2_plan_artifact_role",

        # Writing
        "writing_status",
        "fluency_score",
        "structure_score",
        "organization_score",
        "writing_score",
        "writing_rationale",
        "writing_error",

        # Informativeness: lexical
        "informativeness_status",
        "rouge_l",
        "meteor",
        "informativeness_error",

        # Informativeness: concept-based LLM judge
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

        # Verifiability
        "verifiability_status",
        "citation_recall",
        "citation_precision",
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
        "cscs_checked_facts",
        "cscs_error",

        # Artifact / pipeline diagnostics
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

    exists = os.path.exists(output_path)

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


def build_nli_from_config(config: dict, args) -> NLIEvaluator:
    model_name = args.nli_model

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
        threshold=args.nli_threshold,
    )


def make_empty_row(item: dict, island: str, method: str) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_file": item.get("source_file", ""),
        "input_json": item.get("input_json", ""),
        "reference_file": "",
        "context_file": "",
        "plan_file": "",
        "island_name": island,
        "method": method,

        # Artifact usage flags
        "uses_retrieval_context": "",
        "uses_agent2_plan": "",
        "context_artifact_role": "",
        "agent2_plan_artifact_role": "",

        # Writing
        "writing_status": "",
        "fluency_score": "",
        "structure_score": "",
        "organization_score": "",
        "writing_score": "",
        "writing_rationale": "",
        "writing_error": "",

        # Informativeness: lexical
        "informativeness_status": "",
        "rouge_l": "",
        "meteor": "",
        "informativeness_error": "",

        # Informativeness: concept-based
        "concept_status": "",
        "concept_coverage_score": "",
        "concept_accuracy_score": "",
        "concept_relevance_score": "",
        "concept_organization_score": "",
        "concept_score": "",
        "missing_key_concepts": "",
        "inaccurate_or_unsupported_concepts": "",
        "concept_rationale": "",
        "concept_error": "",

        # Verifiability
        "verifiability_status": "",
        "citation_recall": "",
        "citation_precision": "",
        "citation_rate": "",
        "num_sentences": "",
        "num_cited_sentences": "",
        "num_citation_links": "",
        "num_supported_citation_links": "",
        "verifiability_error": "",

        # CSCS
        "cscs_status": "",
        "cscs": "",
        "cscs_edges": "",
        "cscs_checked_facts": "",
        "cscs_error": "",

        # Diagnostics
        "diagnostics_error": "",

        "generated_section_count": "",
        "generated_empty_section_count": "",
        "generated_error_section_count": "",
        "generated_sentence_count": "",

        "citation_marker_count": "",
        "valid_citation_marker_count": "",
        "invalid_citation_marker_count": "",
        "sections_with_invalid_citations": "",
        "citation_index_validity": "",

        "context_status": "",
        "context_section_count": "",
        "context_total_chunks": "",
        "context_empty_chunk_section_count": "",
        "context_empty_chunk_sections": "",
        "context_chunk_coverage": "",
        "context_avg_chunks_per_section": "",

        "plan_status": "",
        "plan_node_count": "",
        "plan_edge_count": "",
        "plan_graph_density": "",
        "plan_avg_dependencies": "",
        "plan_max_dependencies": "",
        "plan_source_node_count": "",
        "plan_source_nodes": "",
        "plan_missing_summary_count": "",
        "plan_missing_summaries": "",
        "plan_order_violation_count": "",

        "plan_output_alignment_status": "",
        "planned_section_coverage": "",
        "matched_planned_sections": "",
        "missing_planned_sections": "",
        "extra_generated_sections": "",
    }


def evaluate(args):
    input_files = discover_input_files(args.input, args.input_dir)

    if not input_files:
        raise FileNotFoundError("No input files found. Use --input or --input-dir.")

    output_path = args.output or os.path.join(
        DEFAULT_OUTPUT_DIR,
        f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    config = load_config()

    agent4_evaluator = None
    if not args.skip_writing or not args.skip_concept:
        agent4_evaluator = Agent4Evaluator(config=config)

    nli = None
    if not args.skip_nli:
        nli = build_nli_from_config(config, args)

    print("\n" + "=" * 80)
    print("Full Evaluation")
    print("=" * 80)
    print(f"Input files:   {len(input_files)}")
    print(f"Output CSV:    {output_path}")
    print(f"Writing:       {'skipped' if args.skip_writing else 'enabled'}")
    print(f"Concept judge: {'skipped' if args.skip_concept else 'enabled'}")
    print(f"NLI:           {'enabled' if nli else 'skipped'}")
    print(f"CSCS:          {'skipped' if args.skip_cscs else 'enabled'}")
    print("Diagnostics:   enabled")
    print("=" * 80 + "\n")

    for input_file in input_files:
        print(f"\nReading: {input_file}")

        try:
            items = normalize_result_file(input_file)
        except Exception as e:
            print(f"Could not read {input_file}: {e}")
            continue

        for item in items:
            island = item["island_name"]
            method = item["method"]
            article = item["generated_article"]
            sections = item["sections"]

            print(f"\nEvaluating: {island} | {method}")

            row = make_empty_row(item, island, method)

            # ---------------------------------------------------------
            # 1. Writing evaluation
            # ---------------------------------------------------------
            if agent4_evaluator and not args.skip_writing:
                try:
                    writing = agent4_evaluator.evaluate_article(
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

            else:
                row["writing_status"] = "skipped"
                row["writing_error"] = "Writing evaluation skipped."

            # ---------------------------------------------------------
            # 2. Reference loading + lexical informativeness
            # ---------------------------------------------------------
            reference = None

            try:
                reference, reference_file = load_reference_article(
                    island,
                    args.references_dir,
                )

                row["reference_file"] = reference_file

                info = compute_informativeness(article, reference)
                row.update(info)

            except Exception as e:
                row["informativeness_status"] = "failed"
                row["informativeness_error"] = str(e)

            # ---------------------------------------------------------
            # 3. Concept-based LLM evaluation
            # ---------------------------------------------------------
            if args.skip_concept:
                row["concept_status"] = "skipped"
                row["concept_error"] = "Concept evaluation skipped."

            elif not reference:
                row["concept_status"] = "missing_reference"
                row["concept_error"] = "No reference article found."

            elif agent4_evaluator:
                try:
                    concept_eval = agent4_evaluator.evaluate_concepts(
                        generated_article=article,
                        reference_article=reference,
                        island_name=island,
                        method_name=method,
                    )

                    concept_scores = concept_eval.get("scores", {})

                    row["concept_status"] = concept_eval.get("status", "")
                    row["concept_coverage_score"] = concept_scores.get(
                        "concept_coverage_score", ""
                    )
                    row["concept_accuracy_score"] = concept_scores.get(
                        "concept_accuracy_score", ""
                    )
                    row["concept_relevance_score"] = concept_scores.get(
                        "concept_relevance_score", ""
                    )
                    row["concept_organization_score"] = concept_scores.get(
                        "concept_organization_score", ""
                    )
                    row["concept_score"] = concept_scores.get("concept_score", "")

                    row["missing_key_concepts"] = "; ".join(
                        concept_scores.get("missing_key_concepts", []) or []
                    )
                    row["inaccurate_or_unsupported_concepts"] = "; ".join(
                        concept_scores.get("inaccurate_or_unsupported_concepts", []) or []
                    )
                    row["concept_rationale"] = concept_scores.get("brief_rationale", "")

                except Exception as e:
                    row["concept_status"] = "failed"
                    row["concept_error"] = str(e)

            else:
                row["concept_status"] = "failed"
                row["concept_error"] = "Agent 4 evaluator was not initialized."

            # ---------------------------------------------------------
            # 4. Context path
            # ---------------------------------------------------------
            context_path = infer_context_path(
                island_name=island,
                input_json=item.get("input_json", ""),
                context_dir=args.context_dir,
            )

            row["context_file"] = context_path

            # ---------------------------------------------------------
            # 5. Load Agent 2 plan
            # ---------------------------------------------------------
            plan, plan_file = load_agent2_plan(island, args.plans_dir)
            row["plan_file"] = plan_file

            # ---------------------------------------------------------
            # 6. Artifact / pipeline diagnostics
            # ---------------------------------------------------------
            try:
                diagnostics = compute_artifact_diagnostics(
                    sections=sections,
                    context_path=context_path,
                    plan=plan,
                    method_name=method,
                )

                row.update(diagnostics)

            except Exception as e:
                row["diagnostics_error"] = str(e)

            # ---------------------------------------------------------
            # 7. Verifiability
            # ---------------------------------------------------------
            if args.skip_nli:
                row["verifiability_status"] = "skipped"
                row["verifiability_error"] = "NLI skipped."

            else:
                try:
                    url_to_texts = load_url_text_map(context_path)

                    verif = compute_verifiability(
                        sections=sections,
                        article=article,
                        url_to_texts=url_to_texts,
                        nli=nli,
                    )

                    row.update(verif)

                except Exception as e:
                    row["verifiability_status"] = "failed"
                    row["verifiability_error"] = str(e)

            # ---------------------------------------------------------
            # 8. CSCS
            # ---------------------------------------------------------
            method_key = str(method).strip().lower()

            if args.skip_cscs:
                row["cscs_status"] = "skipped"
                row["cscs"] = ""
                row["cscs_edges"] = ""
                row["cscs_checked_facts"] = ""
                row["cscs_error"] = "CSCS skipped."

            elif method_key != "method3":
                row["cscs_status"] = "not_applicable"
                row["cscs"] = ""
                row["cscs_edges"] = ""
                row["cscs_checked_facts"] = ""
                row["cscs_error"] = (
                    "CSCS is only applicable to method3 because only method3 "
                    "uses the Agent 2 dependency graph during generation."
                )

            elif nli is None:
                row["cscs_status"] = "missing_nli"
                row["cscs"] = ""
                row["cscs_edges"] = ""
                row["cscs_checked_facts"] = ""
                row["cscs_error"] = (
                    "Proposal-aligned CSCS requires NLI. Do not use --skip-nli "
                    "if you want CSCS for method3."
                )

            else:
                try:
                    cscs = compute_cscs(
                        sections=sections,
                        article=article,
                        plan=plan,
                        nli=nli,
                        max_facts_per_edge=args.cscs_max_facts,
                    )

                    row.update(cscs)

                except Exception as e:
                    row["cscs_status"] = "failed"
                    row["cscs"] = ""
                    row["cscs_edges"] = ""
                    row["cscs_checked_facts"] = ""
                    row["cscs_error"] = str(e)

            # ---------------------------------------------------------
            # 9. Save row
            # ---------------------------------------------------------
            write_csv(output_path, [row])

            print(
                f"{island} | {method} | "
                f"uses_context={row['uses_retrieval_context']} | "
                f"uses_agent2_plan={row['uses_agent2_plan']} | "
                f"context_role={row['context_artifact_role']} | "
                f"plan_role={row['agent2_plan_artifact_role']} | "
                f"fluency={row['fluency_score']} | "
                f"structure={row['structure_score']} | "
                f"organization={row['organization_score']} | "
                f"writing_avg={row['writing_score']} | "
                f"concept_coverage={row['concept_coverage_score']} | "
                f"concept_accuracy={row['concept_accuracy_score']} | "
                f"concept_relevance={row['concept_relevance_score']} | "
                f"concept_organization={row['concept_organization_score']} | "
                f"concept_avg={row['concept_score']} | "
                f"ROUGE-L={row['rouge_l']} | "
                f"METEOR={row['meteor']} | "
                f"citation_rate={row['citation_rate']} | "
                f"citation_validity={row['citation_index_validity']} | "
                f"error_sections={row['generated_error_section_count']} | "
                f"context_coverage={row['context_chunk_coverage']} | "
                f"plan_edges={row['plan_edge_count']} | "
                f"planned_coverage={row['planned_section_coverage']} | "
                f"CSCS_status={row['cscs_status']} | "
                f"CSCS={row['cscs']}"
            )

    print("\nDone.")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Full evaluator for generated Wikipedia-style articles."
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Path to one result JSON.",
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory of result JSONs.",
    )

    parser.add_argument(
        "--references-dir",
        default=DEFAULT_REFERENCES_DIR,
        help="Directory containing human Wikipedia references.",
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
        "--output",
        default=None,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--skip-writing",
        action="store_true",
        help="Skip Agent 4 writing evaluation.",
    )

    parser.add_argument(
        "--skip-concept",
        action="store_true",
        help="Skip Agent 4 concept-based reference evaluation.",
    )

    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip citation NLI verification.",
    )

    parser.add_argument(
        "--skip-cscs",
        action="store_true",
        help="Skip CSCS.",
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
    evaluate(args)


if __name__ == "__main__":
    main()