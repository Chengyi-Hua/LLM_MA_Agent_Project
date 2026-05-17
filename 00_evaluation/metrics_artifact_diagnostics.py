"""
00_evaluation/metrics_artifact_diagnostics.py

Artifact / Pipeline Diagnostics.

These metrics consume existing files only:
  - generated output JSON
  - RAG context JSON
  - Agent 2 plan JSON

They help explain evaluation results but are not WikiGenBench metrics.

Important distinction:
  - Some artifacts are available for every island.
  - Not every method actually used every artifact during generation.

Method usage:
  method0 = pure generation
  method1 = naive RAG
  method2 = hierarchical RAG
  method3 = Agent 2 plan + Agent 3 context-aware generation

Therefore:
  - method0 does not use retrieval context or Agent 2 plan.
  - method1 uses retrieval context but not Agent 2 plan.
  - method2 uses retrieval context but not Agent 2 plan.
  - method3 uses both retrieval context and Agent 2 plan.

The diagnostics report both:
  1. artifact properties, e.g. plan_edge_count
  2. artifact usage flags, e.g. uses_agent2_plan
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from eval_utils import extract_citation_numbers, split_sentences


ERROR_PATTERNS = [
    "i'm sorry",
    "i’m sorry",
    "i do not see",
    "i don't see",
    "please provide",
    "cannot generate",
    "no documents included",
]


METHODS_USING_RETRIEVAL_CONTEXT = {"method1", "method2", "method3"}
METHODS_USING_AGENT2_PLAN = {"method3"}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def normalize_heading(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def heading_match(a: str, b: str, threshold: float = 0.65) -> bool:
    a_norm = normalize_heading(a)
    b_norm = normalize_heading(b)

    if not a_norm or not b_norm:
        return False

    if a_norm == b_norm:
        return True

    if a_norm in b_norm or b_norm in a_norm:
        return True

    return SequenceMatcher(None, a_norm, b_norm).ratio() >= threshold


def load_json_if_exists(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bool_to_string(value: bool) -> str:
    return "true" if value else "false"


# ---------------------------------------------------------------------
# Artifact usage diagnostics
# ---------------------------------------------------------------------

def compute_artifact_usage(
    method_name: str,
    context_data: Optional[dict],
    plan: Optional[dict],
) -> dict:
    """
    Reports whether a method actually used the available artifacts.

    This avoids misleading rows such as:
      method0 | plan_edge_count=6

    The plan may exist for the island, but method0 did not use it.
    """
    method_name = str(method_name or "").strip().lower()

    uses_retrieval_context = method_name in METHODS_USING_RETRIEVAL_CONTEXT
    uses_agent2_plan = method_name in METHODS_USING_AGENT2_PLAN

    if not context_data:
        context_role = "missing_context"
    elif uses_retrieval_context:
        context_role = "used_by_generation"
    else:
        context_role = "available_but_not_used_by_generation"

    if not plan:
        plan_role = "missing_plan"
    elif uses_agent2_plan:
        plan_role = "used_by_generation"
    else:
        plan_role = "available_for_alignment_only"

    return {
        "uses_retrieval_context": bool_to_string(uses_retrieval_context),
        "uses_agent2_plan": bool_to_string(uses_agent2_plan),
        "context_artifact_role": context_role,
        "agent2_plan_artifact_role": plan_role,
    }


# ---------------------------------------------------------------------
# Generated output diagnostics
# ---------------------------------------------------------------------

def compute_generated_output_diagnostics(sections: List[dict]) -> dict:
    """
    Diagnostics over the generated output itself.

    Applies to all methods.
    """
    section_count = len(sections or [])
    empty_sections = 0
    error_sections = 0

    total_sentences = 0
    citation_marker_count = 0
    valid_citation_marker_count = 0
    invalid_citation_marker_count = 0
    sections_with_invalid_citations = 0

    for sec in sections or []:
        content = sec.get("content", "") or ""
        citations = sec.get("citations", []) or []

        if not content.strip():
            empty_sections += 1

        lower_content = content.lower()
        if any(pattern in lower_content for pattern in ERROR_PATTERNS):
            error_sections += 1

        section_has_invalid = False

        for sent in split_sentences(content):
            total_sentences += 1
            nums = extract_citation_numbers(sent)

            for n in nums:
                citation_marker_count += 1

                if 1 <= n <= len(citations):
                    valid_citation_marker_count += 1
                else:
                    invalid_citation_marker_count += 1
                    section_has_invalid = True

        if section_has_invalid:
            sections_with_invalid_citations += 1

    citation_index_validity = (
        round(valid_citation_marker_count / citation_marker_count, 4)
        if citation_marker_count > 0
        else ""
    )

    return {
        "generated_section_count": section_count,
        "generated_empty_section_count": empty_sections,
        "generated_error_section_count": error_sections,
        "generated_sentence_count": total_sentences,
        "citation_marker_count": citation_marker_count,
        "valid_citation_marker_count": valid_citation_marker_count,
        "invalid_citation_marker_count": invalid_citation_marker_count,
        "sections_with_invalid_citations": sections_with_invalid_citations,
        "citation_index_validity": citation_index_validity,
    }


# ---------------------------------------------------------------------
# RAG context diagnostics
# ---------------------------------------------------------------------

def compute_context_diagnostics(context_data: Optional[dict]) -> dict:
    """
    Diagnostics over the cached RAG context file.

    This describes the context artifact itself.
    Whether the method used it is reported separately by compute_artifact_usage().
    """
    if not context_data:
        return {
            "context_status": "missing_context",
            "context_section_count": "",
            "context_total_chunks": "",
            "context_empty_chunk_section_count": "",
            "context_empty_chunk_sections": "",
            "context_chunk_coverage": "",
            "context_avg_chunks_per_section": "",
        }

    sections_data = (
        context_data.get("blueprint_data", {})
        .get("sections_data", {})
    )

    section_count = len(sections_data)
    total_chunks = 0
    empty_sections = []

    for section_name, section_obj in sections_data.items():
        chunks = section_obj.get("chunks", []) or []
        total_chunks += len(chunks)

        if len(chunks) == 0:
            empty_sections.append(section_name)

    nonempty_count = section_count - len(empty_sections)

    context_chunk_coverage = (
        round(nonempty_count / section_count, 4)
        if section_count
        else ""
    )

    context_avg_chunks_per_section = (
        round(total_chunks / section_count, 4)
        if section_count
        else ""
    )

    return {
        "context_status": "success",
        "context_section_count": section_count,
        "context_total_chunks": total_chunks,
        "context_empty_chunk_section_count": len(empty_sections),
        "context_empty_chunk_sections": "; ".join(empty_sections),
        "context_chunk_coverage": context_chunk_coverage,
        "context_avg_chunks_per_section": context_avg_chunks_per_section,
    }


# ---------------------------------------------------------------------
# Agent 2 plan diagnostics
# ---------------------------------------------------------------------

def compute_plan_diagnostics(plan: Optional[dict]) -> dict:
    """
    Diagnostics over the Agent 2 dependency plan.

    This describes the plan artifact itself.
    Whether the method used it is reported separately by compute_artifact_usage().
    """
    if not plan:
        return {
            "plan_status": "missing_plan",
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
        }

    order = plan.get("order", []) or []
    dependency = plan.get("dependency", {}) or {}
    summaries = plan.get("summaries", {}) or {}

    # Build a robust node list.
    # Prefer Agent 2 order, but include any dependency children, parents,
    # or summary keys that may be missing from the order list.
    nodes = list(order)

    seen_nodes = set(nodes)

    for child, parents in dependency.items():
        if child not in seen_nodes:
            nodes.append(child)
            seen_nodes.add(child)

        for parent in parents or []:
            if parent not in seen_nodes:
                nodes.append(parent)
                seen_nodes.add(parent)

    for node in summaries.keys():
        if node not in seen_nodes:
            nodes.append(node)
            seen_nodes.add(node)

    node_count = len(nodes)

    edge_count = sum(len(v or []) for v in dependency.values())
    max_possible_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 0

    graph_density = (
        round(edge_count / max_possible_edges, 4)
        if max_possible_edges
        else 0.0
    )

    dep_counts = [
        len(dependency.get(node, []) or [])
        for node in nodes
    ]

    source_nodes = [
        node
        for node in nodes
        if len(dependency.get(node, []) or []) == 0
    ]

    missing_summaries = [
        node
        for node in nodes
        if not str(summaries.get(node, "")).strip()
    ]

    order_index = {
        name: i
        for i, name in enumerate(order)
    }

    order_violations = 0

    for child, parents in dependency.items():
        for parent in parents or []:
            if child in order_index and parent in order_index:
                if order_index[parent] > order_index[child]:
                    order_violations += 1

    return {
        "plan_status": "success",
        "plan_node_count": node_count,
        "plan_edge_count": edge_count,
        "plan_graph_density": graph_density,
        "plan_avg_dependencies": round(edge_count / node_count, 4)
        if node_count
        else "",
        "plan_max_dependencies": max(dep_counts) if dep_counts else "",
        "plan_source_node_count": len(source_nodes),
        "plan_source_nodes": "; ".join(source_nodes),
        "plan_missing_summary_count": len(missing_summaries),
        "plan_missing_summaries": "; ".join(missing_summaries),
        "plan_order_violation_count": order_violations,
    }


# ---------------------------------------------------------------------
# Output-vs-plan alignment diagnostics
# ---------------------------------------------------------------------

def compute_plan_output_alignment(sections: List[dict], plan: Optional[dict]) -> dict:
    """
    Compares generated section names with Agent 2 planned section names.

    This can be useful for every method as an external alignment diagnostic,
    even though only method3 used the Agent 2 plan during generation.
    """
    if not plan:
        return {
            "plan_output_alignment_status": "missing_plan",
            "planned_section_coverage": "",
            "matched_planned_sections": "",
            "missing_planned_sections": "",
            "extra_generated_sections": "",
        }

    planned = plan.get("order", []) or []
    generated = [
        sec.get("section_name", "")
        for sec in sections or []
    ]

    matched = []
    missing = []

    for planned_section in planned:
        if any(heading_match(planned_section, gen_section) for gen_section in generated):
            matched.append(planned_section)
        else:
            missing.append(planned_section)

    extra = []

    for gen_section in generated:
        if not any(heading_match(gen_section, planned_section) for planned_section in planned):
            extra.append(gen_section)

    coverage = (
        round(len(matched) / len(planned), 4)
        if planned
        else ""
    )

    return {
        "plan_output_alignment_status": "success",
        "planned_section_coverage": coverage,
        "matched_planned_sections": "; ".join(matched),
        "missing_planned_sections": "; ".join(missing),
        "extra_generated_sections": "; ".join(extra),
    }


# ---------------------------------------------------------------------
# Combined diagnostics
# ---------------------------------------------------------------------

def compute_artifact_diagnostics(
    sections: List[dict],
    context_path: str,
    plan: Optional[dict],
    method_name: str = "",
) -> dict:
    """
    Combined artifact diagnostics.

    Parameters
    ----------
    sections:
        Generated output sections from one method result.

    context_path:
        Path to the cached *_rag_context.json file.

    plan:
        Loaded Agent 2 plan dictionary.

    method_name:
        method0, method1, method2, or method3.
        Used only to report whether artifacts were actually used by generation.
    """
    context_data = load_json_if_exists(context_path)

    usage_diag = compute_artifact_usage(
        method_name=method_name,
        context_data=context_data,
        plan=plan,
    )

    output_diag = compute_generated_output_diagnostics(sections)
    context_diag = compute_context_diagnostics(context_data)
    plan_diag = compute_plan_diagnostics(plan)
    alignment_diag = compute_plan_output_alignment(sections, plan)

    return {
        **usage_diag,
        **output_diag,
        **context_diag,
        **plan_diag,
        **alignment_diag,
    }