"""
Cross-Sectional Consistency Score, CSCS.

This version evaluates whether Agent 2 dependency links are reflected
as semantic relationships between generated sections.

It does NOT require downstream sections to repeat upstream facts.

For each dependency edge:
    child_section -> parent_section

It compares:
    parent section summary/text
    child generated section text

CSCS = average semantic relatedness score across dependency edges.

Score range:
    0.0 = unrelated
    1.0 = strongly related / coherent
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from eval_utils import safe_name


def load_agent2_plan(island_name: str, plans_dir: str) -> Tuple[Optional[dict], str]:
    """
    Load Agent 2 dependency plan for an island.
    Tries exact and loose filename matching.
    """
    candidates = [
        f"{safe_name(island_name)}_plan.json",
        f"{island_name.replace(' ', '_')}_plan.json",
        f"{island_name}_plan.json",
    ]

    for fname in candidates:
        path = os.path.join(plans_dir, fname)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path

    # Loose fallback for names with special punctuation.
    normalized_target = safe_name(island_name).lower()

    if os.path.exists(plans_dir):
        for fname in os.listdir(plans_dir):
            if not fname.endswith("_plan.json"):
                continue

            normalized_fname = safe_name(fname).lower()

            if normalized_target in normalized_fname or normalized_fname in normalized_target:
                path = os.path.join(plans_dir, fname)

                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f), path

    return None, ""


def get_section_text_map(sections: List[dict], article: str) -> Dict[str, str]:
    """
    Build section_name -> generated section text.
    """
    section_map = {}

    for sec in sections or []:
        name = sec.get("section_name", "")
        content = sec.get("content", "")

        if name:
            section_map[name.strip().lower()] = content

    return section_map


class SemanticSimilarityScorer:
    """
    Embedding-based semantic similarity scorer.

    Uses sentence-transformers.
    Default model is small and fast.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.np = np

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0

        emb = self.model.encode(
            [text_a[:3000], text_b[:3000]],
            normalize_embeddings=True,
        )

        score = float(self.np.dot(emb[0], emb[1]))

        # cosine can theoretically be negative; clamp to 0–1
        if score < 0:
            score = 0.0
        if score > 1:
            score = 1.0

        return round(score, 4)


def compute_cscs(
    sections: List[dict],
    article: str,
    plan: Optional[dict],
    nli=None,
    max_facts_per_edge: int = 5,
) -> dict:
    """
    Compute CSCS as average semantic relatedness over Agent 2 dependency edges.

    Parameters keep `nli` and `max_facts_per_edge` for compatibility with
    full_evaluation.py, but this CSCS does not use NLI entailment.
    """
    if not plan:
        return {
            "cscs_status": "missing_plan",
            "cscs": "",
            "cscs_edges": 0,
            "cscs_checked_facts": 0,
            "cscs_error": "No Agent 2 plan found.",
        }

    dependency = plan.get("dependency", {}) or {}
    summaries = plan.get("summaries", {}) or {}

    section_map = get_section_text_map(sections, article)

    scorer = SemanticSimilarityScorer()

    edge_scores = []
    checked_edges = 0

    for child_section, parent_sections in dependency.items():
        child_text = section_map.get(child_section.strip().lower(), "")

        if not child_text:
            continue

        for parent_section in parent_sections:
            checked_edges += 1

            # Prefer Agent 2's summary for parent section.
            parent_text = summaries.get(parent_section, "")

            # Fallback: use generated parent section content.
            if not parent_text:
                parent_text = section_map.get(parent_section.strip().lower(), "")

            if not parent_text:
                continue

            score = scorer.similarity(parent_text, child_text)
            edge_scores.append(score)

    if not edge_scores:
        return {
            "cscs_status": "no_edges_checked",
            "cscs": "",
            "cscs_edges": checked_edges,
            "cscs_checked_facts": 0,
            "cscs_error": "No dependency edges could be scored.",
        }

    return {
        "cscs_status": "success",
        "cscs": round(sum(edge_scores) / len(edge_scores), 4),
        "cscs_edges": len(edge_scores),
        "cscs_checked_facts": 0,
        "cscs_error": "",
    }