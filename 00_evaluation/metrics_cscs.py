"""
Cross-Sectional Consistency Score, CSCS.

Proposal-aligned definition:

CSCS evaluates whether dependency information from upstream sections is
preserved or appropriately reflected in downstream sections.

Important:
  - Agent 2 produces a dependency graph.
  - Only method3 uses this graph during generation.
  - Therefore, full_evaluation.py should compute CSCS only for method3.
  - For method0, method1, and method2, CSCS should be marked not_applicable.

For each dependency edge in the Agent 2 plan:

    child_section -> parent_section

the evaluator checks whether key facts from the generated parent section are
reflected in the generated child section.

CSCS = average parent-child reflection score across checked parent facts.

Scoring:
  - If NLI says the child section entails the parent fact, score = 1.0.
  - Otherwise, semantic similarity is used as a soft fallback.

Score range:
  0.0 = downstream sections do not reflect upstream dependency information
  1.0 = downstream sections strongly preserve or reflect upstream dependency information
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from eval_utils import safe_name, split_sentences, strip_citations


_CACHED_SCORER = None


# ---------------------------------------------------------------------
# Agent 2 plan loading
# ---------------------------------------------------------------------

def load_agent2_plan(island_name: str, plans_dir: str) -> Tuple[Optional[dict], str]:
    """
    Load Agent 2 plan with alias support.

    Handles cases like:
      Hashima -> Hashima Island_plan.json
      Howland -> Howland Island_plan.json
      Jarvis -> Jarvis Island_plan.json
      Tromelin -> Tromelin Island_plan.json
    """
    alias_map = {
        "Hashima": ["Hashima", "Hashima Island", "Hashima_Island"],
        "Hashima Island": ["Hashima Island", "Hashima_Island", "Hashima"],

        "Howland": ["Howland", "Howland Island", "Howland_Island"],
        "Howland Island": ["Howland Island", "Howland_Island", "Howland"],

        "Jarvis": ["Jarvis", "Jarvis Island", "Jarvis_Island"],
        "Jarvis Island": ["Jarvis Island", "Jarvis_Island", "Jarvis"],

        "Tromelin": ["Tromelin", "Tromelin Island", "Tromelin_Island"],
        "Tromelin Island": ["Tromelin Island", "Tromelin_Island", "Tromelin"],

        "Nishinoshima": [
            "Nishinoshima",
            "Nishinoshima (Ogasawara)",
            "Nishinoshima_(Ogasawara)",
        ],
        "Nishinoshima (Ogasawara)": [
            "Nishinoshima (Ogasawara)",
            "Nishinoshima_(Ogasawara)",
            "Nishinoshima",
        ],

        "Hunga Tonga–Hunga Haʻapai": [
            "Hunga Tonga–Hunga Haʻapai",
            "Hunga_Tonga–Hunga_Ha_apai",
            "Hunga_Tonga_Hunga_Ha_apai",
            "Hunga_Tonga_Hunga_Haapai",
            "Hunga Tonga Hunga Haapai",
            "Hunga Tonga Hunga Ha apai",
        ],
    }

    names = [island_name]
    names.extend(alias_map.get(island_name, []))

    candidates = []

    for name in names:
        candidates.extend(
            [
                f"{safe_name(name)}_plan.json",
                f"{str(name).replace(' ', '_')}_plan.json",
                f"{name}_plan.json",
            ]
        )

    seen = set()
    candidates = [
        c for c in candidates
        if c and not (c in seen or seen.add(c))
    ]

    for fname in candidates:
        path = os.path.join(plans_dir, fname)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path

    return None, ""


# ---------------------------------------------------------------------
# Section matching helpers
# ---------------------------------------------------------------------

def normalize_heading(text: str) -> str:
    text = str(text or "").lower().strip()
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


def get_section_text_map(sections: List[dict], article: str) -> Dict[str, str]:
    """
    Build normalized section_name -> generated section text.

    Uses the structured section list first. If no structured sections exist,
    falls back to parsing == Section == blocks from the article.
    """
    section_map = {}

    for sec in sections or []:
        name = sec.get("section_name", "") or ""
        content = sec.get("content", "") or ""

        if name:
            section_map[normalize_heading(name)] = content

    if section_map:
        return section_map

    # Fallback: parse article headings.
    parts = re.split(r"==\s*(.+?)\s*==", article or "")

    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if name:
            section_map[normalize_heading(name)] = content

    return section_map


def find_section_text(section_name: str, section_map: Dict[str, str]) -> str:
    """
    Find generated section text for a planned Agent 2 section name.

    First tries exact normalized match, then fuzzy heading match.
    """
    target = normalize_heading(section_name)

    if target in section_map:
        return section_map[target]

    for generated_name, generated_text in section_map.items():
        if heading_match(target, generated_name):
            return generated_text

    return ""


# ---------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------

def extract_parent_facts(text: str, max_facts: int = 5) -> List[str]:
    """
    Extract short fact-like units from a parent section or parent summary.

    For CSCS, these are the upstream facts that should be preserved or
    reflected in the downstream child section.
    """
    facts = []

    for sentence in split_sentences(text or ""):
        fact = strip_citations(sentence).strip()

        if not fact:
            continue

        lower_fact = fact.lower()

        # Skip Agent 2 placeholder summaries.
        if "no relevant information" in lower_fact:
            continue

        # Avoid extremely tiny fragments.
        if len(fact.split()) < 4:
            continue

        facts.append(fact)

        if len(facts) >= max_facts:
            break

    return facts


# ---------------------------------------------------------------------
# Semantic reflection scorer
# ---------------------------------------------------------------------

class SemanticSimilarityScorer:
    """
    Small embedding-based scorer used as a soft fallback when NLI entailment
    is too strict.

    This lets CSCS measure whether upstream facts are meaningfully reflected
    in downstream sections, even if they are not repeated verbatim.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        print(f"Loading CSCS semantic model: {model_name}")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.np = np

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0

        embeddings = self.model.encode(
            [str(text_a)[:1000], str(text_b)[:3000]],
            normalize_embeddings=True,
        )

        score = float(self.np.dot(embeddings[0], embeddings[1]))

        if score < 0:
            score = 0.0

        if score > 1:
            score = 1.0

        return round(score, 4)


def get_cached_scorer() -> SemanticSimilarityScorer:
    global _CACHED_SCORER

    if _CACHED_SCORER is None:
        _CACHED_SCORER = SemanticSimilarityScorer()

    return _CACHED_SCORER


def parent_fact_reflection_score(
    parent_fact: str,
    child_text: str,
    nli=None,
    scorer: Optional[SemanticSimilarityScorer] = None,
) -> float:
    """
    Score how strongly a parent fact is reflected in a child section.

    Priority:
      1. If NLI says the child entails the parent fact, score = 1.0.
      2. Otherwise use semantic similarity as a soft reflection score.
    """
    if not parent_fact or not child_text:
        return 0.0

    if nli is not None:
        try:
            if nli.entails(child_text, parent_fact):
                return 1.0
        except Exception:
            pass

    if scorer is None:
        scorer = get_cached_scorer()

    return scorer.similarity(parent_fact, child_text)


# ---------------------------------------------------------------------
# CSCS
# ---------------------------------------------------------------------

def compute_cscs(
    sections: List[dict],
    article: str,
    plan: Optional[dict],
    nli=None,
    max_facts_per_edge: int = 5,
) -> dict:
    """
    Compute proposal-aligned CSCS as a continuous parent-child reflection score.

    For each Agent 2 dependency edge child -> parent:
      1. use the generated parent section as the upstream source
      2. extract key parent facts
      3. compare each parent fact against the generated child section
      4. score each fact using:
            NLI entailment = 1.0
            otherwise semantic similarity fallback
      5. average over all checked parent facts

    This measures whether downstream sections reflect upstream dependency
    information without requiring exact repetition.
    """
    if not plan:
        return {
            "cscs_status": "missing_plan",
            "cscs": "",
            "cscs_edges": 0,
            "cscs_checked_edges": 0,
            "cscs_checked_facts": 0,
            "cscs_error": "No Agent 2 plan found.",
        }

    dependency = plan.get("dependency", {}) or {}
    summaries = plan.get("summaries", {}) or {}

    total_edges = sum(len(parents or []) for parents in dependency.values())

    if total_edges == 0:
        return {
            "cscs_status": "no_edges_in_plan",
            "cscs": "",
            "cscs_edges": 0,
            "cscs_checked_edges": 0,
            "cscs_checked_facts": 0,
            "cscs_error": "Agent 2 plan contains no dependency edges.",
        }

    section_map = get_section_text_map(sections, article)
    scorer = get_cached_scorer()

    checked_edges = 0
    checked_facts = 0
    fact_score_sum = 0.0

    missing_child_edges = 0
    missing_parent_fact_edges = 0

    for child_section, parent_sections in dependency.items():
        child_text = find_section_text(child_section, section_map)

        for parent_section in parent_sections or []:
            checked_edges += 1

            # The dependency graph comes from Agent 2.
            # The upstream facts come from the generated parent section first.
            parent_text = find_section_text(parent_section, section_map)

            # Fallback: use Agent 2 summary only if the generated parent section is unavailable.
            if not parent_text:
                parent_text = summaries.get(parent_section, "")

            if not parent_text:
                missing_parent_fact_edges += 1
                continue

            parent_facts = extract_parent_facts(
                parent_text,
                max_facts=max_facts_per_edge,
            )

            if not parent_facts:
                missing_parent_fact_edges += 1
                continue

            # Missing child means the upstream facts are checked but receive score 0.
            if not child_text:
                missing_child_edges += 1

                for _ in parent_facts:
                    checked_facts += 1
                    fact_score_sum += 0.0

                continue

            for fact in parent_facts:
                checked_facts += 1

                score = parent_fact_reflection_score(
                    parent_fact=fact,
                    child_text=child_text,
                    nli=nli,
                    scorer=scorer,
                )

                fact_score_sum += score

    if checked_facts == 0:
        return {
            "cscs_status": "no_facts_checked",
            "cscs": "",
            "cscs_edges": total_edges,
            "cscs_checked_edges": checked_edges,
            "cscs_checked_facts": 0,
            "cscs_error": (
                "No parent facts could be checked against child sections. "
                f"missing_child_edges={missing_child_edges}; "
                f"missing_parent_fact_edges={missing_parent_fact_edges}"
            ),
        }

    return {
        "cscs_status": "success",
        "cscs": round(fact_score_sum / checked_facts, 4),
        "cscs_edges": total_edges,
        "cscs_checked_edges": checked_edges,
        "cscs_checked_facts": checked_facts,
        "cscs_error": "",
    }