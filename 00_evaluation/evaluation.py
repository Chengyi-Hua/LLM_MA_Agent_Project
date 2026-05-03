"""
00_evaluation/full_evaluation.py

Full evaluation driver for generated island Wikipedia-style articles.

It combines:
  1. Writing:
      - calls agents/agent4_evaluator.py
      - fluency_score
      - structure_score
      - organization_score
      - writing_score

  2. Informativeness:
      - ROUGE-L
      - METEOR
      - compares generated article against human Wikipedia reference

  3. Verifiability:
      - citation_recall
      - citation_precision
      - citation_rate
      - NLI-based citation checking

  4. Cross-section:
      - CSCS
      - checks whether facts from upstream dependency sections are preserved downstream

Run from project root.

Examples:

    python 00_evaluation/full_evaluation.py ^
      --input data/outputs/result_all_Hawaii_20260502_150004.json

    python 00_evaluation/full_evaluation.py ^
      --input-dir data/outputs

    python 00_evaluation/full_evaluation.py ^
      --input data/outputs/result_all_Hawaii_20260502_150004.json ^
      --skip-nli

Expected reference files:
    00_evaluation/references/Hawaii.txt

or:
    00_evaluation/references/Hawaii.json

JSON reference can contain one of:
    reference_article
    article
    content
    text
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents.agent4_evaluator import Agent4Evaluator
from methods.base_rag import load_config


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "evaluations")
DEFAULT_REFERENCES_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "references")
DEFAULT_CONTEXT_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_PLANS_DIR = os.path.join(PROJECT_ROOT, "logs", "agent2_plans")


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def strip_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text).strip()


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter that keeps citation markers attached.
    Good enough for evaluation; replace with spaCy later if needed.
    """
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"==.*?==", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if len(strip_citations(p).strip()) > 5]
    return sentences


def extract_citation_numbers(sentence: str) -> List[int]:
    nums = re.findall(r"\[(\d+)\]", sentence)
    return [int(n) for n in nums]


def normalize_url(url: str) -> str:
    if not url:
        return ""
    return str(url).strip().rstrip("/")


# ---------------------------------------------------------------------
# Input result normalization
# ---------------------------------------------------------------------

def normalize_result_file(path: str) -> List[dict]:
    """
    Supports:

    Shape A, full_pipeline --method all:
      {
        "method0": {...},
        "method1": {...}
      }

    Shape B, batch_experiments:
      {
        "metadata": {...},
        "result": {...}
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    # Shape B
    if isinstance(data, dict) and "metadata" in data and "result" in data:
        metadata = data.get("metadata", {})
        result = data.get("result", {})

        rows.append({
            "source_file": path,
            "input_json": metadata.get("input_json", ""),
            "island_name": metadata.get("island") or result.get("island_name") or "unknown",
            "method": metadata.get("method") or result.get("method") or "unknown",
            "generated_article": result.get("generated_article", ""),
            "sections": result.get("sections", []),
        })
        return rows

    # Shape A
    for method_name, result in data.items():
        if not isinstance(result, dict):
            continue
        if "generated_article" not in result:
            continue

        rows.append({
            "source_file": path,
            "input_json": "",
            "island_name": result.get("island_name", "unknown"),
            "method": method_name,
            "generated_article": result.get("generated_article", ""),
            "sections": result.get("sections", []),
        })

    return rows


def discover_input_files(input_path: Optional[str], input_dir: Optional[str]) -> List[str]:
    files = []

    if input_path:
        files.append(os.path.abspath(input_path))

    if input_dir:
        input_dir = os.path.abspath(input_dir)
        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(".json"):
                files.append(os.path.join(input_dir, fname))

    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)

    return unique


# ---------------------------------------------------------------------
# Reference loading for ROUGE-L / METEOR
# ---------------------------------------------------------------------

def load_reference_article(island_name: str, references_dir: str) -> Tuple[Optional[str], str]:
    """
    Looks for:
      references/Hawaii.txt
      references/Hawaii.json
      references/Hawaii_Island.txt
      references/Hawaii_Island.json
    """
    candidates = [
        island_name,
        safe_name(island_name),
        island_name.replace(" ", "_"),
    ]

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for base in candidates:
        txt_path = os.path.join(references_dir, f"{base}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read(), txt_path

        json_path = os.path.join(references_dir, f"{base}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in ["reference_article", "article", "content", "text", "generated_article"]:
                if isinstance(data, dict) and data.get(key):
                    return str(data[key]), json_path

    return None, ""


# ---------------------------------------------------------------------
# Informativeness: ROUGE-L and METEOR
# ---------------------------------------------------------------------

def lcs_length(a: List[str], b: List[str]) -> int:
    """
    O(len(a)*len(b)) LCS. Fine for moderate articles.
    """
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)

    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr

    return prev[-1]


def rouge_l_f1(generated: str, reference: str) -> float:
    """
    Returns ROUGE-L F1 on 0–100 scale.
    """
    gen_tokens = tokenize(generated)
    ref_tokens = tokenize(reference)

    if not gen_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(gen_tokens, ref_tokens)
    precision = lcs / len(gen_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return round(100 * (2 * precision * recall / (precision + recall)), 4)


def meteor_score(generated: str, reference: str) -> float:
    """
    Returns METEOR on 0–100 scale.

    Tries NLTK's METEOR first. If unavailable, falls back to a simple
    unigram F-mean approximation.
    """
    gen_tokens = tokenize(generated)
    ref_tokens = tokenize(reference)

    if not gen_tokens or not ref_tokens:
        return 0.0

    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score
        score = nltk_meteor_score([ref_tokens], gen_tokens)
        return round(100 * score, 4)
    except Exception:
        gen_set = set(gen_tokens)
        ref_set = set(ref_tokens)
        overlap = len(gen_set & ref_set)

        if overlap == 0:
            return 0.0

        precision = overlap / len(gen_set)
        recall = overlap / len(ref_set)

        # METEOR-like: recall weighted higher than precision
        alpha = 0.9
        denom = alpha * precision + (1 - alpha) * recall

        if denom == 0:
            return 0.0

        score = (precision * recall) / denom
        return round(100 * score, 4)


def compute_informativeness(generated: str, reference: Optional[str]) -> dict:
    if not reference:
        return {
            "rouge_l": "",
            "meteor": "",
            "informativeness_status": "missing_reference",
            "informativeness_error": "No reference article found.",
        }

    return {
        "rouge_l": rouge_l_f1(generated, reference),
        "meteor": meteor_score(generated, reference),
        "informativeness_status": "success",
        "informativeness_error": "",
    }


# ---------------------------------------------------------------------
# NLI evaluator for Verifiability and CSCS
# ---------------------------------------------------------------------

class NLIEvaluator:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        entailment_index: int = 1,
        threshold: float = 0.5,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.entailment_index = entailment_index
        self.threshold = threshold
        self.model = None

        if not enabled:
            return

        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading NLI model: {model_name}")
            self.model = CrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(
                "Could not load sentence-transformers CrossEncoder. "
                "Install it with: pip install sentence-transformers"
            ) from e

    @staticmethod
    def _softmax(values: List[float]) -> List[float]:
        m = max(values)
        exps = [math.exp(v - m) for v in values]
        total = sum(exps)
        return [v / total for v in exps]

    def entails(self, premise: str, hypothesis: str) -> Optional[bool]:
        if not self.enabled:
            return None

        if not premise or not hypothesis:
            return False

        premise = premise[:4000]
        hypothesis = hypothesis[:1000]

        raw = self.model.predict([(premise, hypothesis)])

        scores = raw[0]
        if not isinstance(scores, (list, tuple)):
            try:
                scores = scores.tolist()
            except Exception:
                scores = [float(scores)]

        if len(scores) == 1:
            return float(scores[0]) >= self.threshold

        probs = self._softmax([float(s) for s in scores])
        entail_prob = probs[self.entailment_index]

        return entail_prob >= self.threshold


# ---------------------------------------------------------------------
# Context loading for citation verification
# ---------------------------------------------------------------------

def infer_context_path(island_name: str, input_json: str, context_dir: str) -> str:
    if input_json and os.path.exists(input_json):
        return input_json

    candidates = [
        f"{safe_name(island_name)}_rag_context.json",
        f"{island_name.replace(' ', '_')}_rag_context.json",
        f"{island_name}_rag_context.json",
    ]

    for fname in candidates:
        path = os.path.join(context_dir, fname)
        if os.path.exists(path):
            return path

    return ""


def collect_url_texts(obj: Any, url_to_texts: Dict[str, List[str]]):
    """
    Recursively collect {url -> text chunks} from rag_context JSON.
    This is intentionally flexible because chunk schema may vary.
    """
    if isinstance(obj, dict):
        url = None
        for key in ["url", "source_url", "link", "href"]:
            if obj.get(key):
                url = normalize_url(obj.get(key))
                break

        text_parts = []
        for key in ["content", "text", "chunk", "body", "snippet", "summary"]:
            if obj.get(key) and isinstance(obj.get(key), str):
                text_parts.append(obj.get(key))

        if url and text_parts:
            url_to_texts[url].append("\n".join(text_parts))

        for value in obj.values():
            collect_url_texts(value, url_to_texts)

    elif isinstance(obj, list):
        for item in obj:
            collect_url_texts(item, url_to_texts)


def load_url_text_map(context_path: str) -> Dict[str, List[str]]:
    url_to_texts = defaultdict(list)

    if not context_path or not os.path.exists(context_path):
        return url_to_texts

    with open(context_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    collect_url_texts(data, url_to_texts)
    return url_to_texts


# ---------------------------------------------------------------------
# Verifiability: citation recall, precision, rate
# ---------------------------------------------------------------------

def get_section_text_map(sections: List[dict], article: str) -> Dict[str, str]:
    section_map = {}

    if sections:
        for sec in sections:
            name = sec.get("section_name", "")
            content = sec.get("content", "")
            if name:
                section_map[name.strip().lower()] = content

    if section_map:
        return section_map

    # Fallback: parse ==Section== blocks
    parts = re.split(r"==(.+?)==", article)
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        section_map[name.lower()] = content

    return section_map


def verify_sentence_with_citations(
    sentence: str,
    citations: List[str],
    section_citations: List[str],
    url_to_texts: Dict[str, List[str]],
    nli: Optional[NLIEvaluator],
) -> Tuple[int, int]:
    """
    Returns:
      supported_count, total_citations
    """
    if not citations:
        return 0, 0

    hypothesis = strip_citations(sentence)
    supported = 0
    total = 0

    for citation_num in citations:
        idx = citation_num - 1

        if idx < 0 or idx >= len(section_citations):
            total += 1
            continue

        url = normalize_url(section_citations[idx])
        passages = url_to_texts.get(url, [])

        # fallback: try normalized comparisons
        if not passages:
            for known_url, texts in url_to_texts.items():
                if normalize_url(known_url) == url:
                    passages = texts
                    break

        total += 1

        if not passages or nli is None:
            continue

        premise = "\n".join(passages)[:4000]

        try:
            if nli.entails(premise, hypothesis):
                supported += 1
        except Exception:
            pass

    return supported, total


def compute_verifiability(
    sections: List[dict],
    article: str,
    url_to_texts: Dict[str, List[str]],
    nli: Optional[NLIEvaluator],
) -> dict:
    total_sentences = 0
    recall_sum = 0.0
    precision_sum = 0.0
    weighted_recall_sum = 0.0
    total_words = 0

    cited_sentences = 0
    total_citations = 0
    supported_citations = 0

    if not sections:
        sections = [{"section_name": "article", "content": article, "citations": []}]

    for sec in sections:
        content = sec.get("content", "")
        section_citations = sec.get("citations", []) or []
        sentences = split_sentences(content)

        for sent in sentences:
            nums = extract_citation_numbers(sent)
            wc = word_count(strip_citations(sent))

            if wc == 0:
                continue

            total_sentences += 1
            total_words += wc

            if nums:
                cited_sentences += 1

            supported, total = verify_sentence_with_citations(
                sentence=sent,
                citations=nums,
                section_citations=section_citations,
                url_to_texts=url_to_texts,
                nli=nli,
            )

            total_citations += total
            supported_citations += supported

            recall_i = 1.0 if supported > 0 else 0.0

            if total > 0:
                precision_i = supported / total
            else:
                precision_i = 0.0

            recall_sum += recall_i
            precision_sum += precision_i
            weighted_recall_sum += wc * recall_i

    if total_sentences == 0:
        return {
            "citation_recall": "",
            "citation_precision": "",
            "citation_rate": "",
            "num_sentences": 0,
            "num_cited_sentences": 0,
            "num_citations": 0,
            "num_supported_citations": 0,
            "verifiability_status": "no_sentences",
            "verifiability_error": "No sentences found.",
        }

    return {
        "citation_recall": round(recall_sum / total_sentences, 4),
        "citation_precision": round(precision_sum / total_sentences, 4),
        "citation_rate": round(weighted_recall_sum / total_words, 4) if total_words else 0.0,
        "num_sentences": total_sentences,
        "num_cited_sentences": cited_sentences,
        "num_citations": total_citations,
        "num_supported_citations": supported_citations,
        "verifiability_status": "success",
        "verifiability_error": "",
    }


# ---------------------------------------------------------------------
# CSCS
# ---------------------------------------------------------------------

def load_agent2_plan(island_name: str, plans_dir: str) -> Tuple[Optional[dict], str]:
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

    return None, ""


def compute_cscs(
    sections: List[dict],
    article: str,
    plan: Optional[dict],
    nli: Optional[NLIEvaluator],
    max_facts_per_edge: int = 5,
) -> dict:
    if not plan:
        return {
            "cscs": "",
            "cscs_edges": 0,
            "cscs_checked_facts": 0,
            "cscs_status": "missing_plan",
            "cscs_error": "No Agent 2 plan found.",
        }

    if nli is None:
        return {
            "cscs": "",
            "cscs_edges": 0,
            "cscs_checked_facts": 0,
            "cscs_status": "missing_nli",
            "cscs_error": "NLI disabled or unavailable.",
        }

    dependency = plan.get("dependency", {}) or {}
    summaries = plan.get("summaries", {}) or {}

    section_map = get_section_text_map(sections, article)

    checked = 0
    preserved = 0
    edge_count = 0

    for downstream, upstream_list in dependency.items():
        downstream_text = section_map.get(downstream.strip().lower(), "")

        if not downstream_text:
            continue

        for upstream in upstream_list:
            edge_count += 1

            fact_source = summaries.get(upstream, "")
            if not fact_source:
                fact_source = section_map.get(upstream.strip().lower(), "")

            facts = split_sentences(fact_source)[:max_facts_per_edge]

            for fact in facts:
                fact = strip_citations(fact)

                if not fact:
                    continue

                checked += 1

                try:
                    if nli.entails(downstream_text, fact):
                        preserved += 1
                except Exception:
                    pass

    if checked == 0:
        return {
            "cscs": "",
            "cscs_edges": edge_count,
            "cscs_checked_facts": 0,
            "cscs_status": "no_facts_checked",
            "cscs_error": "No dependency facts could be checked.",
        }

    return {
        "cscs": round(preserved / checked, 4),
        "cscs_edges": edge_count,
        "cscs_checked_facts": checked,
        "cscs_status": "success",
        "cscs_error": "",
    }


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def write_csv(output_path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp",
        "source_file",
        "input_json",
        "island_name",
        "method",

        "writing_status",
        "fluency_score",
        "structure_score",
        "organization_score",
        "writing_score",
        "writing_rationale",
        "writing_error",

        "reference_file",
        "informativeness_status",
        "rouge_l",
        "meteor",
        "informativeness_error",

        "context_file",
        "verifiability_status",
        "citation_recall",
        "citation_precision",
        "citation_rate",
        "num_sentences",
        "num_cited_sentences",
        "num_citations",
        "num_supported_citations",
        "verifiability_error",

        "plan_file",
        "cscs_status",
        "cscs",
        "cscs_edges",
        "cscs_checked_facts",
        "cscs_error",
    ]

    exists = os.path.exists(output_path)

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate(args):
    input_files = discover_input_files(args.input, args.input_dir)

    if not input_files:
        raise FileNotFoundError("No input files found. Use --input or --input-dir.")

    output_path = args.output or os.path.join(
        DEFAULT_OUTPUT_DIR,
        f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    config = load_config()

    writing_evaluator = None
    if not args.skip_writing:
        writing_evaluator = Agent4Evaluator(config=config)

    nli = None
    if not args.skip_nli:
        nli_model = args.nli_model

        if not nli_model:
            nli_model = (
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

        nli = NLIEvaluator(
            model_name=nli_model,
            entailment_index=entailment_index,
            threshold=args.nli_threshold,
            enabled=True,
        )

    print("\n" + "=" * 80)
    print("Full Evaluation")
    print("=" * 80)
    print(f"Input files:    {len(input_files)}")
    print(f"Output CSV:     {output_path}")
    print(f"Writing judge:  {'enabled' if writing_evaluator else 'skipped'}")
    print(f"NLI:            {'enabled' if nli else 'skipped'}")
    print("=" * 80 + "\n")

    for input_file in input_files:
        print(f"\nReading: {input_file}")

        try:
            items = normalize_result_file(input_file)
        except Exception as e:
            print(f"❌ Could not read {input_file}: {e}")
            continue

        for item in items:
            island = item["island_name"]
            method = item["method"]
            article = item["generated_article"]
            sections = item["sections"]

            print(f"\nEvaluating: {island} | {method}")

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source_file": item.get("source_file", ""),
                "input_json": item.get("input_json", ""),
                "island_name": island,
                "method": method,

                "writing_status": "",
                "fluency_score": "",
                "structure_score": "",
                "organization_score": "",
                "writing_score": "",
                "writing_rationale": "",
                "writing_error": "",

                "reference_file": "",
                "informativeness_status": "",
                "rouge_l": "",
                "meteor": "",
                "informativeness_error": "",

                "context_file": "",
                "verifiability_status": "",
                "citation_recall": "",
                "citation_precision": "",
                "citation_rate": "",
                "num_sentences": "",
                "num_cited_sentences": "",
                "num_citations": "",
                "num_supported_citations": "",
                "verifiability_error": "",

                "plan_file": "",
                "cscs_status": "",
                "cscs": "",
                "cscs_edges": "",
                "cscs_checked_facts": "",
                "cscs_error": "",
            }

            # 1. Writing
            if writing_evaluator:
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
            else:
                row["writing_status"] = "skipped"

            # 2. Informativeness
            reference, reference_file = load_reference_article(island, args.references_dir)
            row["reference_file"] = reference_file

            info = compute_informativeness(article, reference)
            row.update(info)

            # 3. Verifiability
            context_path = infer_context_path(
                island_name=island,
                input_json=item.get("input_json", ""),
                context_dir=args.context_dir,
            )
            row["context_file"] = context_path

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

            # 4. CSCS
            plan, plan_file = load_agent2_plan(island, args.plans_dir)
            row["plan_file"] = plan_file

            if args.skip_cscs:
                row["cscs_status"] = "skipped"
                row["cscs_error"] = "CSCS skipped."
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
                    row["cscs_error"] = str(e)

            write_csv(output_path, [row])

            print(
                f"✅ {island} | {method} | "
                f"writing={row['writing_score']} | "
                f"R-L={row['rouge_l']} | "
                f"METEOR={row['meteor']} | "
                f"cit_recall={row['citation_recall']} | "
                f"cscs={row['cscs']}"
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("\n" + "=" * 80)
    print("Full evaluation complete")
    print("=" * 80)
    print(f"Saved to: {output_path}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation driver for generated Wikipedia-style articles."
    )

    parser.add_argument("--input", default=None, help="Path to one result JSON file.")
    parser.add_argument("--input-dir", default=None, help="Directory of result JSON files.")

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
        "--output",
        default=None,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--skip-writing",
        action="store_true",
        help="Skip Agent 4 LLM writing evaluation.",
    )

    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip NLI-based citation checking.",
    )

    parser.add_argument(
        "--skip-cscs",
        action="store_true",
        help="Skip CSCS.",
    )

    parser.add_argument(
        "--nli-model",
        default=None,
        help="NLI model name. Defaults to settings.yaml agent2 graph_logic model.",
    )

    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="Entailment probability threshold. Default: 0.5.",
    )

    parser.add_argument(
        "--cscs-max-facts",
        type=int,
        default=5,
        help="Maximum upstream facts checked per dependency edge.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep between article evaluations.",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()