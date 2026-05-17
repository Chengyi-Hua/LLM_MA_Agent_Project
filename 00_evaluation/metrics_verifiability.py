"""
Verifiability metrics:
  - citation_recall
  - citation_precision
  - citation_link_precision
  - citation_rate

Definitions used here:
  citation_rate           = cited sentences / total sentences
  citation_precision      = supported cited sentences / cited sentences
  citation_recall         = supported cited sentences / total sentences
  citation_link_precision = supported citation links / total citation links

Diagnostic counts:
  num_citation_links
  num_supported_citation_links
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from eval_utils import (
    extract_citation_numbers,
    normalize_url,
    safe_name,
    split_sentences,
    strip_citations,
)


class NLIEvaluator:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        entailment_index: int = 1,
        threshold: float = 0.5,
    ):
        from sentence_transformers import CrossEncoder

        print(f"Loading NLI model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.entailment_index = entailment_index
        self.threshold = threshold

    @staticmethod
    def _softmax(values: List[float]) -> List[float]:
        m = max(values)
        exps = [math.exp(v - m) for v in values]
        total = sum(exps)
        return [v / total for v in exps]

    def entails(self, premise: str, hypothesis: str) -> bool:
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

def infer_context_path(island_name: str, input_json: str, context_dir: str) -> str:
    """
    Find the matching *_rag_context.json file with alias support.

    Handles cases like:
      Hashima  -> Hashima_Island_rag_context.json
      Howland  -> Howland_Island_rag_context.json
      Jarvis   -> Jarvis_Island_rag_context.json
      Tromelin -> Tromelin_Island_rag_context.json
    """
    if input_json and os.path.exists(input_json):
        return input_json

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
                f"{safe_name(name)}_rag_context.json",
                f"{str(name).replace(' ', '_')}_rag_context.json",
                f"{name}_rag_context.json",
            ]
        )

    seen = set()
    candidates = [
        c for c in candidates
        if c and not (c in seen or seen.add(c))
    ]

    for fname in candidates:
        path = os.path.join(context_dir, fname)

        if os.path.exists(path):
            return path

    return ""


def collect_url_texts(obj: Any, url_to_texts: Dict[str, List[str]]):
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


def split_claim_units(sentence: str, max_units: int = 6) -> List[str]:
    """
    Split a sentence into smaller claim-like units.

    This is used because a citation may support one factual clause in a
    sentence even when it does not entail the entire sentence.
    """
    text = strip_citations(sentence)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    units = [text]

    parts = re.split(
        r"\s*(?:;|:|—|–)\s*|\s+\b(?:but|while|whereas|although|though|however)\b\s+",
        text,
        flags=re.IGNORECASE,
    )

    for part in parts:
        part = part.strip(" ,.;:-")

        if len(part.split()) >= 4:
            units.append(part)

    # Remove duplicates while preserving order.
    seen = set()
    unique_units = []

    for unit in units:
        key = unit.lower()

        if key not in seen:
            unique_units.append(unit)
            seen.add(key)

        if len(unique_units) >= max_units:
            break

    return unique_units


def citation_supports_any_claim(
    passages: List[str],
    claim_units: List[str],
    nli: NLIEvaluator,
) -> bool:
    """
    Return True if any retrieved passage entails at least one claim unit.

    This avoids concatenating all passages from the same URL and then losing
    relevant evidence because of the 4000-character NLI truncation.
    """
    for passage in passages:
        if not passage or not str(passage).strip():
            continue

        for claim in claim_units:
            try:
                if nli.entails(str(passage), claim):
                    return True
            except Exception:
                pass

    return False

def verify_sentence_with_citations(
    sentence: str,
    citation_numbers: List[int],
    section_citations: List[str],
    url_to_texts: Dict[str, List[str]],
    nli: NLIEvaluator,
) -> Tuple[int, int, bool]:
    """
    Returns:
      supported_links
      total_links
      sentence_supported

    A citation link is counted as supported if its source passage entails
    either:
      - the full sentence, or
      - at least one smaller claim unit inside the sentence.

    This avoids unfairly marking multi-claim sentences as unsupported when
    the citation supports one factual part of the sentence.
    """
    if not citation_numbers:
        return 0, 0, False

    claim_units = split_claim_units(sentence)

    if not claim_units:
        return 0, 0, False

    supported_links = 0
    total_links = 0

    for citation_num in citation_numbers:
        idx = citation_num - 1

        if idx < 0 or idx >= len(section_citations):
            total_links += 1
            continue

        url = normalize_url(section_citations[idx])
        passages = url_to_texts.get(url, [])

        if not passages:
            for known_url, texts in url_to_texts.items():
                if normalize_url(known_url) == url:
                    passages = texts
                    break

        total_links += 1

        if not passages:
            continue

        citation_supports_sentence = citation_supports_any_claim(
            passages=passages,
            claim_units=claim_units,
            nli=nli,
        )

        if citation_supports_sentence:
            supported_links += 1

    sentence_supported = supported_links > 0

    return supported_links, total_links, sentence_supported


def compute_verifiability(
    sections: List[dict],
    article: str,
    url_to_texts: Dict[str, List[str]],
    nli: NLIEvaluator,
) -> dict:
    if not sections:
        sections = [{"section_name": "article", "content": article, "citations": []}]

    total_sentences = 0
    cited_sentences = 0
    supported_cited_sentences = 0

    total_citation_links = 0
    supported_citation_links = 0

    for sec in sections:
        content = sec.get("content", "")
        section_citations = sec.get("citations", []) or []

        for sentence in split_sentences(content):
            total_sentences += 1

            citation_numbers = extract_citation_numbers(sentence)

            if citation_numbers:
                cited_sentences += 1

            supported_links, total_links, sentence_supported = verify_sentence_with_citations(
                sentence=sentence,
                citation_numbers=citation_numbers,
                section_citations=section_citations,
                url_to_texts=url_to_texts,
                nli=nli,
            )

            total_citation_links += total_links
            supported_citation_links += supported_links

            if sentence_supported:
                supported_cited_sentences += 1

    if total_sentences == 0:
        return {
            "verifiability_status": "no_sentences",
            "citation_recall": "",
            "citation_precision": "",
            "citation_link_precision": "",
            "citation_rate": "",
            "num_sentences": 0,
            "num_cited_sentences": 0,
            "num_citation_links": 0,
            "num_supported_citation_links": 0,
            "verifiability_error": "No sentences found.",
        }

    citation_rate = cited_sentences / total_sentences

    citation_recall = supported_cited_sentences / total_sentences

    # Sentence-level citation precision:
    # Among cited sentences, how many are supported by at least one citation?
    citation_precision = (
        supported_cited_sentences / cited_sentences
        if cited_sentences > 0
        else 0.0
    )

    citation_link_precision = (
        supported_citation_links / total_citation_links
        if total_citation_links > 0
        else 0.0
    )

    return {
        "verifiability_status": "success",
        "citation_recall": round(citation_recall, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_link_precision": round(citation_link_precision, 4),
        "citation_rate": round(citation_rate, 4),
        "num_sentences": total_sentences,
        "num_cited_sentences": cited_sentences,
        "num_citation_links": total_citation_links,
        "num_supported_citation_links": supported_citation_links,
        "verifiability_error": "",
    }