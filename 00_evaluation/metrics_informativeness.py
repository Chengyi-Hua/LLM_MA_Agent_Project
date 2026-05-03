"""
Informativeness metrics:
  - ROUGE-L
  - METEOR
"""

from typing import List, Optional

from eval_utils import tokenize


def lcs_length(a: List[str], b: List[str]) -> int:
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
    ROUGE-L F1 on 0–100 scale.
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
    METEOR on 0–100 scale.

    Uses NLTK if available.
    Falls back to a simple unigram approximation if NLTK is unavailable.
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

        alpha = 0.9
        denom = alpha * precision + (1 - alpha) * recall

        if denom == 0:
            return 0.0

        score = (precision * recall) / denom
        return round(100 * score, 4)


def compute_informativeness(generated: str, reference: Optional[str]) -> dict:
    if not reference:
        return {
            "informativeness_status": "missing_reference",
            "rouge_l": "",
            "meteor": "",
            "informativeness_error": "No reference article found.",
        }

    return {
        "informativeness_status": "success",
        "rouge_l": rouge_l_f1(generated, reference),
        "meteor": meteor_score(generated, reference),
        "informativeness_error": "",
    }