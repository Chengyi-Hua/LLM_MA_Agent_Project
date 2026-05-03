"""
agents/agent4_evaluator.py

Agent 4: LLM-as-a-Judge Evaluator.

This agent supports two LLM-judge evaluation modes:

1. Writing evaluation
   - fluency_score       (0–5)
   - structure_score     (0–5)
   - organization_score  (0–5)
   - writing_score       average of the three

2. Concept-based evaluation
   - concept_coverage_score      (0–5)
   - concept_accuracy_score      (0–5)
   - concept_relevance_score     (0–5)
   - concept_organization_score  (0–5)
   - concept_score               average of the four
   - missing_key_concepts
   - inaccurate_or_unsupported_concepts

Other metrics are implemented separately:
  - ROUGE-L
  - METEOR
  - citation recall
  - citation precision
  - citation rate
  - CSCS
  - artifact diagnostics
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from methods.base_rag import load_config


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "evaluations")


# ---------------------------------------------------------------------
# Writing evaluation prompts
# ---------------------------------------------------------------------

AGENT4_WRITING_SYSTEM_PROMPT = """
You are Agent 4, an expert evaluator of Wikipedia-style article writing.

You evaluate ONLY writing quality.
Do not evaluate factual correctness.
Do not evaluate citation correctness.
Do not compare the article against a reference article.
Do not reward the article simply for being long.
Do not punish the article for missing facts unless that affects writing structure or organization.

Return only valid JSON.
""".strip()


AGENT4_WRITING_PROMPT = """
Evaluate the writing quality of the following generated Wikipedia-style article.

Target entity: {island_name}
Generation method: {method_name}

Rate the article on exactly three metrics, each from 0 to 5.

1. fluency_score
Evaluate grammar, readability, sentence-level clarity, and whether the prose sounds natural.

0 = unreadable or incoherent
1 = very poor fluency with major grammar/readability issues
2 = weak fluency; understandable but awkward or repetitive
3 = acceptable fluency with some issues
4 = good fluency; mostly clear and readable
5 = excellent fluency; polished Wikipedia-style prose

2. structure_score
Evaluate the article's section structure. This corresponds to section/outline quality.
Check whether the article has appropriate section headings, whether sections are meaningful,
and whether the structure fits a Wikipedia-style article about the target entity.

0 = no useful structure
1 = very poor section structure
2 = weak structure; headings are missing, confusing, or poorly chosen
3 = acceptable structure but incomplete or uneven
4 = good section structure with mostly appropriate headings
5 = excellent Wikipedia-style structure with clear and comprehensive headings

3. organization_score
Evaluate logical organization and coherence within and across sections.
Check whether ideas flow in a sensible order and whether the article avoids unnecessary repetition.

0 = disorganized or incoherent
1 = very poor organization
2 = weak organization with major flow or repetition problems
3 = acceptable organization with some issues
4 = good organization and mostly logical flow
5 = excellent organization; coherent, well connected, and easy to follow

Return ONLY valid JSON in this exact format:

{{
  "fluency_score": 0,
  "structure_score": 0,
  "organization_score": 0,
  "brief_rationale": "one or two concise sentences explaining the scores"
}}

Article:
\"\"\"
{article}
\"\"\"
""".strip()


# ---------------------------------------------------------------------
# Concept evaluation prompts
# ---------------------------------------------------------------------

AGENT4_CONCEPT_SYSTEM_PROMPT = """
You are Agent 4, an expert evaluator of Wikipedia-style article content.

You evaluate concept-level content quality by comparing a generated article
against a human-written reference Wikipedia article.

Focus on key concepts, important facts, topical coverage, and conceptual accuracy.
Do not evaluate citation formatting.
Do not evaluate prose fluency unless it affects conceptual organization.
Do not require exact wording from the reference article.
Do not reward unnecessary length.
Do not punish different but reasonable section names if the key concepts are covered.

Return only valid JSON.
""".strip()


AGENT4_CONCEPT_PROMPT = """
Evaluate the concept-level quality of the generated Wikipedia-style article by comparing it
against the human-written reference article.

Target entity: {island_name}
Generation method: {method_name}

You must rate exactly four metrics, each from 0 to 5.

1. concept_coverage_score
How well does the generated article cover the key concepts from the reference article?

0 = covers almost none of the key concepts
1 = covers very few key concepts
2 = covers some key concepts but misses many important ones
3 = covers several important concepts but is incomplete
4 = covers most key concepts
5 = covers nearly all central concepts from the reference article

2. concept_accuracy_score
Are the concepts that appear in the generated article accurate with respect to the reference?

0 = mostly inaccurate or misleading
1 = many serious inaccuracies
2 = several inaccuracies or distortions
3 = mostly accurate with some issues
4 = accurate with only minor issues
5 = highly accurate representation of the reference concepts

3. concept_relevance_score
Is the generated article focused on the target entity and free from irrelevant or off-topic content?

0 = mostly irrelevant
1 = much irrelevant content
2 = some relevant content but substantial off-topic material
3 = mostly relevant with some off-topic or weakly related material
4 = relevant and focused
5 = highly focused on the target entity

4. concept_organization_score
Are the covered concepts arranged in a sensible Wikipedia-style conceptual structure?

0 = concepts are incoherent or randomly arranged
1 = very poor conceptual organization
2 = weak organization with major placement issues
3 = acceptable organization
4 = good conceptual organization
5 = excellent conceptual organization

Return ONLY valid JSON in this exact format:

{{
  "concept_coverage_score": 0,
  "concept_accuracy_score": 0,
  "concept_relevance_score": 0,
  "concept_organization_score": 0,
  "missing_key_concepts": [
    "short phrase for an important missing concept"
  ],
  "inaccurate_or_unsupported_concepts": [
    "short phrase for an inaccurate, distorted, or unsupported concept"
  ],
  "brief_rationale": "one or two concise sentences explaining the concept scores"
}}

Reference Wikipedia article:
\"\"\"
{reference_article}
\"\"\"

Generated article:
\"\"\"
{generated_article}
\"\"\"
""".strip()


class Agent4Evaluator:
    """
    Agent 4: LLM-as-a-Judge evaluator.

    Main public methods:
      - evaluate_article(...)   -> writing evaluation
      - evaluate_concepts(...)  -> concept-based reference evaluation
    """

    agent_key = "agent4"

    def __init__(self, config: Optional[dict] = None):
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

        self.config = config or load_config()
        self.llm_config = self.config.get("llm", {})
        self.agent_config = self.llm_config.get(self.agent_key, {})

        self.provider = self.agent_config.get("provider", "openrouter")
        self.model = self.agent_config.get("model", "openrouter/free")

        self.temperature = self.agent_config.get(
            "temperature",
            self.llm_config.get("temperature", 0.0),
        )

        self.max_tokens = self.agent_config.get("max_tokens", 900)

        # Keep prompt sizes bounded for free/routed models.
        self.generated_article_max_chars = self.agent_config.get(
            "generated_article_max_chars",
            18000,
        )
        self.reference_article_max_chars = self.agent_config.get(
            "reference_article_max_chars",
            22000,
        )

        self.client = self._build_client()

    def _build_client(self) -> OpenAI:
        api_keys = self.config.get("api_keys", {})

        if self.provider == "openrouter":
            env_name = api_keys.get("openrouter_env", "OPENROUTER_API_KEY")
            api_key = os.getenv(env_name)

            if not api_key:
                raise EnvironmentError(
                    f"{env_name} not found. Add it to your project-root .env file."
                )

            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

        if self.provider == "openai":
            env_name = api_keys.get("openai_env", "OPENAI_API_KEY")
            api_key = os.getenv(env_name)

            if not api_key:
                raise EnvironmentError(
                    f"{env_name} not found. Add it to your project-root .env file."
                )

            return OpenAI(api_key=api_key)

        raise ValueError(f"Unsupported Agent 4 provider: {self.provider}")

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        text = str(text or "").strip()

        if len(text) <= max_chars:
            return text

        head = text[: int(max_chars * 0.65)]
        tail = text[-int(max_chars * 0.35):]

        return (
            head
            + "\n\n[... middle of article truncated for judge context length ...]\n\n"
            + tail
        )

    def _call_judge(
        self,
        prompt: str,
        system_prompt: str,
        retries: int = 10,
    ) -> str:
        """
        Calls the judge model with bounded retry.

        Reason:
        - openrouter/free can be unstable or routed to different providers.
        - Some endpoints reject reasoning options.
        - Odd attempts try reasoning.exclude=True.
        - Even attempts use a plain request.
        """
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }

                if attempt % 2 == 1:
                    request_kwargs["extra_body"] = {
                        "reasoning": {
                            "exclude": True
                        }
                    }

                response = self.client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content

                if content is None or not str(content).strip():
                    raise ValueError(
                        f"Judge returned empty content. "
                        f"finish_reason={response.choices[0].finish_reason}"
                    )

                return content.strip()

            except Exception as e:
                last_error = e
                wait_seconds = min(3 * attempt, 20)

                print(
                    f"! Agent 4 judge call failed on attempt "
                    f"{attempt}/{retries}: {e}"
                )
                print(f"   Waiting {wait_seconds}s before retrying...")

                time.sleep(wait_seconds)

        raise RuntimeError(
            f"Agent 4 judge failed after {retries} attempts: {last_error}"
        )

    def _extract_json(self, text: str) -> dict:
        if text is None:
            raise ValueError("Cannot parse JSON from None.")

        text = text.strip()

        fenced = re.search(
            r"```(?:json)?\s*(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced:
            text = fenced.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"Could not parse JSON from judge response:\n{text}")

    @staticmethod
    def _clamp_score(value: Any) -> Optional[float]:
        try:
            score = float(value)
        except Exception:
            return None

        if score < 0:
            return 0.0
        if score > 5:
            return 5.0

        return round(score, 2)

    @staticmethod
    def _clean_list(value: Any, max_items: int = 8) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            return []

        cleaned = []
        for item in value:
            item = str(item).strip()
            if item:
                cleaned.append(item)

        return cleaned[:max_items]

    def _normalize_writing_scores(self, data: dict) -> dict:
        fluency = self._clamp_score(data.get("fluency_score"))
        structure = self._clamp_score(data.get("structure_score"))
        organization = self._clamp_score(data.get("organization_score"))

        valid_scores = [
            s for s in [fluency, structure, organization]
            if s is not None
        ]

        writing_score = (
            round(sum(valid_scores) / len(valid_scores), 2)
            if valid_scores
            else None
        )

        return {
            "fluency_score": fluency,
            "structure_score": structure,
            "organization_score": organization,
            "writing_score": writing_score,
            "brief_rationale": str(data.get("brief_rationale", "")).strip(),
        }

    def _normalize_concept_scores(self, data: dict) -> dict:
        coverage = self._clamp_score(data.get("concept_coverage_score"))
        accuracy = self._clamp_score(data.get("concept_accuracy_score"))
        relevance = self._clamp_score(data.get("concept_relevance_score"))
        organization = self._clamp_score(data.get("concept_organization_score"))

        valid_scores = [
            s for s in [coverage, accuracy, relevance, organization]
            if s is not None
        ]

        concept_score = (
            round(sum(valid_scores) / len(valid_scores), 2)
            if valid_scores
            else None
        )

        return {
            "concept_coverage_score": coverage,
            "concept_accuracy_score": accuracy,
            "concept_relevance_score": relevance,
            "concept_organization_score": organization,
            "concept_score": concept_score,
            "missing_key_concepts": self._clean_list(
                data.get("missing_key_concepts"),
                max_items=10,
            ),
            "inaccurate_or_unsupported_concepts": self._clean_list(
                data.get("inaccurate_or_unsupported_concepts"),
                max_items=10,
            ),
            "brief_rationale": str(data.get("brief_rationale", "")).strip(),
        }

    # ------------------------------------------------------------------
    # Public method 1: writing evaluation
    # ------------------------------------------------------------------

    def evaluate_article(
        self,
        article: str,
        island_name: str,
        method_name: str,
    ) -> dict:
        """
        Backward-compatible writing evaluator.
        full_evaluation.py already calls this method.
        """
        if article is None or not article.strip():
            raise ValueError("Generated article is empty.")

        clipped_article = self._clip_text(
            article,
            self.generated_article_max_chars,
        )

        prompt = AGENT4_WRITING_PROMPT.format(
            island_name=island_name,
            method_name=method_name,
            article=clipped_article,
        )

        raw_response = self._call_judge(
            prompt=prompt,
            system_prompt=AGENT4_WRITING_SYSTEM_PROMPT,
        )

        parsed = self._extract_json(raw_response)
        scores = self._normalize_writing_scores(parsed)

        return {
            "status": "success",
            "agent": "agent4",
            "task": "writing_evaluation",
            "island_name": island_name,
            "method": method_name,
            "judge_provider": self.provider,
            "judge_model": self.model,
            "scores": scores,
            "raw_response": raw_response,
        }

    # ------------------------------------------------------------------
    # Public method 2: concept-based evaluation
    # ------------------------------------------------------------------

    def evaluate_concepts(
        self,
        generated_article: str,
        reference_article: str,
        island_name: str,
        method_name: str,
    ) -> dict:
        """
        Concept-based LLM-as-a-judge evaluation.

        Compares generated article against the human-written reference article.
        """
        if generated_article is None or not generated_article.strip():
            raise ValueError("Generated article is empty.")

        if reference_article is None or not reference_article.strip():
            raise ValueError("Reference article is empty or missing.")

        clipped_generated = self._clip_text(
            generated_article,
            self.generated_article_max_chars,
        )

        clipped_reference = self._clip_text(
            reference_article,
            self.reference_article_max_chars,
        )

        prompt = AGENT4_CONCEPT_PROMPT.format(
            island_name=island_name,
            method_name=method_name,
            reference_article=clipped_reference,
            generated_article=clipped_generated,
        )

        raw_response = self._call_judge(
            prompt=prompt,
            system_prompt=AGENT4_CONCEPT_SYSTEM_PROMPT,
        )

        parsed = self._extract_json(raw_response)
        scores = self._normalize_concept_scores(parsed)

        return {
            "status": "success",
            "agent": "agent4",
            "task": "concept_evaluation",
            "island_name": island_name,
            "method": method_name,
            "judge_provider": self.provider,
            "judge_model": self.model,
            "scores": scores,
            "raw_response": raw_response,
        }

    # ------------------------------------------------------------------
    # Existing CLI helpers: writing-only evaluation
    # ------------------------------------------------------------------

    def evaluate_file(self, input_path: str) -> List[dict]:
        with open(input_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        return self.evaluate_result_dict(result_data, source_file=input_path)

    def evaluate_result_dict(self, result_data: dict, source_file: str = "") -> List[dict]:
        items = self._normalize_result_dict(result_data, source_file=source_file)
        evaluations = []

        for item in items:
            island_name = item["island_name"]
            method_name = item["method"]
            article = item["generated_article"]

            print(f"Agent 4 evaluating writing: {island_name} | {method_name}")

            try:
                evaluation = self.evaluate_article(
                    article=article,
                    island_name=island_name,
                    method_name=method_name,
                )

            except Exception as e:
                evaluation = {
                    "status": "failed",
                    "agent": "agent4",
                    "task": "writing_evaluation",
                    "island_name": island_name,
                    "method": method_name,
                    "judge_provider": self.provider,
                    "judge_model": self.model,
                    "scores": {},
                    "raw_response": "",
                    "error": str(e),
                }

            evaluation["source_file"] = source_file
            evaluations.append(evaluation)

        return evaluations

    def _normalize_result_dict(self, data: dict, source_file: str = "") -> List[dict]:
        rows = []

        # Shape B: batch_experiments output
        # {
        #   "metadata": {...},
        #   "result": {...}
        # }
        if isinstance(data, dict) and "metadata" in data and "result" in data:
            metadata = data.get("metadata", {})
            result = data.get("result", {})

            rows.append(
                {
                    "source_file": source_file,
                    "island_name": metadata.get("island")
                    or result.get("island_name")
                    or "unknown",
                    "method": metadata.get("method")
                    or result.get("method")
                    or "unknown",
                    "generated_article": result.get("generated_article", ""),
                }
            )

            return rows

        # Shape A: full_pipeline --method all output
        # {
        #   "method0": {...},
        #   "method1": {...}
        # }
        for method_name, method_result in data.items():
            if not isinstance(method_result, dict):
                continue

            if "generated_article" not in method_result:
                continue

            rows.append(
                {
                    "source_file": source_file,
                    "island_name": method_result.get("island_name", "unknown"),
                    "method": method_name,
                    "generated_article": method_result.get("generated_article", ""),
                }
            )

        return rows


def flatten_evaluation(evaluation: dict) -> dict:
    scores = evaluation.get("scores", {})

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_file": evaluation.get("source_file", ""),
        "island_name": evaluation.get("island_name", ""),
        "method": evaluation.get("method", ""),
        "status": evaluation.get("status", ""),
        "judge_provider": evaluation.get("judge_provider", ""),
        "judge_model": evaluation.get("judge_model", ""),
        "fluency_score": scores.get("fluency_score", ""),
        "structure_score": scores.get("structure_score", ""),
        "organization_score": scores.get("organization_score", ""),
        "writing_score": scores.get("writing_score", ""),
        "brief_rationale": scores.get("brief_rationale", ""),
        "error": evaluation.get("error", ""),
    }


def write_csv(output_path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "timestamp",
        "source_file",
        "island_name",
        "method",
        "status",
        "judge_provider",
        "judge_model",
        "fluency_score",
        "structure_score",
        "organization_score",
        "writing_score",
        "brief_rationale",
        "error",
    ]

    file_exists = os.path.exists(output_path)

    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


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
    unique_files = []

    for path in files:
        if path not in seen:
            unique_files.append(path)
            seen.add(path)

    return unique_files


def main():
    parser = argparse.ArgumentParser(
        description="Agent 4: LLM-as-a-Judge evaluator for writing quality."
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Path to one result JSON file.",
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing result JSON files.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to 00_evaluation/evaluations/.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between files. Default: 0.5.",
    )

    args = parser.parse_args()

    input_files = discover_input_files(args.input, args.input_dir)

    if not input_files:
        raise FileNotFoundError("No input JSON files found. Use --input or --input-dir.")

    output_path = args.output or os.path.join(
        DEFAULT_OUTPUT_DIR,
        f"agent4_writing_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    evaluator = Agent4Evaluator()

    print("\n" + "=" * 80)
    print("Agent 4 Writing Evaluation")
    print("=" * 80)
    print(f"Judge provider: {evaluator.provider}")
    print(f"Judge model:    {evaluator.model}")
    print(f"Input files:    {len(input_files)}")
    print(f"Output CSV:     {output_path}")
    print("=" * 80 + "\n")

    for input_file in input_files:
        print(f"\nReading: {input_file}")

        evaluations = evaluator.evaluate_file(input_file)
        rows = [flatten_evaluation(e) for e in evaluations]
        write_csv(output_path, rows)

        for row in rows:
            if row["status"] == "success":
                print(
                    f"{row['island_name']} | {row['method']} | "
                    f"writing={row['writing_score']} "
                    f"fluency={row['fluency_score']} "
                    f"structure={row['structure_score']} "
                    f"organization={row['organization_score']}"
                )
            else:
                print(
                    f"{row['island_name']} | {row['method']} | "
                    f"{row['error']}"
                )

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\n" + "=" * 80)
    print("Agent 4 writing evaluation complete")
    print("=" * 80)
    print(f"Saved to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()