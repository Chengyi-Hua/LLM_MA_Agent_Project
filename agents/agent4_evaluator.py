"""
agents/agent4_evaluator.py

Agent 4: LLM-as-a-Judge Writing Evaluator.

This follows the project evaluation rubric and overlaps with WikiGenBench's
writing evaluation.

Project Writing metrics:
  - fluency_score       (0–5)
  - structure_score     (0–5)
  - organization_score  (0–5)
  - writing_score       average of the three

Mapping to WikiGenBench:
  - fluency_score      ≈ Fluency Score
  - structure_score    ≈ Outline Score / section heading and article structure quality
  - organization_score ≈ Organization Score

This agent does NOT compute:
  - ROUGE-L
  - METEOR
  - citation recall
  - citation precision
  - citation rate
  - CSCS

Those are implemented separately.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from methods.base_rag import load_config


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "evaluations")


AGENT4_SYSTEM_PROMPT = """
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


class Agent4Evaluator:
    """
    Agent 4: LLM-as-a-Judge evaluator for the Writing dimension.
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

    def _call_judge(self, prompt: str, retries: int = 2) -> str:
        last_error = None

        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": AGENT4_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                content = response.choices[0].message.content

                if content is None or not str(content).strip():
                    raise ValueError(f"Judge returned empty content. Raw response: {response}")

                return content.strip()

            except Exception as e:
                last_error = e
                print(f"⚠️ Agent 4 judge call failed on attempt {attempt + 1}: {e}")

                if attempt < retries:
                    time.sleep(1.5)

        raise RuntimeError(f"Agent 4 judge failed after {retries + 1} attempts: {last_error}")

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

    def _normalize_scores(self, data: dict) -> dict:
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

    def evaluate_article(
        self,
        article: str,
        island_name: str,
        method_name: str,
    ) -> dict:
        if article is None or not article.strip():
            raise ValueError("Generated article is empty.")

        prompt = AGENT4_WRITING_PROMPT.format(
            island_name=island_name,
            method_name=method_name,
            article=article,
        )

        raw_response = self._call_judge(prompt)
        parsed = self._extract_json(raw_response)
        scores = self._normalize_scores(parsed)

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
                    f"✅ {row['island_name']} | {row['method']} | "
                    f"writing={row['writing_score']} "
                    f"fluency={row['fluency_score']} "
                    f"structure={row['structure_score']} "
                    f"organization={row['organization_score']}"
                )
            else:
                print(
                    f"❌ {row['island_name']} | {row['method']} | "
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