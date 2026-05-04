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


import json
import os
import re
import sys
import time
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from methods.base_rag import load_config


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

Do not explain step by step.
Do not include reasoning.
Return only the final JSON object.
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
  "brief_rationale": "one concise sentence explaining the scores"
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

Do not explain step by step.
Do not include reasoning.
Return only the final JSON object.
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

    Public methods:
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

        self.max_tokens = self.agent_config.get("max_tokens", 700)

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

        head_len = int(max_chars * 0.65)
        tail_len = int(max_chars * 0.35)

        head = text[:head_len]
        tail = text[-tail_len:]

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

        This version avoids provider-specific reasoning options because some
        endpoints reject them, while others require reasoning. The prompts
        already instruct the model to return only final JSON.
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

                response = self.client.chat.completions.create(**request_kwargs)

                choices = getattr(response, "choices", None)

                if not choices:
                    raise ValueError(f"Judge returned no choices. Raw response: {response}")

                choice = choices[0]
                message = getattr(choice, "message", None)
                finish_reason = getattr(choice, "finish_reason", "")

                if message is None:
                    raise ValueError(
                        f"Judge returned no message. finish_reason={finish_reason}. "
                        f"Raw response: {response}"
                    )

                content = getattr(message, "content", None)

                if content is None or not str(content).strip():
                    raise ValueError(
                        f"Judge returned empty content. finish_reason={finish_reason}"
                    )

                return str(content).strip()

            except Exception as e:
                last_error = e
                wait_seconds = min(3 * attempt, 30)

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

        text = str(text).strip()

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
    
    def _call_parse_normalize(
        self,
        prompt: str,
        system_prompt: str,
        normalizer,
        retries: int = 5,
    ):
        """
        Calls the judge, parses JSON, validates all required scores,
        and retries if the response is invalid or incomplete.
        """
        last_error = None
        raw_response = ""

        for attempt in range(1, retries + 1):
            try:
                raw_response = self._call_judge(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    retries=1,
                )

                parsed = self._extract_json(raw_response)
                scores = normalizer(parsed)

                return raw_response, scores

            except Exception as e:
                last_error = e
                wait_seconds = min(2 * attempt, 15)

                print(
                    f"! Agent 4 JSON/score validation failed on attempt "
                    f"{attempt}/{retries}: {e}"
                )
                print(f"   Waiting {wait_seconds}s before retrying...")

                time.sleep(wait_seconds)

        raise RuntimeError(
            f"Agent 4 failed to produce complete valid scores after "
            f"{retries} attempts: {last_error}"
        )

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

        if None in [fluency, structure, organization]:
            raise ValueError(f"Writing judge returned incomplete scores: {data}")

        writing_score = round((fluency + structure + organization) / 3, 2)

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

        if None in [coverage, accuracy, relevance, organization]:
            raise ValueError(f"Concept judge returned incomplete scores: {data}")

        concept_score = round(
            (coverage + accuracy + relevance + organization) / 4,
            2,
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
        Writing evaluator used by full_evaluation.py.
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

        raw_response, scores = self._call_parse_normalize(
            prompt=prompt,
            system_prompt=AGENT4_WRITING_SYSTEM_PROMPT,
            normalizer=self._normalize_writing_scores,
        )

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

        raw_response, scores = self._call_parse_normalize(
            prompt=prompt,
            system_prompt=AGENT4_CONCEPT_SYSTEM_PROMPT,
            normalizer=self._normalize_concept_scores,
        )

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
