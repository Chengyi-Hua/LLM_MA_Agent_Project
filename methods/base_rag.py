"""
BaseRAG: shared logic for all three methods.
Handles LLM initialization, model switching, and basic retrieval utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BaseRAG:
    """
    Shared base for PureGeneration, NaiveRAG, HierarchicalRAG.
    Each subclass passes its agent_key to load the correct model from config.
    """

    agent_key: str = "method1"  # subclasses override this

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.llm_client, self.model_name = self._init_llm()

    # ------------------------------------------------------------------
    # LLM setup
    # ------------------------------------------------------------------

    def _init_llm(self):
        agent_config = self.config["llm"][self.agent_key]
        provider = agent_config["provider"]
        model = agent_config["model"]

        if provider == "openai":
            from openai import OpenAI
            api_key = os.getenv(self.config["api_keys"]["openai_env"])
            return OpenAI(api_key=api_key), model

        elif provider == "groq":
            from groq import Groq
            api_key = os.getenv(self.config["api_keys"]["groq_env"])
            return Groq(api_key=api_key), model

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'groq'.")

    def _call_llm(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        """Single entry point for all LLM calls."""
        print(f"\n--- PROMPT ---\n{prompt}\n--- END ---\n")
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            temperature=self.config["llm"].get("temperature", 0.0),
            max_tokens=self.config["llm"].get("max_tokens", 4096),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Retrieval utilities
    # ------------------------------------------------------------------

    def _rerank_chunks(self, chunks: list[dict], query: str, top_l: Optional[int] = None) -> list[dict]:
        """
        BM25 reranker.
        TODO: swap to DPR if needed — just replace this method.
        """
        from rank_bm25 import BM25Okapi

        top_l = top_l or self.config["retrieval"]["top_l"]
        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_l]]

    def _format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        """Format chunks into numbered documents for LLM prompt."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"Document {i}: {chunk['text']}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _parse_article_to_sections(self, article_text: str, chunks: list[dict]) -> list[dict]:
        """Parse ==SECTION NAME== formatted article into structured sections."""
        import re
        sections = []
        parts = re.split(r"==(.+?)==", article_text)

        i = 1
        while i < len(parts) - 1:
            section_name = parts[i].strip()
            content = parts[i + 1].strip()

            cited_indices = [int(x) - 1 for x in re.findall(r"\[(\d+)\]", content)]
            citations = []
            for idx in cited_indices:
                if 0 <= idx < len(chunks):
                    url = chunks[idx].get("source_url", "")
                    if url and url not in citations:
                        citations.append(url)

            sections.append({
                "section_name": section_name,
                "content": content,
                "citations": citations
            })
            i += 2

        return sections

    def _build_output(self, method: str, island_name: str, article_text: str, chunks: list[dict]) -> dict:
        """Standard output format consumed by Chengyi's evaluator."""
        return {
            "method": method,
            "island_name": island_name,
            "generated_article": article_text,
            "sections": self._parse_article_to_sections(article_text, chunks)
        }

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def generate(self, input_data: dict) -> dict:
        raise NotImplementedError("Subclasses must implement generate()")