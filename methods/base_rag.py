"""
BaseRAG: shared logic for all three methods.
Handles LLM initialization, model switching, reranking, and output formatting.
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
    Each subclass sets agent_key to load the correct model from config.
    """

    agent_key: str = "method1"
    reranker_type: str = "bm25"  # "bm25", "cross-encoder", or "none"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.llm_client, self.model_name = self._init_llm()
        # only read from config if not already overridden by subclass
        if not hasattr(self.__class__, 'use_top_l'):
            self.use_top_l = self.config["methods"]["use_top_l"]
            self.top_l = self.config["methods"]["top_l"]
        else:
            self.top_l = self.config["methods"]["top_l"]
        self.rerank_scope = self.config["methods"].get("rerank_scope", "per-section")
        self.reranker_type = self.config["methods"].get("reranker_type", "bm25")

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
    # Input parsing
    # ------------------------------------------------------------------

    def _parse_input(self, input_data: dict) -> tuple[str, dict[str, list[dict]]]:
        """
        Parse input format into (island_name, sections_data).
        sections_data: { "Geology": {"chunks": [...]}, ... }
        """
        island_name = input_data["blueprint_data"]["island_name"]
        sections_data = input_data["blueprint_data"]["sections_data"]
        return island_name, sections_data

    def _get_all_chunks(self, sections_data: dict) -> list[dict]:
        """Flatten all chunks from all sections into a single list."""
        all_chunks = []
        for section_info in sections_data.values():
            all_chunks.extend(section_info["chunks"])
        return all_chunks

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank_chunks(self, chunks: list[dict], query: str) -> list[dict]:
        """
        Rerank chunks by relevance to query.
        Applies top_l truncation if use_top_l is True.
        """
        if self.reranker_type == "none" or not chunks:
            return chunks

        if self.reranker_type == "bm25":
            ranked = self._rerank_bm25(chunks, query)
        elif self.reranker_type == "cross-encoder":
            ranked = self._rerank_cross_encoder(chunks, query)
        else:
            raise ValueError(f"Unknown reranker: {self.reranker_type}")

        if self.use_top_l:
            return ranked[:self.top_l]
        return ranked

    def _rerank_bm25(self, chunks: list[dict], query: str) -> list[dict]:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]

    def _rerank_cross_encoder(self, chunks: list[dict], query: str) -> list[dict]:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, c["text"]] for c in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]

    def _format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        """Format chunks into numbered documents for LLM prompt."""
        return "\n\n".join(f"Document {i}: {c['text']}" for i, c in enumerate(chunks, 1))

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _parse_article_to_sections(self, article_text: str, chunks: list[dict]) -> list[dict]:
        """Parse ==SECTION NAME== formatted article into structured sections.
        Ignores ===sub-sections=== by merging their content into the parent section.
        """
        import re
        sections = []
        # Split only on top-level ==Section== (not ===Sub-section===)
        parts = re.split(r"(?<!=)==(?!=)(.+?)(?<!=)==(?!=)", article_text)

        i = 1
        while i < len(parts) - 1:
            section_name = parts[i].strip()
            content = parts[i + 1].strip()

            # Remove any ===sub-section=== markers from content
            content = re.sub(r"===.+?===", "", content).strip()

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

    def _build_output(
        self,
        method: str,
        island_name: str,
        article_text: str,
        chunks: list[dict],
        rerank_strategy: str = "none",
        top_l_applied_at: str = "none"
    ) -> dict:
        """Standard output format consumed by Chengyi's evaluator."""
        return {
            "method": method,
            "island_name": island_name,
            "metadata": {
                "reranker": self.reranker_type,
                "rerank_strategy": rerank_strategy,
                "use_top_l": self.use_top_l,
                "top_l": self.top_l if self.use_top_l else None,
                "top_l_applied_at": top_l_applied_at  # "none", "global", "per-section"
            },
            "generated_article": article_text,
            "sections": self._parse_article_to_sections(article_text, chunks)
        }

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def generate(self, input_data: dict) -> dict:
        raise NotImplementedError("Subclasses must implement generate()")