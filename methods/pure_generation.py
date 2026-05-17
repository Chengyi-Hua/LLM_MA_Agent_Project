"""
Method 0: Pure Generation (no RAG).
Generates a Wikipedia-style article using only the island name.
Serves as the baseline to quantify the contribution of RAG.
"""

from methods.base_rag import BaseRAG


SYSTEM_PROMPT = (
    "You are an expert encyclopedia writer. "
    "Write accurate, well-structured Wikipedia-style articles."
)

ARTICLE_PROMPT = """\
Write a full Wikipedia article about the following island: {island_name}

Requirements:
- Start directly with the first section, do not write a lead paragraph
- Organize the article into sections
- Start each section with ==SECTION NAME==
- Do NOT use sub-sections (===Sub-section===) — use only top-level sections
- Do NOT include a "References", "See also", or "External links" section at the end
- Write in a neutral, encyclopedic tone
- Cover geography, geology, ecology, climate, and history where relevant
"""


class PureGeneration(BaseRAG):
    agent_key = "method0"
    use_top_l = False  # no chunks, top_l irrelevant

    def __init__(self, config=None):
        super().__init__(config)
        self.reranker_type = "none"  # override yaml setting — method0 has no chunks

    def generate(self, input_data: dict) -> dict:
        island_name, _ = self._parse_input(input_data)

        prompt = ARTICLE_PROMPT.format(island_name=island_name)
        article_text = self._call_llm(prompt, system=SYSTEM_PROMPT)

        return self._build_output(
            method="method0",
            island_name=island_name,
            article_text=article_text,
            chunks=[],
            rerank_strategy="none",
            top_l_applied_at="none"
        )