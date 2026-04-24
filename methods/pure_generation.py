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
- Organize the article into sections
- Start each section with ==SECTION NAME==
- Write in a neutral, encyclopedic tone
- Cover geography, geology, ecology, climate, and history where relevant
"""


class PureGeneration(BaseRAG):
    agent_key = "method0"
    """
    Method 0: No retrieval. LLM generates directly from island name only.
    """

    def generate(self, input_data: dict) -> dict:
        island_name = input_data["island_name"]

        prompt = ARTICLE_PROMPT.format(island_name=island_name)

        article_text = self._call_llm(prompt, system=SYSTEM_PROMPT)

        return self._build_output(
            method="method0",
            island_name=island_name,
            article_text=article_text,
            chunks=[]
        )