"""
Method 1: Naive RAG — Retrieve-then-Read (RR).
Merges all chunks from all sections, reranks globally,
then feeds top-L chunks into a single prompt for full article generation.
Based on Zhang et al. (2024) Section 4, RR framework.
"""

from methods.base_rag import BaseRAG


SYSTEM_PROMPT = (
    "You are an expert encyclopedia writer. "
    "Write accurate, well-structured Wikipedia-style articles with citations."
)

ARTICLE_PROMPT = """\
I have a topic "{island_name}" that contains the following documents:

{documents}

Based on the above information, write a Wikipedia article about this island.
Start directly with the first section, do not write a lead paragraph.
Organize the content by sections. Before writing each section, always start with "==SECTION NAME==".
You MUST cite the most relevant document for every sentence you write.
Use ONLY this exact citation format: "This is an example sentence.[1]" or "Another sentence.[2][3]"
where the number inside the brackets refers to the Document number above.
Do NOT write a "References", "See also", or "External links" section at the end — only inline citations like [1], [2], [3].
"""


class NaiveRAG(BaseRAG):
    agent_key = "method1"

    def generate(self, input_data: dict) -> dict:
        island_name, sections_data = self._parse_input(input_data)

        # Merge all chunks from all sections into one list
        all_chunks = self._get_all_chunks(sections_data)

        # Rerank globally against island name
        top_chunks = self._rerank_chunks(all_chunks, query=island_name)

        documents = self._format_chunks_for_prompt(top_chunks)
        prompt = ARTICLE_PROMPT.format(island_name=island_name, documents=documents)
        article_text = self._call_llm(prompt, system=SYSTEM_PROMPT)

        return self._build_output(
            method="method1",
            island_name=island_name,
            article_text=article_text,
            chunks=top_chunks,
            rerank_strategy="global"
        )