"""
Method 2: Hierarchical RAG — Plan-Retrieve-Read (PRR).
- Plan: sections provided by Eden
- Retrieve: for each section, reranker finds relevant chunks
- Read: LLM generates each section independently (stateless between sections)
Based on Zhang et al. (2024) Section 4, PRR framework.
"""

from methods.base_rag import BaseRAG


SYSTEM_PROMPT = (
    "You are an expert encyclopedia writer. "
    "Write accurate, well-structured Wikipedia-style articles with citations."
)

SECTION_PROMPT = """\
I have a topic "{island_name}" and a section "{section}" that contains the following documents:

{documents}

Based on the above information, write the "{section}" section of a Wikipedia article \
about {island_name}.
You MUST cite the most relevant document for every sentence you write.
Use ONLY this exact citation format: "This is an example sentence.[1]" or "Another sentence.[2][3]"
where the number inside the brackets refers to the Document number above.
Do NOT repeat the section title inside the content — just write the content directly.
Do NOT write a "References", "See also", or "External links" section — only inline citations like [1], [2], [3].
"""


class HierarchicalRAG(BaseRAG):
    agent_key = "method2"
    def generate(self, input_data: dict) -> dict:
        island_name, sections_data = self._parse_input(input_data)
        all_chunks = self._get_all_chunks(sections_data)

        full_article_parts = []
        all_used_chunks = []
        sections_output = []

        for section, section_info in sections_data.items():
            if self.rerank_scope == "global":
                chunks_to_rerank = all_chunks
                rerank_strategy = "global"
            else:
                chunks_to_rerank = section_info["chunks"]
                rerank_strategy = "per-section"

            top_chunks, section_text = self._generate_section(
                island_name=island_name,
                section=section,
                section_chunks=chunks_to_rerank
            )

            # 用這個 section 自己的 top_chunks 立刻解析 citation
            parsed = self._parse_article_to_sections(f"=={section}==\n{section_text}", top_chunks)
            sections_output.extend(parsed)

            full_article_parts.append(f"=={section}==\n{section_text}")
            all_used_chunks.extend(top_chunks)

        article_text = "\n\n".join(full_article_parts)

        return {
            "method": "method2",
            "island_name": island_name,
            "metadata": {
                "reranker": self.reranker_type,
                "rerank_strategy": rerank_strategy,
                "top_l": self.top_l,
            },
            "generated_article": article_text,
            "sections": sections_output
        }

    def _generate_section(
        self,
        island_name: str,
        section: str,
        section_chunks: list[dict],
        context: str = ""  # reserved for Method 3 to inject summaries from previous sections
    ) -> tuple[list[dict], str]:
        """
        Rerank chunks for this section, then generate section content.
        Returns (used_chunks, section_text).
        The context parameter is reserved for Method 3 (InterSectionRAG).
        """
        query = f"{island_name} {section}"
        top_chunks = self._rerank_chunks(section_chunks, query=query)
        documents = self._format_chunks_for_prompt(top_chunks)

        prompt = SECTION_PROMPT.format(
            island_name=island_name,
            section=section,
            documents=documents
        )

        section_text = self._call_llm(prompt)
        return top_chunks, section_text