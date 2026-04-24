"""
Method 2: Hierarchical RAG — Plan-Retrieve-Read (PRR).
- Plan: sections provided by Eden (from Wikipedia or fallback defaults)
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
You must cite the most relevant document for every sentence you write, \
in the format "This is an example sentence.[k]", where k denotes Document k.
"""


class HierarchicalRAG(BaseRAG):
    agent_key = "method2"
    """
    Method 2: Plan-Retrieve-Read.
    Sections (outline) come from Eden's input.
    Each section is generated independently with its own retrieved chunks.
    Designed to be inherited by Method 3 (InterSectionRAG).
    """

    def generate(self, input_data: dict) -> dict:
        island_name = input_data["island_name"]
        chunks = input_data["chunks"]
        sections = input_data["sections"]  # provided by Eden (from Wikipedia or defaults)

        full_article_parts = []
        all_used_chunks = []

        for section in sections:
            section_chunks, section_text = self._generate_section(
                island_name=island_name,
                section=section,
                all_chunks=chunks
            )
            full_article_parts.append(f"=={section}==\n{section_text}")
            all_used_chunks.extend(section_chunks)

        article_text = "\n\n".join(full_article_parts)

        return self._build_output(
            method="method2",
            island_name=island_name,
            article_text=article_text,
            chunks=all_used_chunks
        )

    def _generate_section(
        self,
        island_name: str,
        section: str,
        all_chunks: list[dict],
        context: str = ""  # reserved for Method 3 to inject summaries from previous sections
    ) -> tuple[list[dict], str]:
        """
        Retrieve top-L chunks relevant to this section, then generate section content.
        Returns (used_chunks, section_text).
        The context parameter is reserved for Method 3 (InterSectionRAG).
        """
        query = f"{island_name} {section}"
        section_chunks = self._rerank_chunks(all_chunks, query=query)
        documents = self._format_chunks_for_prompt(section_chunks)

        prompt = SECTION_PROMPT.format(
            island_name=island_name,
            section=section,
            documents=documents
        )

        section_text = self._call_llm(prompt)
        return section_chunks, section_text