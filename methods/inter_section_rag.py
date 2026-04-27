"""
Method 3: Inter-Section Aware RAG.
Extends HierarchicalRAG with NLI-based dependency ordering and context injection.
- Agent 2 (fuzheng): NLI, DAG, topological sort → determines generation order
- Agent 3 (Thomas): generates each section with context from previous sections
"""

from methods.hierarchical_rag import HierarchicalRAG, SECTION_PROMPT


class InterSectionRAG(HierarchicalRAG):
    agent_key = "agent3"

    def generate(self, input_data: dict) -> dict:
        island_name, sections_data = self._parse_input(input_data)

        # TODO (fuzheng - Agent 2):
        # 1. Run NLI across section chunks to compute entailment probabilities
        # 2. Build DAG from entailment matrix
        # 3. Topological sort → ordered_sections
        ordered_sections = list(sections_data.keys())  # placeholder

        full_article_parts = []
        all_used_chunks = []
        previous_summaries = {}

        for section in ordered_sections:
            section_chunks = sections_data[section]["chunks"]
            context = self._build_context(previous_summaries)

            top_chunks, section_text = self._generate_section(
                island_name=island_name,
                section=section,
                section_chunks=section_chunks,
                context=context
            )

            previous_summaries[section] = self._summarize(section_text)
            full_article_parts.append(f"=={section}==\n{section_text}")
            all_used_chunks.extend(top_chunks)

        article_text = "\n\n".join(full_article_parts)

        return self._build_output(
            method="method3",
            island_name=island_name,
            article_text=article_text,
            chunks=all_used_chunks,
            rerank_strategy="per-section",
            top_l_applied_at="per-section" if self.use_top_l else "none"
        )

    def _generate_section(
        self,
        island_name: str,
        section: str,
        section_chunks: list[dict],
        context: str = ""
    ) -> tuple[list[dict], str]:
        query = f"{island_name} {section}"
        top_chunks = self._rerank_chunks(section_chunks, query=query)
        documents = self._format_chunks_for_prompt(top_chunks)

        context_block = f"\nContext from previously generated sections:\n{context}\n" if context else ""

        prompt = SECTION_PROMPT.format(
            island_name=island_name,
            section=section,
            documents=documents
        ) + context_block

        section_text = self._call_llm(prompt)
        return top_chunks, section_text

    def _build_context(self, previous_summaries: dict) -> str:
        if not previous_summaries:
            return ""
        return "\n".join(f"[{s}]: {summary}" for s, summary in previous_summaries.items())

    def _summarize(self, section_text: str) -> str:
        prompt = f"Summarize the following section in 2-3 sentences:\n\n{section_text}"
        return self._call_llm(prompt)