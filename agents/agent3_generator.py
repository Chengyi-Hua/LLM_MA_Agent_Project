"""
agents/agent3_generator.py

Agent 3: Context-Aware Generation Agent (Step C).
Inherits from HierarchicalRAG (which inherits from BaseRAG).

Reuses from inheritance chain:
  BaseRAG         → _parse_input(), _get_all_chunks(), _rerank_chunks(),
                    _format_chunks_for_prompt(), _call_llm(), _build_output()
  HierarchicalRAG → _generate_section() skeleton, SYSTEM_PROMPT, SECTION_PROMPT

New logic added here:
  - generate()                  : DAG-ordered loop with context injection
  - _generate_section()         : override — uses context prompt when deps exist
  - _build_dependency_context() : formats dep summaries into hard-context block
  - SECTION_WITH_CONTEXT_PROMPT : extended prompt template for sections with deps

Note on summaries:
  Agent 2 (GraphAwareRAG) already generates a quick summary per section
  as part of its NLI pipeline. Agent 3 receives these via agent2_output["summaries"]
  and uses them directly as injection context — no re-summarization needed.
"""

from methods.hierarchical_rag import HierarchicalRAG, SYSTEM_PROMPT, SECTION_PROMPT


# ── New prompt for sections that have declared dependencies ────────────────────

SECTION_WITH_CONTEXT_PROMPT = """\
I have a topic "{island_name}" and a section "{section}" that contains the following documents:

{documents}

Additionally, here is mandatory context from sections that this section logically depends on:

{context}

Based on the above documents and dependency context, write the "{section}" section \
of a Wikipedia article about {island_name}.
Ensure logical consistency with the dependency context and do NOT repeat \
information already covered there.
You MUST cite the most relevant document for every sentence you write.
Use ONLY this exact citation format: "This is an example sentence.[1]" or "Another sentence.[2][3]"
where the number inside the brackets refers to the Document number above.
Do NOT repeat the section title inside the content — just write the content directly.
Do NOT write a "References", "See also", or "External links" section.
"""


class Agent3Generator(HierarchicalRAG):
    """
    Agent 3: Context-Aware Generation Agent.

    Key difference from HierarchicalRAG (Method 2):
    - Follows DAG execution order from Agent 2 (not Eden's original section order)
    - Injects Agent 2's pre-built summaries of dependency sections as hard context
    - Only declared dependencies (from dependency_map) are injected — not all prior sections
    """

    agent_key = "method3"

    def generate(self, input_data: dict) -> dict:
        """
        Parameters
        ----------
        input_data : {
            "blueprint_data": { "island_name": str, "sections_data": { ... } },
            "agent2_output" : {
                "status"    : "success",
                "order"     : list[str],          # DAG topological order
                "dependency": dict[str, list[str]], # section → list of dependency sections
                "summaries" : dict[str, str]       # section → pre-built summary from Agent 2
            }
        }

        Returns
        -------
        Standard output dict for Agent 4:
        { "method", "island_name", "metadata", "generated_article", "sections" }
        """
        # Use BaseRAG's _parse_input — same as all other methods
        island_name, sections_data = self._parse_input(input_data)
        all_chunks = self._get_all_chunks(sections_data)  # BaseRAG

        agent2_output  = input_data["agent2_output"]
        execution_order = agent2_output["order"]        # Agent 2 key: "order"
        dependency_map  = agent2_output["dependency"]   # Agent 2 key: "dependency"
        dep_summaries   = agent2_output["summaries"]    # Agent 2 pre-built summaries — reuse directly

        full_article_parts = []
        all_used_chunks    = []

        for section_name in execution_order:

            # Build hard context from declared dependencies only
            declared_deps = dependency_map.get(section_name, [])
            context_block = self._build_dependency_context(declared_deps, dep_summaries)

            # Log what is being injected
            if declared_deps:
                injected = [d for d in declared_deps if d in dep_summaries]
                missing  = [d for d in declared_deps if d not in dep_summaries]
                print(f"  [{section_name}] injecting context from: {injected}")
                if missing:
                    print(f"  [{section_name}] WARNING — deps missing from Agent 2 summaries: {missing}")
            else:
                print(f"  [{section_name}] no dependencies — generating independently")

            # Get this section's chunks
            # Respect rerank_scope from settings.yaml (global or per-section)
            if self.rerank_scope == "global":
                chunks_to_rerank = all_chunks
            else:
                chunks_to_rerank = sections_data.get(section_name, {}).get("chunks", all_chunks)

            # Generate section (override below picks prompt based on context)
            top_chunks, section_text = self._generate_section(
                island_name=island_name,
                section=section_name,
                section_chunks=chunks_to_rerank,
                context=context_block
            )

            full_article_parts.append(f"=={section_name}==\n{section_text}")
            all_used_chunks.extend(top_chunks)

        article_text = "\n\n".join(full_article_parts)

        return self._build_output(
            method="method3",
            island_name=island_name,
            article_text=article_text,
            chunks=all_used_chunks,
            rerank_strategy=self.rerank_scope,
            top_l_applied_at=self.rerank_scope if self.use_top_l else "none"
        )

    # ── Override: _generate_section ───────────────────────────────────────────

    def _generate_section(
        self,
        island_name: str,
        section: str,
        section_chunks: list,
        context: str = ""
    ) -> tuple:
        """
        Override of HierarchicalRAG._generate_section.

        No context  → uses base SECTION_PROMPT         (identical to Method 2)
        With context → uses SECTION_WITH_CONTEXT_PROMPT (injects dep summaries)

        Reuses from BaseRAG:
          _rerank_chunks()            — BM25 or cross-encoder chunk selection
          _format_chunks_for_prompt() — numbered Document 1, 2, 3... formatting
          _call_llm()                 — single LLM call entry point
        """
        query      = f"{island_name} {section}"
        top_chunks = self._rerank_chunks(section_chunks, query=query)   # BaseRAG
        documents  = self._format_chunks_for_prompt(top_chunks)         # BaseRAG

        if context:
            prompt = SECTION_WITH_CONTEXT_PROMPT.format(
                island_name=island_name,
                section=section,
                documents=documents,
                context=context
            )
        else:
            # No dependencies — identical to HierarchicalRAG
            prompt = SECTION_PROMPT.format(
                island_name=island_name,
                section=section,
                documents=documents
            )

        section_text = self._call_llm(prompt, system=SYSTEM_PROMPT)  # BaseRAG
        return top_chunks, section_text

    # ── New helper ────────────────────────────────────────────────────────────

    def _build_dependency_context(
        self,
        declared_deps: list,
        dep_summaries: dict
    ) -> str:
        """
        Format Agent 2's pre-built summaries of dependency sections
        into a hard-context block for prompt injection.

        Only includes declared deps that have a summary available.
        Returns empty string if no deps → caller uses base SECTION_PROMPT.
        """
        if not declared_deps:
            return ""

        available = {
            dep: dep_summaries[dep]
            for dep in declared_deps
            if dep in dep_summaries
        }
        if not available:
            return ""

        lines = ["--- Dependency Context (do not repeat this information) ---"]
        for dep, summary in available.items():
            lines.append(f"\n[{dep}]\n{summary}")
        lines.append("\n--- End of Dependency Context ---")
        return "\n".join(lines)