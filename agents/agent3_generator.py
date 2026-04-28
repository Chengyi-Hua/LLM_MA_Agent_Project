"""
agents/agent3_generator.py

Agent 3: Context-Aware Generation Agent.
Inherits from HierarchicalRAG (which inherits from BaseRAG).

What it reuses from the inheritance chain:
  BaseRAG         → _call_llm(), _rerank_chunks(), _format_chunks_for_prompt(), _build_output()
  HierarchicalRAG → _generate_section() skeleton, SYSTEM_PROMPT, SECTION_PROMPT

What it adds (new logic only):
  - generate()                  : DAG-ordered generation loop with context injection
  - _generate_section()         : override to inject dependency context when available
  - _summarize_section()        : compress generated section for downstream injection
  - _build_dependency_context() : format dependency summaries into hard-context block
  - SECTION_WITH_CONTEXT_PROMPT : new prompt template extending SECTION_PROMPT
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
You must cite the most relevant document for every sentence you write, \
in the format "This is an example sentence.[k]", where k denotes Document k.
"""

SUMMARIZE_PROMPT = """\
Summarize the following "{section}" section in 3-5 concise, fact-dense sentences.
Keep only information that other sections might logically depend on.

{content}
"""


# ── Agent 3 ───────────────────────────────────────────────────────────────────

class Agent3Generator(HierarchicalRAG):
    """
    Agent 3: Context-Aware Generation Agent (Step C).

    Inherits from HierarchicalRAG. The key difference from Method 2:
    - Follows DAG execution order from Agent 2 (not just Eden's section list order)
    - Injects summaries of declared dependency sections as hard context into each prompt
    - Summarizes each generated section immediately for downstream injection

    Called by InterSectionRAG.generate() after Agent 2 has run.
    """

    agent_key = "method3"

    def generate(self, input_data: dict) -> dict:
        """
        Parameters
        ----------
        input_data : {
            "island_name"   : str,
            "chunks"        : list[dict],  # flat list of all Tavily chunks
            "sections"      : list[str],   # fallback section order (from Eden)
            "agent2_output" : dict         # from Agent 2: execution_order + dependency_map
        }

        Returns
        -------
        Standard output dict consumed by Agent 4:
        { "method", "island_name", "generated_article", "sections" }
        """
        island_name   = input_data["island_name"]
        chunks        = input_data["chunks"]
        agent2_output = input_data["agent2_output"]

        execution_order = agent2_output["execution_order"]
        dependency_map  = agent2_output["dependency_map"]

        full_article_parts = []
        all_used_chunks    = []
        section_summaries  = {}  # section_name → short summary for downstream injection

        for section_name in execution_order:

            # Build hard context from declared dependencies only
            declared_deps = dependency_map.get(section_name, [])
            context_block = self._build_dependency_context(declared_deps, section_summaries)

            # Log what is being injected
            if declared_deps:
                injected = [d for d in declared_deps if d in section_summaries]
                missing  = [d for d in declared_deps if d not in section_summaries]
                print(f"  [{section_name}] injecting context from: {injected}")
                if missing:
                    print(f"  [{section_name}] WARNING — deps not yet generated: {missing}")

            # Generate (override below picks the right prompt based on context)
            section_chunks, section_text = self._generate_section(
                island_name=island_name,
                section=section_name,
                all_chunks=chunks,
                context=context_block
            )

            full_article_parts.append(f"=={section_name}==\n{section_text}")
            all_used_chunks.extend(section_chunks)

            # Summarize immediately so downstream sections can use it
            section_summaries[section_name] = self._summarize_section(
                section_name, section_text
            )

        article_text = "\n\n".join(full_article_parts)

        # _build_output() from BaseRAG — produces standard dict for Agent 4
        return self._build_output(
            method="method3",
            island_name=island_name,
            article_text=article_text,
            chunks=all_used_chunks
        )

    # ── Override: _generate_section ───────────────────────────────────────────

    def _generate_section(
        self,
        island_name: str,
        section: str,
        all_chunks: list,
        context: str = ""
    ) -> tuple:
        """
        Override of HierarchicalRAG._generate_section.

        No context  → uses base SECTION_PROMPT         (identical to Method 2 behaviour)
        With context → uses SECTION_WITH_CONTEXT_PROMPT (injects dependency summaries)

        Reuses from BaseRAG:
          _rerank_chunks()           — BM25 chunk selection
          _format_chunks_for_prompt() — numbered Document 1, 2, 3... formatting
          _call_llm()                — single LLM call entry point
        """
        query          = f"{island_name} {section}"
        section_chunks = self._rerank_chunks(all_chunks, query=query)    # BaseRAG
        documents      = self._format_chunks_for_prompt(section_chunks)  # BaseRAG

        if context:
            prompt = SECTION_WITH_CONTEXT_PROMPT.format(
                island_name=island_name,
                section=section,
                documents=documents,
                context=context
            )
        else:
            # No dependencies — fall back to HierarchicalRAG's prompt exactly
            prompt = SECTION_PROMPT.format(
                island_name=island_name,
                section=section,
                documents=documents
            )

        section_text = self._call_llm(prompt, system=SYSTEM_PROMPT)  # BaseRAG
        return section_chunks, section_text

    # ── New helpers ───────────────────────────────────────────────────────────

    def _summarize_section(self, section_name: str, content: str) -> str:
        """
        Compress a generated section into 3-5 sentences for downstream injection.
        Uses _call_llm() from BaseRAG — no raw API calls.
        """
        prompt = SUMMARIZE_PROMPT.format(section=section_name, content=content)
        return self._call_llm(
            prompt,
            system="You are a precise summarizer. Return only the summary, no preamble."
        )

    def _build_dependency_context(
        self,
        declared_deps: list,
        section_summaries: dict
    ) -> str:
        """
        Format dependency summaries into a hard-context block for prompt injection.
        Only includes deps that have already been generated (guards against ordering issues).
        Returns empty string if no dependencies — caller then uses base SECTION_PROMPT.
        """
        if not declared_deps:
            return ""

        available = {
            dep: section_summaries[dep]
            for dep in declared_deps
            if dep in section_summaries
        }
        if not available:
            return ""

        lines = ["--- Dependency Context (do not repeat this information) ---"]
        for dep, summary in available.items():
            lines.append(f"\n[{dep}]\n{summary}")
        lines.append("\n--- End of Dependency Context ---")
        return "\n".join(lines)