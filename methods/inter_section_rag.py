"""
methods/inter_section_rag.py

Method 3: Inter-Section Aware RAG.
Thin coordinator — contains no generation or NLI logic.

Inherits from HierarchicalRAG to satisfy the pipeline interface.
pipeline.py calls .generate(input_data) identically on all methods.

Coordination flow:
  1. Agent 2 (GraphAwareRAG)  — reranks chunks, builds NLI summaries, constructs
                                DAG, returns execution order + dependency map
  2. Agent 3 (Agent3Generator) — generates each section in DAG order,
                                 injecting Agent 2's summaries as hard context

input_data format (standard across all methods, from Eden + mock_data.py):
{
    "metadata"       : { ... },
    "blueprint_data" : {
        "island_name"  : str,
        "sections_data": {
            "Geology": { "chunks": [ { "chunk_id", "text", "source_url", ... } ] },
            ...
        }
    }
}
"""

from methods.hierarchical_rag import HierarchicalRAG
from agents.agent2_orchestrator import GraphAwareRAG   # Agent 2's actual class name
from agents.agent3_generator import Agent3Generator


class InterSectionRAG(HierarchicalRAG):
    """
    Method 3: Inter-Section Aware RAG.
    Coordinates Agent 2 → Agent 3 and returns standard output for Agent 4.
    """

    agent_key = "method3"

    def generate(self, input_data: dict) -> dict:
        """
        Parameters
        ----------
        input_data : standard blueprint dict (same format as Methods 0-2)

        Returns
        -------
        Standard output dict consumed by Agent 4:
        { "method", "island_name", "metadata", "generated_article", "sections" }
        """
        island_name = input_data["blueprint_data"]["island_name"]

        print(f"\n{'='*60}")
        print(f"[Method 3] Inter-Section RAG | island: {island_name}")
        print(f"{'='*60}")

        # ── Agent 2: NLI + DAG + topological sort ─────────────────────────────
        # GraphAwareRAG inherits BaseRAG — passes same config, no double init cost
        print("\n[Method 3] Running Agent 2 (NLI + DAG) ...")
        agent2 = GraphAwareRAG(config=self.config)
        agent2_output = agent2.generate(input_data)
        # agent2_output = {
        #     "status"    : "success",
        #     "order"     : list[str],            # DAG topological execution order
        #     "dependency": dict[str, list[str]], # section → list of dep sections
        #     "summaries" : dict[str, str]        # section → pre-built NLI summary
        # }
        print(f"[Method 3] Agent 2 done. Execution order: {agent2_output['order']}")

        # ── Agent 3: context-aware generation in DAG order ────────────────────
        print("\n[Method 3] Running Agent 3 (context-aware generation) ...")
        agent3 = Agent3Generator(config=self.config)

        # Inject agent2_output into input_data for Agent 3
        agent3_input = {**input_data, "agent2_output": agent2_output}
        result = agent3.generate(agent3_input)

        print(f"\n[Method 3] Complete. Sections generated: {len(result['sections'])}")
        return result