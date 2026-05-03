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
import os
import json
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


        # ── Graph metrics evaluation ───────────────────────────────────────────
        plan_path = os.path.join("logs", "agent2_plans", f"{island_name}_plan.json")
        if os.path.exists(plan_path):
            try:
                graph_report = self._evaluate_graph_metrics(plan_path, island_name)
                print(f"[Method 3] Graph metrics: "
                      f"depth={graph_report['graph_analysis']['max_graph_depth']}, "
                      f"orphan_ratio={graph_report['graph_analysis']['orphan_node_ratio']}")
            except Exception as e:
                print(f"[Method 3] Graph metrics failed (non-critical): {e}")
        else:
            print(f"[Method 3] Skipping graph metrics — plan file not found at {plan_path}")


        # ── Agent 3: context-aware generation in DAG order ────────────────────
        print("\n[Method 3] Running Agent 3 (context-aware generation) ...")
        agent3 = Agent3Generator(config=self.config)

        # Inject agent2_output into input_data for Agent 3
        agent3_input = {**input_data, "agent2_output": agent2_output}
        result = agent3.generate(agent3_input)

        print(f"\n[Method 3] Complete. Sections generated: {len(result['sections'])}")
        return result
    def _evaluate_graph_metrics(self, blueprint_path: str, island_name: str) -> dict:
        """
        Reads the Agent 2 plan JSON and computes:
          - Orphan node ratio: sections with no dependencies
          - Max graph depth: length of the longest dependency chain
 
        Saves report to logs/agent2_plans/<island>_graph_metrics.json
        """
        with open(blueprint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
 
        dependency  = data.get("dependency", {})
        nodes       = list(dependency.keys())
        total_nodes = len(nodes)
 
        if total_nodes == 0:
            return {"error": "No node data found"}
 
        # Metric 1: orphan node ratio (sections with no dependencies)
        orphan_nodes = [node for node, deps in dependency.items() if len(deps) == 0]
        orphan_ratio = len(orphan_nodes) / total_nodes
 
        # Metric 2: max graph depth (longest dependency chain)
        memo = {}
        def get_depth(node):
            if node in memo:
                return memo[node]
            deps = dependency.get(node, [])
            if not deps:
                memo[node] = 1
                return 1
            memo[node] = max(get_depth(d) for d in deps) + 1
            return memo[node]
 
        max_depth = max(get_depth(n) for n in nodes) if nodes else 0
 
        report = {
            "graph_analysis": {
                "total_sections"    : total_nodes,
                "orphan_nodes_count": len(orphan_nodes),
                "orphan_nodes_list" : orphan_nodes,
                "orphan_node_ratio" : f"{orphan_ratio:.2%}",
                "max_graph_depth"   : max_depth
            }
        }
 
        # Save to logs/agent2_plans/ alongside the plan file
        save_path = os.path.join(
            "logs", "agent2_plans", f"{island_name}_graph_metrics.json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"💾 [Method 3] Graph metrics saved to: {save_path}")
 
        return report