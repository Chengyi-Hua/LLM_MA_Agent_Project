"""
agents/agent2_orchestrator.py

Agent 2: Orchestration & Planning Agent (Steps A + B).
Inherits from BaseRAG.

Responsibilities:
  - Rerank chunks per section and generate a quick NLI summary for each
  - Compute asymmetric entailment probabilities across section summaries (Step A)
  - Build a DAG, resolve cycles, perform topological sort (Step B)
  - Return execution order + dependency map + summaries for Agent 3

Output format:
{
    "status"    : "success",
    "order"     : list[str],             # DAG topological execution order
    "dependency": dict[str, list[str]],  # section → list of dependency sections
    "summaries" : dict[str, str]         # section → pre-built NLI summary
}
"""

import os
import networkx as nx
from scipy.special import softmax
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from methods.base_rag import BaseRAG, load_config

load_dotenv()


class GraphAwareRAG(BaseRAG):
    """
    Agent 2: NLI-based dependency graph builder.
    Uses BaseRAG for LLM calls (_call_llm) and chunk reranking (_rerank_chunks).
    Adds NLI cross-encoder and DAG construction on top.
    """

    agent_key = "agent2"   # maps to config["llm"]["agent2"] in settings.yaml

    def __init__(self, config=None):
        super().__init__(config)

        # Load Agent 2 specific graph config from settings.yaml
        graph_config = self.config["llm"]["agent2"]["graph_logic"]
        model_name = graph_config["nli_model"]["model_name"]
        self.nli_threshold = graph_config["algorithm"]["threshold"]

        print(f"⏳ Loading NLI model ({model_name}) ...")
        print(f"⚙️  NLI threshold set to: {self.nli_threshold}")
        self.nli_model = CrossEncoder(model_name)
        print("✅ Agent 2 initialised.")

    # ── Override _init_llm to use agent2's own llm config block ───────────────

    def _init_llm(self):
        """
        Agent 2 has its own LLM config under config["llm"]["agent2"]["llm"],
        separate from the method-level configs.
        """
        agent2_llm = self.config["llm"]["agent2"]["llm"]
        provider   = agent2_llm["provider"]
        model      = agent2_llm["model"]

        if provider == "openai":
            from openai import OpenAI
            api_key = os.getenv(self.config["api_keys"]["openai_env"])
            return OpenAI(api_key=api_key), model

        elif provider == "groq":
            from groq import Groq
            api_key = os.getenv(self.config["api_keys"]["groq_env"])
            return Groq(api_key=api_key), model

        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ── Core methods ──────────────────────────────────────────────────────────

    def _generate_quick_summary(self, section_name: str, reranked_chunks: list) -> str:
        """
        Compress reranked chunks into one dense factual paragraph.
        Used as input to the NLI model for dependency detection.
        Uses _call_llm() from BaseRAG — no raw API calls here.
        """
        if not reranked_chunks:
            return ""

        combined_text = "\n\n".join([c["text"] for c in reranked_chunks])
        prompt = (
            f"Topic: {section_name}\n"
            f"Docs:\n{combined_text}\n"
            f"Summarize into one dense factual paragraph."
        )
        return self._call_llm(prompt, system="You are a technical summarizer.")

    def _build_nli_graph(self, summaries: dict) -> dict:
        """
        Step A: compute asymmetric entailment probabilities across all section pairs.
        Step B: build DAG, resolve cycles, topological sort.

        Returns { "order": list[str], "map": dict[str, list[str]] }
        """
        sections      = list(summaries.keys())
        summary_texts = list(summaries.values())

        dag = nx.DiGraph()
        dag.add_nodes_from(sections)

        # Step A: asymmetric entailment matrix
        for i, sec_a in enumerate(sections):
            for j, sec_b in enumerate(sections):
                if i == j:
                    continue
                logits = self.nli_model.predict([(summary_texts[i], summary_texts[j])])[0]
                # softmax converts raw logits → probabilities summing to 1
                # index 1 = entailment probability (contradiction=0, entailment=1, neutral=2)
                score = softmax(logits)[1]
                if score > self.nli_threshold:
                    dag.add_edge(sec_a, sec_b, weight=score)

        # Step B: resolve cycles by removing weakest edge
        while not nx.is_directed_acyclic_graph(dag):
            cycle   = nx.find_cycle(dag)
            min_edge = min(cycle, key=lambda e: dag.edges[e[0], e[1]]["weight"])
            dag.remove_edge(*min_edge)

        return {
            "order": list(nx.topological_sort(dag)),
            "map"  : {n: list(dag.predecessors(n)) for n in dag.nodes()}
        }

    def generate(self, input_data: dict) -> dict:
        """
        Main entry point. Accepts standard blueprint input_data (same as all methods).

        Returns
        -------
        {
            "status"    : "success",
            "order"     : list[str],
            "dependency": dict[str, list[str]],
            "summaries" : dict[str, str]
        }
        """
        island_name, sections_data = self._parse_input(input_data)  # BaseRAG

        # Phase 1: rerank + summarize each section
        print("\n[Agent 2] Phase 1: reranking and summarizing sections ...")
        summaries = {}
        for section, data in sections_data.items():
            best_chunks          = self._rerank_chunks(data.get("chunks", []), query=section)
            summaries[section]   = self._generate_quick_summary(section, best_chunks)
            print(f"  [{section}] summary ready ({len(summaries[section])} chars)")

        # Phase 2: NLI graph → execution order + dependency map
        print("\n[Agent 2] Phase 2: building NLI dependency graph ...")
        plan = self._build_nli_graph(summaries)
        print(f"[Agent 2] Execution order: {plan['order']}")

        return {
            "status"    : "success",
            "order"     : plan["order"],
            "dependency": plan["map"],
            "summaries" : summaries
        }